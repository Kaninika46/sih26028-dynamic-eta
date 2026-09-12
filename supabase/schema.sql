-- =====================================================================
-- RUNTIME - Supabase schema (run once in Supabase > SQL Editor > New query)
--
-- Roles:  passenger  - passenger view only
--         station    - station display board (+ passenger view)
--         controller - control room, analytics, overrides, cascade, audit
--         admin      - everything + can change roles
-- Everyone who signs up becomes 'passenger'. Only an admin (or the SQL editor)
-- can promote someone. Row-level security (RLS) enforces this in the database,
-- so even a hacked browser cannot read the audit log or write overrides.
-- =====================================================================

create type public.app_role as enum ('passenger', 'station', 'controller', 'admin');

-- ---------- profiles: one row per auth user ------------------------------
create table public.profiles (
  id            uuid primary key references auth.users (id) on delete cascade,
  full_name     text,
  role          public.app_role not null default 'passenger',
  station_code  text,                -- for station terminals, e.g. 'JP'
  employee_id   text,                -- for staff, e.g. 'NWR-8841'
  created_at    timestamptz not null default now()
);
alter table public.profiles enable row level security;

-- the role of the logged-in user (security definer avoids RLS recursion)
create or replace function public.my_role() returns public.app_role
language sql stable security definer set search_path = public as $$
  select role from public.profiles where id = auth.uid()
$$;

-- new sign-ups always start as passenger (role is never taken from user input)
create or replace function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, full_name)
  values (new.id, coalesce(new.raw_user_meta_data ->> 'full_name', split_part(new.email, '@', 1)));
  return new;
end $$;
create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();

create policy "read own profile"      on public.profiles for select using (id = auth.uid());
create policy "admins read profiles"  on public.profiles for select using (public.my_role() = 'admin');
create policy "admins change roles"   on public.profiles for update using (public.my_role() = 'admin');
-- no update policy for normal users: nobody can promote themselves

-- ---------- stations and predictions (own copy of the model output) -------
create table public.stations (
  code      text primary key,
  name      text not null,
  lat       double precision,
  lon       double precision,
  n_trains  int
);
alter table public.stations enable row level security;
create policy "stations are public" on public.stations for select using (true);

create table public.eta_predictions (
  train_no            int  not null,
  journey_date        date not null,
  station_sequence    int  not null,
  station_code        text references public.stations (code),
  train_name          text,
  current_delay_min   real,
  pred_delay_min      real,     -- fused predicted delay at the final station
  low_min             real,     -- 80% range
  high_min            real,
  eta                 timestamptz,
  updated_at          timestamptz not null default now(),
  primary key (train_no, journey_date, station_sequence)
);
alter table public.eta_predictions enable row level security;
create policy "predictions are public" on public.eta_predictions for select using (true);
-- writes only via the server's secret key (bypasses RLS), never from browsers

-- ---------- overrides: controller-reported incidents ----------------------
create table public.overrides (
  id            bigint generated always as identity primary key,
  created_at    timestamptz not null default now(),
  created_by    uuid not null default auth.uid() references auth.users (id),
  train_no      int  not null,
  incident_type text not null,
  location      text,
  severity      text not null,
  note          text,
  bump_min      real not null check (bump_min between 0 and 240),
  active        boolean not null default true
);
alter table public.overrides enable row level security;
create policy "controllers read overrides"  on public.overrides for select
  using (public.my_role() in ('controller', 'admin'));
create policy "controllers add overrides"   on public.overrides for insert
  with check (public.my_role() in ('controller', 'admin') and created_by = auth.uid());
create policy "controllers close overrides" on public.overrides for update
  using (public.my_role() in ('controller', 'admin'));

-- ---------- audit log: who did what (append-only) -------------------------
create table public.audit_log (
  id          bigint generated always as identity primary key,
  created_at  timestamptz not null default now(),
  user_id     uuid not null default auth.uid() references auth.users (id),
  role        public.app_role,
  action      text not null,
  train_no    int,
  detail      jsonb
);
alter table public.audit_log enable row level security;
create policy "staff write audit" on public.audit_log for insert
  with check (public.my_role() in ('station', 'controller', 'admin') and user_id = auth.uid());
create policy "controllers read audit" on public.audit_log for select
  using (public.my_role() in ('controller', 'admin'));
-- no update/delete policies: the audit log cannot be edited from the app

-- ---------- make a user staff (run in SQL editor, replace the email) -------
-- update public.profiles set role = 'controller', employee_id = 'NWR-8841'
--   where id = (select id from auth.users where email = 'controller@example.com');

-- =====================================================================
-- Train movement storage (own database for train data)
-- =====================================================================
create table public.train_movements (
  train_no                      int  not null,
  journey_date                  date not null,
  station_sequence              int  not null,
  station_code                  text,
  station_name                  text,
  train_name                    text,
  scheduled_departure           timestamptz,
  actual_departure              timestamptz,
  current_delay_min             real,
  scheduled_remaining_time_min  real,
  actual_remaining_time_min     real,
  remaining_delay_min           real,
  distance_travelled_km         real,
  distance_remaining_km         real,
  latitude                      double precision,
  longitude                     double precision,
  source                        text not null default 'railkit',
  inserted_at                   timestamptz not null default now(),
  primary key (train_no, journey_date, station_sequence)
);
create index train_movements_date_idx on public.train_movements (journey_date);
create index train_movements_station_idx on public.train_movements (station_code);
alter table public.train_movements enable row level security;
create policy "staff read movements" on public.train_movements for select
  using (public.my_role() in ('station', 'controller', 'admin'));

create table public.live_positions (
  id             bigint generated always as identity primary key,
  captured_at    timestamptz not null default now(),
  train_no       int  not null,
  train_name     text,
  latitude       double precision,
  longitude      double precision,
  station_code   text,
  next_station   text,
  delay_min      real,
  speed_kmh      real,
  source         text not null default 'railradar'
);
create index live_positions_time_idx on public.live_positions (captured_at desc);
create index live_positions_train_idx on public.live_positions (train_no, captured_at desc);
alter table public.live_positions enable row level security;
create policy "staff read live positions" on public.live_positions for select
  using (public.my_role() in ('station', 'controller', 'admin'));
-- writes to both tables happen server-side with the secret key, never from a browser
