"""
Tiny Supabase client (Auth + PostgREST over HTTPS, only `requests`).

Keys (Supabase > Project Settings > API):
  SUPABASE_URL            https://<project>.supabase.co
  SUPABASE_PUBLISHABLE_KEY  publishable / anon key  - safe in the browser
  SUPABASE_SECRET_KEY       secret / service_role key - SERVER ONLY, never in the browser

The server verifies every access token with Supabase (GET /auth/v1/user) and reads the
user's role from public.profiles. Writes made on behalf of a user are sent with THAT
user's token, so the database's row-level security checks them a second time.
"""
import config  # noqa: F401  (loads .env before the keys are read)
import hashlib
import os
import time

import requests

URL = os.getenv("SUPABASE_URL", "").rstrip("/")
PUBLISHABLE = os.getenv("SUPABASE_PUBLISHABLE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))
SECRET = os.getenv("SUPABASE_SECRET_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
TIMEOUT = 10
_cache = {}             # token hash -> (expires, user dict)


class AuthError(Exception):
    def __init__(self, msg, status=401):
        super().__init__(msg)
        self.status = status


def enabled():
    return bool(URL and PUBLISHABLE)


def _headers(key, token=None):
    h = {"apikey": key, "Content-Type": "application/json"}
    bearer = token or (key if key.startswith("eyJ") else None)   # legacy JWT keys also go in Authorization
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    return h


def verify(token):
    """Access token -> {"id", "email", "role", "station_code", "full_name", "employee_id"}.
    Raises AuthError. Results are cached for 60 s to keep the UI fast."""
    if not token:
        raise AuthError("not signed in")
    k = hashlib.sha256(token.encode()).hexdigest()
    hit = _cache.get(k)
    if hit and hit[0] > time.time():
        return hit[1]
    r = requests.get(f"{URL}/auth/v1/user", headers=_headers(PUBLISHABLE, token), timeout=TIMEOUT)
    if r.status_code != 200:
        raise AuthError("session expired or invalid - please sign in again")
    u = r.json()
    prof = select("profiles", {"id": f"eq.{u['id']}", "select": "role,station_code,full_name,employee_id"},
                  token=token)
    if not prof:
        raise AuthError("no profile for this account", 403)
    user = {"id": u["id"], "email": u.get("email"), **prof[0]}
    _cache[k] = (time.time() + 60, user)
    return user


def select(table, params, token=None):
    """Read rows. With a user token RLS applies; without one the secret key is used."""
    key = PUBLISHABLE if token else SECRET
    r = requests.get(f"{URL}/rest/v1/{table}", params=params, headers=_headers(key, token), timeout=TIMEOUT)
    if r.status_code >= 400:
        raise AuthError(f"database refused read of {table}: {r.text[:200]}", 403 if r.status_code in (401, 403) else 502)
    return r.json()


def insert(table, rows, token=None, upsert=False):
    """Insert rows (as the user when token is given, so RLS checks the permission)."""
    key = PUBLISHABLE if token else SECRET
    h = _headers(key, token)
    h["Prefer"] = "return=representation" + (",resolution=merge-duplicates" if upsert else "")
    r = requests.post(f"{URL}/rest/v1/{table}", json=rows, headers=h, timeout=30)
    if r.status_code >= 400:
        raise AuthError(f"database refused write to {table}: {r.text[:200]}", 403 if r.status_code in (401, 403) else 502)
    return r.json()


def update(table, match, values, token=None):
    key = PUBLISHABLE if token else SECRET
    h = _headers(key, token)
    h["Prefer"] = "return=representation"
    r = requests.patch(f"{URL}/rest/v1/{table}", params=match, json=values, headers=h, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise AuthError(f"database refused update of {table}: {r.text[:200]}", 403)
    return r.json()


def admin_create_user(email, password, full_name=None):
    """Create a confirmed user with the secret key (Auth admin API)."""
    if not SECRET:
        raise AuthError("SUPABASE_SECRET_KEY is required to create users", 500)
    r = requests.post(f"{URL}/auth/v1/admin/users", headers=_headers(SECRET),
                      json={"email": email, "password": password, "email_confirm": True,
                            "user_metadata": {"full_name": full_name or email.split("@")[0]}},
                      timeout=TIMEOUT)
    if r.status_code >= 400:
        raise AuthError(f"could not create user: {r.text[:200]}", r.status_code)
    return r.json()
