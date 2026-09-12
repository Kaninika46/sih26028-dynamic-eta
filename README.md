# RUNTIME: train ETA intelligence on 40 real trains (SIH)

Real RailKit running data for 40 trains (August 2026, 897 journeys, 491 stations), the
XGBoost **v2 retrained on all 40 trains**, plus 1D Kalman, DBSCAN, 2D Kalman, GCN-LSTM, inverse-MAE
fusion and a cascade engine, served to the team's web UI.

## Run

```bash
pip install -r requirements.txt
python app/server.py              # models are already trained; open http://localhost:8000
python run_all.py                 # retrain everything from data/master_40_trains.csv (10-15 min)
```

The UI (`app/ui/index.html`) is the team's design with four small additions: draw the route
along real OSM track, move the marker along it, and keep the server's train progress. It calls `/api/trains` and
`/api/coords`; `app/server.py` answers with real model output. If the server is not running
the page falls back to its built-in demo trains.

The server replays a held-out test day (default 30 Aug) at the current time of day, so the
trains on screen move as the clock moves. `REPLAY_DATE=2026-08-29 python app/server.py`
picks another day; `/api/trains?t=18:30` picks a time. `/api/cascade?train=14662&station=JP&delay=30`
returns the trains pushed back by that delay.

## Login, roles and own database (Supabase)

| Role | Screens | Can do |
|---|---|---|
| guest / passenger | Passenger | view ETAs |
| station | Station Display, Passenger | log board events |
| controller | Control Room, Station Display, Passenger, Analytics | overrides, cascade, audit log |
| admin | everything | + change roles |

Security is enforced in three layers: the UI hides screens a role may not open; the server
(`app/server.py`) checks the Supabase token and role on every privileged request; and
row-level security in the database (`supabase/schema.sql`) refuses writes/reads even if
someone calls Supabase directly. Nobody can promote themselves: sign-ups are always
`passenger`, and staff accounts are created with a script.

Setup (about 10 minutes):
1. Create a project at supabase.com. In **SQL Editor**, paste and run `supabase/schema.sql`.
2. **Project Settings > API**: copy the URL, the publishable (anon) key and the secret
   (service_role) key into `.env` (see `.env.example`). The secret key stays on the server.
3. **Authentication > Providers > Email**: enabled. For a demo you can turn off
   "Confirm email" so passenger sign-ups work instantly.
4. Create staff accounts:
   ```
   python -m pipeline.create_user controller@nwr.in "StrongPass#1" controller --employee NWR-8841
   python -m pipeline.create_user jp.board@nwr.in "StrongPass#2" station --station JP
   ```
5. Copy stations, movements and predictions into the database: `python -m pipeline.sync_supabase`
   (or one table at a time: `python -m pipeline.sync_supabase --only movements`)
6. `python app/server.py` - the login screens now ask for email + password.

Tables: `profiles` (role), `stations`, `eta_predictions`, `train_movements` (every recorded
departure of the 40 trains - the movement history), `live_positions` (RailRadar snapshots),
`overrides` (controller incidents, applied to everyone's ETAs for 6 h), `audit_log`
(append-only). Movement tables are readable by staff only; both are written server-side with
the secret key.
New API: `GET /api/config`, `GET /api/me`, `POST /api/override`, `GET /api/audit`,
`POST /api/log`; `/api/cascade` now needs a controller.

Without Supabase keys the server runs in **demo mode** (no login, all screens open) and
prints a warning - useful as an offline backup for the presentation.

Tests: `python tests/test_auth.py` runs the real server against a fake Supabase with the same
rules (18 checks: passengers/stations blocked from control endpoints, controllers allowed,
overrides visible to passengers without staff identity).

## Weather (Open-Meteo) and track paths (OpenStreetMap)

Both need internet once; results are cached in `data/cache/`.

```bash
python -m pipeline.add_weather       # ~40 requests: hourly weather for all 491 stations
python -m pipeline.build_osm_paths   # ~21 Overpass requests: real rail geometry for 816 sections
python app/server.py
```

- **Weather:** Open-Meteo archive (temperature, rain, weather code, wind, cloud, pressure) plus
  visibility from the forecast API's `past_days`. Added to every dataset row and shown in the
  UI's weather card for the train's current station (clear / cloudy / rain / heavy rain /
  fog / dense fog, temperature, visibility).
- **Paths:** for each pair of consecutive stations, OSM `railway=rail` ways near both stations
  are joined into a graph and the shortest path along the rails is taken. Sections with no
  connected OSM track within 2 km, or an implausibly long path, fall back to a straight line.
  The UI draws the route along these paths and moves the train marker along the real track.

Without these files the UI still works: no weather card, straight lines between stations.

## Layout

```
XGBOOST/                  first XGBoost package (v1, reference; its split days are reused)
data/master_40_trains.csv teammate's 40-train master dataset
pipeline/prepare_data.py  cleaning: time formats, 0,0 coordinates, distances
pipeline/add_weather.py   Open-Meteo weather per station and hour
pipeline/build_osm_paths.py  OSM rail geometry between consecutive stations
models/xgb_branch.py      XGBoost v2: all 40 trains, delay target, history features, tuned
models/kalman_1d.py       1D Kalman + LiveKalman dead reckoning (Telemetry Guard)
models/dbscan_network.py  station graph, causal positions, DBSCAN, congestion tables
models/kalman_2d.py       2D Kalman on 816 sections, 30-min congestion forecast
models/gcn_lstm.py        GCN-LSTM (PyTorch) on the station graph
models/fusion.py          inverse-MAE weights, sum = 1, per-row re-normalised
models/cascade.py         headway push: who is delayed if train X is late
app/server.py             API for the UI + auth / roles
supabase/schema.sql       tables, roles, row-level security
pipeline/create_user.py   create staff accounts with a role
pipeline/sync_supabase.py copy stations + movements + predictions into Supabase
pipeline/collect_live.py  poll RailRadar live map into Supabase + a local CSV
sources/                  RailKit, RailRadar, OpenStreetMap, Open-Meteo clients (live mode)
tests/                    offline tests
```

## Split

Same as the first XGBoost (70/15/15 of date-sorted rows) turned into whole days, so no
model is tested on rows XGBoost trained on: train up to 23 Aug, validation 24-27 Aug,
test 28-31 Aug.

## Results, test days 28-31 Aug, real data (ETA error in minutes)

| Model | MAE | RMSE | Note |
|---|---|---|---|
| Arrives on schedule | 23.17 | 29.26 | baseline |
| XGBoost v1 (first model) | 12.68 | 18.68 | only 27 trains; ~81 min on the other 13 |
| **XGBoost v2** | **10.63** | 14.87 | all 40 trains (10.50 on the old 27, 10.81 on the 13 added) |
| 1D Kalman | 10.84 | 14.62 | all 40 trains |
| GCN-LSTM | 11.28 | 16.76 | all 40 trains |
| **Fused** | **10.55** | 14.85 | weights XGB 0.35 / Kalman 0.32 / GCN 0.33 |

Fused 80% range contains the true arrival 79.5% of the time.

### What made XGBoost v2 better (`models/xgb_branch.py`)
1. **All 40 trains**: trained on the cleaned master file; v1's feature script dropped 13 trains.
2. **Delay target**: predicts remaining delay, not total remaining time (hundreds of minutes).
3. **History features**: average remaining delay of this train at this station on training
   days (out-of-fold, so no leakage) - now the most important feature.
4. **Running-state features**: delay change since the last station, minutes lost on the last
   section vs schedule, share of the journey done; train and station as categories.
5. **Robust loss + tuning** on validation days (pseudo-Huber, depth 6 was chosen).
6. **Same whole-day split as the other branches**, and quantile models for the 80% range.
Network and weather features are offered too; they are kept only if validation improves
(network did not; weather is offered once `pipeline.add_weather` has run).

## Honest findings (tell the judges before they ask)

- **XGBoost v2 covers all 40 trains.** The first model (kept in `XGBOOST/` and
  `models/xgb_branch_v1_teammate.py` for reference) only knew 27.
- **Fusion now beats every single model, but only just** (10.55 vs 10.63 for XGBoost).
- **The graph does not help yet.** GCN-LSTM with the graph blanked scored 10.88 vs 11.28
  with it. Only 7.7% of rows have another train at the same station within 30 minutes, so
  there is little network interaction to learn. Same reason the 2D Kalman barely beats
  "no congestion" (MSE 0.196 vs 0.200; persistence 0.348) and its neighbour coupling is ~0.
  More trains on shared sections (e.g. all Jaipur-Phulera traffic) are needed.
- DBSCAN finds clusters in 5.3% of train positions; cascades are real but small
  (e.g. 14662 +30 min at Jaipur pushes 14801 by 4 min).
- Weather is always shown in the UI. XGBoost v2 uses it only if it improves validation
  error after `pipeline.add_weather` has run.
- The weather and OSM steps were tested with simulated API responses; they could not reach
  the real servers from the build environment. Run them once with internet and check the
  printed counts (stations with weather, sections routed on OSM vs straight).
