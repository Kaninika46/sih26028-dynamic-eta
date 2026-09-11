"""
RailKit client (https://railkit.in).

Two ways to call it:
  RAILKIT_MODE=rest -> direct REST (https://api.railkit.in, header x-api-key).
                       RailKit's docs say direct REST needs their Advance plan.
  RAILKIT_MODE=sdk  -> the official Node.js SDK through sources/node/railkit_bridge.mjs
                       (works on the free tier). One-time setup:
                           cd sources/node && npm install

Endpoints used (paths from RailKit's public docs):
  history : /api/v1/trains/{no}/history/{DD-MM-YYYY}  (completed run, per-stop actuals)
  live    : /api/v1/trains/{no}/live/{DD-MM-YYYY|today}
  station : /api/v1/stations/{code}                   (name + coordinates)
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAILKIT_KEY, RAILKIT_MODE
from sources.http import get_json

BASE = "https://api.railkit.in"
BRIDGE = Path(__file__).resolve().parent / "node" / "railkit_bridge.mjs"


class RailKitError(RuntimeError):
    pass


def _rest(path, ttl_s):
    if not RAILKIT_KEY:
        raise RailKitError("RAILKIT_KEY is not set (put it in .env)")
    return get_json(BASE + path, provider="railkit",
                    headers={"x-api-key": RAILKIT_KEY, "accept": "application/json"}, ttl_s=ttl_s)


def _sdk(fn, *args, ttl_s=None):
    """Calls the official SDK via Node and caches the JSON like the REST path."""
    from sources.http import CACHE
    import hashlib, time
    key = hashlib.sha1(json.dumps([fn, args]).encode()).hexdigest()
    cache = CACHE / f"railkit_sdk_{key}.json"
    if cache.exists() and (ttl_s is None or time.time() - cache.stat().st_mtime < ttl_s):
        return json.loads(cache.read_text())
    if not RAILKIT_KEY:
        raise RailKitError("RAILKIT_KEY is not set (put it in .env)")
    if not (BRIDGE.parent / "node_modules" / "railkit").exists():
        raise RailKitError("RailKit SDK not installed: run `cd sources/node && npm install`")
    p = subprocess.run(["node", str(BRIDGE), fn, *args], capture_output=True, text=True, timeout=60,
                       env={**__import__("os").environ, "RAILKIT_KEY": RAILKIT_KEY})
    if p.returncode != 0:
        raise RailKitError(p.stderr.strip() or "node bridge failed")
    js = json.loads(p.stdout)
    from sources.http import _count
    _count("railkit")
    if js.get("success"):
        cache.write_text(json.dumps(js))
    return js


def _call(fn, rest_path, *args, ttl_s=None):
    js = _rest(rest_path, ttl_s) if RAILKIT_MODE == "rest" else _sdk(fn, *args, ttl_s=ttl_s)
    if js is None:
        return None
    if not js.get("success", False):
        err = js.get("error") or js.get("message") or "unknown error"
        if "not" in str(err).lower() and ("found" in str(err).lower() or "complete" in str(err).lower()):
            return None
        raise RailKitError(str(err))
    return js["data"]


def history(train_no, date_ddmmyyyy):
    """Completed journey. Returns None if the run is not completed / not found."""
    return _call("history", f"/api/v1/trains/{train_no}/history/{date_ddmmyyyy}",
                 str(train_no), date_ddmmyyyy, ttl_s=None)          # history never changes


def live(train_no, date_ddmmyyyy="today"):
    return _call("live", f"/api/v1/trains/{train_no}/live/{date_ddmmyyyy}",
                 str(train_no), date_ddmmyyyy, ttl_s=120)


def station(code):
    return _call("station", f"/api/v1/stations/{code.upper()}", code.upper(), ttl_s=None)


def station_coords(code):
    """(lat, lon) from the station lookup, searching the response for coordinate keys."""
    d = station(code) or {}
    flat = json.dumps(d)
    lat = re.search(r'"(?:lat|latitude)"\s*:\s*"?(-?\d+\.\d+)', flat)
    lon = re.search(r'"(?:lng|lon|long|longitude)"\s*:\s*"?(-?\d+\.\d+)', flat)
    return (float(lat[1]), float(lon[1])) if lat and lon else (None, None)


# ----------------------------------------------------------------------------
# history JSON -> dataset rows (same columns as the team's original dataset)
# ----------------------------------------------------------------------------
_T = re.compile(r"(\d{1,2}):(\d{2})\s+(\d{1,2})-([A-Za-z]{3})")


def parse_rk_time(s, journey_date):
    """'16:50 11-Jun' / '16:50 11-Jun*' -> Timestamp ('SRC'/'DSTN'/'' -> NaT)."""
    if not isinstance(s, str):
        return pd.NaT
    m = _T.search(s)
    if not m:
        return pd.NaT
    ts = pd.to_datetime(f"{m[3]}-{m[4]}-{journey_date.year} {m[1]}:{m[2]}", format="%d-%b-%Y %H:%M")
    if ts < journey_date - pd.Timedelta(days=2):          # Dec -> Jan wrap
        ts += pd.DateOffset(years=1)
    return ts


def _fmt(ts):
    return ts.strftime("%H:%M %d-%b") if pd.notna(ts) else None


def history_to_rows(data, corridor=None):
    """data: RailKit history 'data' object. corridor: ordered list of station codes to keep
    (target is measured to the last kept station). Returns a DataFrame or None."""
    jd = pd.to_datetime(data["journeyDate"], format="%d-%m-%Y")
    st = []
    for s in data.get("stations", []):
        code = s.get("stationCode")
        if corridor is not None and code not in corridor:
            continue
        arr, dep = s.get("arrival", {}) or {}, s.get("departure", {}) or {}
        st.append(dict(code=code, name=s.get("stationName"),
                       km=pd.to_numeric(s.get("distanceKm"), errors="coerce"),
                       sa=parse_rk_time(arr.get("scheduled"), jd), aa=parse_rk_time(arr.get("actual"), jd),
                       sd=parse_rk_time(dep.get("scheduled"), jd), ad=parse_rk_time(dep.get("actual"), jd)))
    st = pd.DataFrame(st)
    if len(st) < 3:
        return None
    if pd.isna(st["km"].iloc[0]):                       # source station has no distance
        st.loc[st.index[0], "km"] = 0.0
    st["km"] = st["km"].interpolate().ffill()
    last = st.iloc[-1]
    exit_sched = last.sa if pd.notna(last.sa) else last.sd
    exit_act = last.aa if pd.notna(last.aa) else last.ad
    if pd.isna(exit_sched) or pd.isna(exit_act):
        return None
    rows, prev_ad = [], pd.NaT
    for i, s in st.iloc[:-1].iterrows():
        if pd.isna(s.sd) or pd.isna(s.ad):
            continue
        rows.append({
            "train_no": int(data["trainNo"]), "train_name": data.get("trainName"),
            "journey_date": jd.strftime("%d-%m-%Y"), "station_code": s.code, "station_name": s["name"],
            "station_sequence": len(rows),
            "distance_travelled_km": float(s.km - st.km.iloc[0]),
            "distance_remaining_km": float(last.km - s.km),
            "scheduled_departure": _fmt(s.sd), "actual_departure": _fmt(s.ad),
            "current_delay_min": (s.ad - s.sd).total_seconds() / 60,
            "scheduled_remaining_time_min": (exit_sched - s.sd).total_seconds() / 60,
            "previous_segment_time_min": (s.ad - prev_ad).total_seconds() / 60 if pd.notna(prev_ad) else np.nan,
            "hour": s.ad.hour, "day_of_week": jd.dayofweek,
            "actual_remaining_time_min": (exit_act - s.ad).total_seconds() / 60,
        })
        prev_ad = s.ad
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["remaining_delay_min"] = df["actual_remaining_time_min"] - df["scheduled_remaining_time_min"]
    return df
