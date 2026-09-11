"""
RailRadar client (https://railradar.in/docs).

  live_train(no)  GET /v1/trains/{number}/live?includeCoordinates=true
                  delay, current station, segmentProgress (0-1), speed, route with lat/lng
  live_map()      GET /v1/legacy/trains/live-map
                  ONE call -> every running train with current/next station lat/lng
                  (this is what DBSCAN uses in live mode)

Auth: Authorization: Bearer <RAILRADAR_KEY>.  Free sandbox = 1,000 requests/month,
so responses are cached (live train 2 min, live map 5 min) and calls are counted
(see sources.http.used("railradar")).
"""
import numpy as np
import pandas as pd

from config import CORRIDOR_BBOX, RAILRADAR_KEY
from sources.http import get_json

BASE = "https://api.railradar.in"


def _get(path, params=None, ttl_s=120):
    if not RAILRADAR_KEY:
        raise RuntimeError("RAILRADAR_KEY is not set (put it in .env)")
    js = get_json(BASE + path, provider="railradar", params=params, ttl_s=ttl_s,
                  headers={"Authorization": f"Bearer {RAILRADAR_KEY}"})
    if js is None or not js.get("success", False):
        raise RuntimeError(f"RailRadar error: {(js or {}).get('error')}")
    return js["data"]


def live_train(train_no, date=None):
    params = {"includeCoordinates": "true"}
    if date:
        params["date"] = date                       # YYYY-MM-DD
    return _get(f"/v1/trains/{train_no}/live", params, ttl_s=120)


def live_map():
    return _get("/v1/legacy/trains/live-map", ttl_s=300)


# ----------------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------------
def live_state(d):
    """Compact state for the ETA models + map from a live_train() response."""
    route = pd.DataFrame(d.get("route", []))
    cur = d.get("currentLocation") or {}
    out = {"train_no": d.get("trainNumber"), "train_name": d.get("trainName"),
           "delay_min": d.get("delayMinutes"), "status": d.get("status"),
           "last_updated": d.get("lastUpdatedAt"), "station_code": cur.get("stationCode"),
           "progress": cur.get("segmentProgress"), "speed_kmh": cur.get("speedKmh"),
           "is_actual_position": cur.get("isActualPosition"), "lat": None, "lon": None,
           "next_station": (d.get("nextHalt") or {}).get("stationCode")}
    if len(route) and "lat" in route and cur.get("stationCode") in set(route.get("stationCode", [])):
        i = route.index[route.stationCode == cur["stationCode"]][0]
        j = min(i + 1, len(route) - 1)
        p = float(cur.get("segmentProgress") or 0.0)
        if cur.get("status") != "departed":
            p = 0.0
        out["lat"] = float(route.lat[i] + p * (route.lat[j] - route.lat[i]))
        out["lon"] = float(route.lng[i] + p * (route.lng[j] - route.lng[i]))
    return out


def route_table(d):
    """Station-by-station table (scheduled / actual) from a live_train() response."""
    r = pd.DataFrame(d.get("route", []))
    for c in ["scheduledArrival", "scheduledDeparture", "actualArrival", "actualDeparture"]:
        if c in r:
            r[c] = pd.to_datetime(r[c], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return r


def snapshot_points(data, bbox=CORRIDOR_BBOX):
    """live_map() -> DataFrame of trains inside the corridor bounding box."""
    df = pd.DataFrame(data)
    if df.empty:
        return df
    s, w, n, e = bbox
    df = df[(df.current_lat.between(s, n)) & (df.current_lng.between(w, e))].copy()
    df = df.rename(columns={"train_number": "train_no", "current_lat": "lat", "current_lng": "lon",
                            "current_station": "station_code", "next_station": "next_station_code"})
    return df[["train_no", "train_name", "type", "lat", "lon", "station_code", "next_station_code",
               "curr_distance", "next_distance", "mins_since_dep"]].reset_index(drop=True)
