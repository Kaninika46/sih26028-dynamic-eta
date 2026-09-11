"""
Open-Meteo hourly weather (no key needed).

  Historical: https://archive-api.open-meteo.com/v1/archive   (lags ~5 days behind today)
  Recent/forecast: https://api.open-meteo.com/v1/forecast       (past_days up to 92 + forecast)

Several stations are sent in one request (comma-separated coordinates), so a whole
corridor is 1-2 calls. Times are local (Asia/Kolkata) to match the train data.
"""
from datetime import date, timedelta

import pandas as pd

from config import WEATHER_COLS
from sources.http import get_json

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"


def _batch(stations, start, end, url, extra, ttl_s):
    params = {"latitude": ",".join(f"{x:.4f}" for x in stations.lat),
              "longitude": ",".join(f"{x:.4f}" for x in stations.lon),
              "hourly": ",".join(WEATHER_COLS), "timezone": "Asia/Kolkata", **extra}
    if url == ARCHIVE:
        params.update(start_date=str(start), end_date=str(end))
    js = get_json(url, provider="openmeteo", params=params, ttl_s=ttl_s)
    js = js if isinstance(js, list) else [js]
    frames = []
    for code, loc in zip(stations.station_code, js):
        h = pd.DataFrame(loc["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h["station_code"] = code
        frames.append(h)
    return pd.concat(frames)


def hourly(stations, start, end, batch=25):
    """stations: DataFrame(station_code, lat, lon). start/end: date or 'YYYY-MM-DD'.
    Returns long table: time, station_code, <WEATHER_COLS>."""
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    stations = stations.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    today = date.today()
    out = []
    for i in range(0, len(stations), batch):
        part = stations.iloc[i:i + batch]
        arch_end = min(end, today - timedelta(days=6))
        if start <= arch_end:
            out.append(_batch(part, start, arch_end, ARCHIVE, {}, ttl_s=None))
        if end > arch_end:                              # recent days via forecast API
            past = min(92, (today - max(start, arch_end + timedelta(days=1))).days + 1)
            fc = _batch(part, None, None, FORECAST, {"past_days": past, "forecast_days": 2}, ttl_s=3600)
            out.append(fc[(fc.time.dt.date > arch_end) & (fc.time.dt.date <= end + timedelta(days=1))])
    return pd.concat(out).drop_duplicates(["station_code", "time"]).reset_index(drop=True)


def merge_weather(rows, weather_long, time_col="actual_departure_ts"):
    """Attach weather of the row's station at the hour of its actual departure."""
    w = weather_long.assign(hour_ts=weather_long.time.dt.floor("h")).drop(columns="time")
    r = rows.assign(hour_ts=rows[time_col].dt.floor("h"))
    r = r.drop(columns=[c for c in WEATHER_COLS if c in r.columns])
    return r.merge(w, on=["station_code", "hour_ts"], how="left").drop(columns="hour_ts")


def recent_hourly(stations, variables, past_days=92, batch=25):
    """Forecast API with past_days: covers the last 3 months and has variables the ERA5
    archive lacks (e.g. visibility). Returns long table: time, station_code, <variables>."""
    stations = stations.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    out = []
    for i in range(0, len(stations), batch):
        part = stations.iloc[i:i + batch]
        params = {"latitude": ",".join(f"{x:.4f}" for x in part.lat),
                  "longitude": ",".join(f"{x:.4f}" for x in part.lon),
                  "hourly": ",".join(variables), "timezone": "Asia/Kolkata",
                  "past_days": int(min(past_days, 92)), "forecast_days": 1}
        js = get_json(FORECAST, provider="openmeteo", params=params, ttl_s=6 * 3600)
        js = js if isinstance(js, list) else [js]
        for code, loc in zip(part.station_code, js):
            h = pd.DataFrame(loc["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h["station_code"] = code
            out.append(h)
    return pd.concat(out, ignore_index=True)


def weather_kind(code, rain_mm=0.0, visibility_km=None):
    """WMO weather code (+ visibility) -> the UI's categories."""
    code = 0 if code is None or code != code else int(code)
    if visibility_km is not None and visibility_km == visibility_km:
        if visibility_km < 0.2:
            return "dense_fog"
        if visibility_km < 1.0:
            return "fog"
    if code in (45, 48):
        return "fog"
    if code in (65, 67, 82) or code >= 95 or (rain_mm == rain_mm and rain_mm is not None and rain_mm >= 4):
        return "heavy_rain"
    if 51 <= code <= 81:
        return "rain"
    return "clear" if code == 0 else "cloudy"
