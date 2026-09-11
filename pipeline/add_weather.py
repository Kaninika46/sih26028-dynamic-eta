"""
Weather from Open-Meteo for every station and hour of the dataset period.

  * archive API (ERA5) for temperature, humidity, precipitation, rain, weather code, wind,
    cloud cover, surface pressure
  * forecast API with past_days for visibility (not in the archive; covers the last 92 days)
All 491 stations go in batches of 25 locations per request (~40 requests), cached forever.

Adds the weather columns to data/master_40_clean.csv (weather at the station and hour of
each actual departure) and saves data/weather_hourly.csv, which app/server.py uses for the
UI's weather card (condition, temperature, visibility).

Run:  python -m pipeline.add_weather
"""
import pandas as pd

from config import CLEAN_CSV, DATA, STATIONS_CSV, WEATHER_COLS
from pipeline.prepare_data import parse_any
from sources import openmeteo

HOURLY_CSV = DATA / "weather_hourly.csv"


def main():
    st = pd.read_csv(STATIONS_CSV)
    df = pd.read_csv(CLEAN_CSV)
    jd = pd.to_datetime(df.journey_date, format="%d-%m-%Y")
    dep = parse_any(df.actual_departure, jd)
    start, end = dep.min().date(), (dep.max() + pd.Timedelta(days=1)).date()
    print(f"{len(st)} stations, {start} to {end}")

    w = openmeteo.hourly(st, start, end)
    try:
        today = pd.Timestamp.today().normalize()
        vis = openmeteo.recent_hourly(st, ["visibility"], past_days=(today - pd.Timestamp(start)).days + 1)
        vis["visibility_km"] = vis.pop("visibility") / 1000.0
        w = w.merge(vis[["time", "station_code", "visibility_km"]], on=["time", "station_code"], how="left")
    except Exception as e:
        print("visibility not available:", e)
        w["visibility_km"] = pd.NA
    w.to_csv(HOURLY_CSV, index=False)

    df["actual_departure_ts"] = dep
    df = openmeteo.merge_weather(df, w.drop(columns=["visibility_km"]))
    df = df.drop(columns=["actual_departure_ts"])
    df.to_csv(CLEAN_CSV, index=False)
    cov = df[WEATHER_COLS[0]].notna().mean()
    print(f"saved {HOURLY_CSV.name} ({len(w)} station-hours); weather on {cov:.1%} of dataset rows")


if __name__ == "__main__":
    main()
