"""
Clean the team's 40-train master dataset for the Kalman / DBSCAN / GCN branches.

Fixes found in the raw file:
  * two time formats mixed ('2026-08-01 06:00:00' and '16:29 01-Aug', sometimes with '*')
  * 300 rows at 13 stations with latitude/longitude missing or 0,0 -> filled from the same
    station on other trains, else interpolated along the route by scheduled time
  * distances up to 36,000 km caused by those 0,0 coordinates -> recomputed as cumulative
    great-circle distance along each journey
Adds: dep_ts, sched_dep_ts, remaining_delay_min (= actual - scheduled remaining time).
Keys (train_no, journey_date, station_sequence) are kept so rows line up with the
team's XGBoost feature file.

Run:  python -m pipeline.prepare_data
"""
import numpy as np
import pandas as pd

from config import CLEAN_CSV, MASTER_RAW, STATIONS_CSV

KEY = ["train_no", "journey_date", "station_sequence"]


def parse_any(col, journey_date):
    s = col.astype(str).str.replace("*", "", regex=False).str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    short = pd.to_datetime(s + "-" + journey_date.dt.year.astype(str), format="%H:%M %d-%b-%Y", errors="coerce")
    wrap = short < journey_date - pd.Timedelta(days=2)
    short[wrap] = short[wrap] + pd.DateOffset(years=1)
    return iso.fillna(short)


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def main():
    df = pd.read_csv(MASTER_RAW)
    df["journey_date"] = pd.to_datetime(df["journey_date"], format="%d-%m-%Y")
    df["dep_ts"] = parse_any(df["actual_departure"], df["journey_date"])
    df["sched_dep_ts"] = parse_any(df["scheduled_departure"], df["journey_date"])
    print("unparsed times:", int(df.dep_ts.isna().sum()), "actual,", int(df.sched_dep_ts.isna().sum()), "scheduled")
    df = df.dropna(subset=["dep_ts", "sched_dep_ts"]).sort_values(KEY).reset_index(drop=True)

    # ---- coordinates
    bad = df.latitude.isna() | (df.latitude.abs() < 1) | (df.longitude.abs() < 1)
    df.loc[bad, ["latitude", "longitude"]] = np.nan
    good = df.dropna(subset=["latitude"]).groupby("station_code")[["latitude", "longitude"]].median()
    df["latitude"] = df.latitude.fillna(df.station_code.map(good.latitude))
    df["longitude"] = df.longitude.fillna(df.station_code.map(good.longitude))
    n_before = df.latitude.isna().sum()
    t = df.sched_dep_ts.astype("int64") / 6e10
    for c in ("latitude", "longitude"):
        df[c] = (df.assign(_t=t).groupby(["train_no", "journey_date"], group_keys=False)
                 .apply(lambda g: g[c].interpolate().ffill().bfill()))
    # a station should have one position: use the median over all its rows
    pos = df.groupby("station_code")[["latitude", "longitude"]].median()
    df["latitude"], df["longitude"] = df.station_code.map(pos.latitude), df.station_code.map(pos.longitude)
    print(f"coordinates fixed: {int(bad.sum())} bad rows, {int(n_before)} needed route interpolation")

    # ---- distances
    g = df.groupby(["train_no", "journey_date"])
    step = haversine(g.latitude.shift(), g.longitude.shift(), df.latitude, df.longitude).fillna(0)
    df["distance_travelled_km"] = step.groupby([df.train_no, df.journey_date]).cumsum().round(2)
    total = df.groupby(["train_no", "journey_date"]).distance_travelled_km.transform("max")
    df["distance_remaining_km"] = (total - df.distance_travelled_km).round(2)

    df["remaining_delay_min"] = df.actual_remaining_time_min - df.scheduled_remaining_time_min
    df = df.dropna(subset=["remaining_delay_min", "current_delay_min"])
    out = df.copy()
    out["journey_date"] = out.journey_date.dt.strftime("%d-%m-%Y")
    out.to_csv(CLEAN_CSV, index=False)

    st = (df.groupby("station_code").agg(station_name=("station_name", "first"), lat=("latitude", "first"),
                                         lon=("longitude", "first"), n_trains=("train_no", "nunique"))
          .reset_index())
    st.to_csv(STATIONS_CSV, index=False)
    print(f"saved {CLEAN_CSV.name}: {len(df)} rows, {df.train_no.nunique()} trains, "
          f"{df.groupby(['train_no', 'journey_date']).ngroups} journeys, {len(st)} stations")
    print("max distance now:", round(df.distance_travelled_km.max(), 1), "km")


if __name__ == "__main__":
    main()
