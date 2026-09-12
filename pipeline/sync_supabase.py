"""
Copy your data into your own Supabase database.

  stations         491 corridor stations
  eta_predictions  latest fused ETA per train and station
  train_movements  every recorded departure of the 40 trains (the movement history)

Uses the SECRET key (server side only). Tables come from supabase/schema.sql.
Run after the models:   python -m pipeline.sync_supabase
                        python -m pipeline.sync_supabase --only movements
"""
import argparse
import numpy as np
import pandas as pd

from config import OUT, STATIONS_CSV
from models.common import KEY, load_data
from sources import supa


def chunks(rows, n=500):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def sync_movements(df):
    """Historical movements: one row per train, journey date and station departure."""
    cols = ["station_code", "station_name", "train_name", "current_delay_min",
            "scheduled_remaining_time_min", "actual_remaining_time_min", "remaining_delay_min",
            "distance_travelled_km", "distance_remaining_km", "latitude", "longitude"]
    rows = []
    for r in df.itertuples():
        row = {"train_no": int(r.train_no), "journey_date": r.journey_date.strftime("%Y-%m-%d"),
               "station_sequence": int(r.station_sequence),
               "scheduled_departure": r.sched_dep_ts.tz_localize("Asia/Kolkata").isoformat(),
               "actual_departure": r.dep_ts.tz_localize("Asia/Kolkata").isoformat(), "source": "railkit"}
        for c in cols:
            v = getattr(r, c, None)
            row[c] = None if v is None or (isinstance(v, float) and np.isnan(v)) else (
                float(v) if isinstance(v, (int, float, np.floating)) and c not in ("station_code", "station_name",
                                                                                   "train_name") else v)
        rows.append(row)
    for part in chunks(rows):
        supa.insert("train_movements", part, upsert=True)
    print(f"train_movements: {len(rows)} upserted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["stations", "predictions", "movements"], default=None,
                    help="sync just one table (default: all three)")
    a = ap.parse_args()
    if not (supa.enabled() and supa.SECRET):
        raise SystemExit("Set SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY and SUPABASE_SECRET_KEY in .env first")
    only = a.only
    st = pd.read_csv(STATIONS_CSV)
    rows = [{"code": r.station_code, "name": r.station_name, "lat": float(r.lat), "lon": float(r.lon),
             "n_trains": int(r.n_trains)} for r in st.itertuples()]
    if only in (None, "stations"):
        for part in chunks(rows):
            supa.insert("stations", part, upsert=True)
        print(f"stations: {len(rows)} upserted")

    data = load_data()
    if only in (None, "movements"):
        sync_movements(data)
    if only == "movements":
        return

    f = pd.read_csv(OUT / "fused_predictions.csv")
    f["journey_date"] = pd.to_datetime(f.journey_date, format="%d-%m-%Y")
    d = data[KEY + ["train_name", "dep_ts", "current_delay_min", "scheduled_remaining_time_min"]]
    m = f.merge(d, on=KEY, how="left")
    eta = m.dep_ts + pd.to_timedelta(m.scheduled_remaining_time_min + m.fused_pred, unit="m")
    known = set(st.station_code)
    rows = [{"train_no": int(r.train_no), "journey_date": r.journey_date.strftime("%Y-%m-%d"),
             "station_sequence": int(r.station_sequence),
             "station_code": r.station_code if r.station_code in known else None,
             "train_name": r.train_name, "current_delay_min": float(r.current_delay_min),
             "pred_delay_min": float(r.current_delay_min + r.fused_pred),
             "low_min": float(r.current_delay_min + r.fused_low), "high_min": float(r.current_delay_min + r.fused_high),
             "eta": e.tz_localize("Asia/Kolkata").isoformat() if pd.notna(e) else None}
            for r, e in zip(m.itertuples(), eta)]
    for part in chunks(rows):
        supa.insert("eta_predictions", part, upsert=True)
    print(f"eta_predictions: {len(rows)} upserted")


if __name__ == "__main__":
    main()
