import argparse
import time
from datetime import datetime, timezone
import pandas as pd

from config import CLEAN_CSV, DATA
from sources import railradar, supa
from sources.http import used

CSV = DATA / "live_positions.csv"


def our_trains():
    return set(pd.read_csv(CLEAN_CSV, usecols=["train_no"]).train_no.astype(int))


def to_rows(pts, ts):
    rows = []
    for p in pts.itertuples():
        rows.append({
            "captured_at": ts,
            "train_no": int(p.train_no),
            "train_name": getattr(p, "train_name", None),
            "latitude": float(p.lat),
            "longitude": float(p.lon),
            "station_code": getattr(p, "station_code", None),
            "next_station": getattr(p, "next_station_code", None),
            "delay_min": None,
            "speed_kmh": None,
            "source": "railradar"
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0, help="minutes between polls")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=900, help="stop when this month's RailRadar calls reach this")
    ap.add_argument("--all-trains", action="store_true", help="keep every train, not only our 40")
    a = ap.parse_args()

    keep = None if a.all_trains else our_trains()
    to_db = supa.enabled() and supa.SECRET
    print(f"storing to {'Supabase + ' if to_db else ''}{CSV.name}; "
          f"RailRadar calls used this month: {used('railradar')}")

    end = time.time() + a.hours * 3600
    while time.time() < end:
        if used("railradar") >= a.budget:
            print("monthly budget reached, stopping")
            break

        ts = datetime.now(timezone.utc).isoformat()
        try:
            pts = railradar.snapshot_points(railradar.live_map())
            if keep is not None and len(pts):
                pts = pts[pts.train_no.astype(int).isin(keep)]
            rows = to_rows(pts, ts) if len(pts) else []

            if rows:
                pd.DataFrame(rows).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
                if to_db:
                    try:
                        supa.insert("live_positions", rows)
                    except Exception as db_err:
                        print(f"[{ts}] Supabase remote sync skipped (local CSV updated): {db_err}")

            print(f"{ts}: {len(rows)} trains stored (calls this month: {used('railradar')})")
        except Exception as e:
            print(f"{ts}: poll failed: {e}")

        time.sleep(a.interval * 60)


if __name__ == "__main__":
    main()