"""
Cascading propagation engine (network version).

"If train A is delayed N minutes at station S, which following trains are pushed back?"

Rule: two trains leaving the same station for the same next station (same section, same
direction) must be at least HEADWAY_MIN apart. When a push happens, the rest of that train's
journey moves by the same amount. Passes repeat until nothing changes (max 10), so delays
cascade along lines and through shared junctions.

    from models.cascade import plan_for_day, propagate
    plan = plan_for_day(df, "2026-08-30")
    impact = propagate(plan, 12468, "JP", 40)
"""
import numpy as np
import pandas as pd

from config import HEADWAY_MIN
from models.common import JOURNEY


def plan_for_day(df, day):
    day = pd.Timestamp(day)
    d = df[(df.dep_ts >= day) & (df.dep_ts < day + pd.Timedelta(days=1))].copy()
    d = d.sort_values(JOURNEY + ["station_sequence"])
    d["next_station"] = d.groupby(JOURNEY).station_code.shift(-1)
    d["t"] = d.dep_ts.values.astype("datetime64[m]").astype("int64").astype(float)
    return d[JOURNEY + ["train_name", "station_sequence", "station_code", "next_station", "t"]].reset_index(drop=True)


def propagate(plan, train_no, station_code, delay_min, headway=HEADWAY_MIN, max_pass=10):
    p = plan.copy()
    p["new_t"] = p["t"]
    me = p[(p.train_no == int(train_no)) & (p.station_code == station_code)]
    if me.empty:
        raise ValueError(f"train {train_no} does not leave {station_code} on this day")
    r0 = me.iloc[0]
    hit = (p.train_no == r0.train_no) & (p.journey_date == r0.journey_date) & (p.station_sequence >= r0.station_sequence)
    p.loc[hit, "new_t"] += delay_min
    for _ in range(max_pass):
        changed = False
        for _, g in p.dropna(subset=["next_station"]).groupby(["station_code", "next_station"]):
            if len(g) < 2:
                continue
            g = g.sort_values("new_t")
            last = last_orig = None
            for idx, r in g.iterrows():
                cur = p.at[idx, "new_t"]
                if last is not None:
                    # push only by the part of the conflict that the change created (real data can
                    # already contain trains closer than the headway, e.g. on parallel lines)
                    orig_gap_short = max(0.0, last_orig + headway - r.t)
                    push = (last + headway - cur) - orig_gap_short
                    if push > 0.05:
                        later = ((p.train_no == r.train_no) & (p.journey_date == r.journey_date)
                                 & (p.station_sequence >= r.station_sequence))
                        p.loc[later, "new_t"] += push
                        changed = True
                        cur += push
                last, last_orig = cur, r.t
        if not changed:
            break
    p["added_min"] = (p.new_t - p.t).round(1)
    imp = (p.groupby(JOURNEY).agg(train_name=("train_name", "first"), added_delay_min=("added_min", "max"),
                                  first_hit_station=("added_min", lambda s: p.station_code[s.gt(0.05).idxmax()]
                                                     if s.gt(0.05).any() else None))
           .reset_index())
    imp = imp[imp.added_delay_min > 0.05].sort_values("added_delay_min", ascending=False)
    imp["journey_date"] = imp.journey_date.dt.strftime("%d-%m-%Y")
    return imp.reset_index(drop=True)
