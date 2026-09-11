"""
DBSCAN congestion engine for the 40-train network (491 stations, many lines).

Graph      nodes = stations, edges = consecutive station pairs on any train's route.
Positions  every 5 minutes, each running train's causal position: last actual departure
           + scheduled running speed on its current section (no future information).
DBSCAN     haversine on all running trains, eps 5 km, min_samples 2 -> trains bunched together.
Tables     per edge and time: occupancy, clustered trains, stall (minutes trains are overdue
           on that section)  -> input for the 2D Kalman and the GCN-LSTM.
Features   per row (causal): headway_min, trains_last_30min, trains_sched_next_30min,
           dbscan_cluster_size.

Run:  python -m models.dbscan_network
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from config import OUT, STORE
from models.common import JOURNEY, load_data

EPS_KM, MIN_SAMPLES, STEP = 5.0, 2, 5
EARTH_KM = 6371.0
FEATURED_CSV = OUT / "dataset_with_network_features.csv"


def cluster_points(lat, lon, eps_km=EPS_KM, min_samples=MIN_SAMPLES):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if len(lat) < min_samples:
        return -np.ones(len(lat), int), np.ones(len(lat), int)
    lab = DBSCAN(eps=eps_km / EARTH_KM, min_samples=min_samples, metric="haversine").fit_predict(
        np.radians(np.column_stack([lat, lon])))
    size = np.ones(len(lab), int)
    for c in set(lab) - {-1}:
        size[lab == c] = (lab == c).sum()
    return lab, size


def minutes(ts):
    return ts.values.astype("datetime64[m]").astype("int64").astype(float)


def build_graph(df):
    st = (df.groupby("station_code").agg(name=("station_name", "first"), lat=("latitude", "median"),
                                         lon=("longitude", "median")).reset_index())
    node = {c: i for i, c in enumerate(st.station_code)}
    edges = {}
    for _, g in df.groupby(JOURNEY):
        codes = g.sort_values("station_sequence").station_code.tolist()
        for a, b in zip(codes, codes[1:]):
            if a != b:
                edges.setdefault(tuple(sorted((a, b))), len(edges))
    return st, node, edges


def build_journeys(df, edges):
    J = []
    for (tn, jd), g in df.groupby(JOURNEY, sort=False):
        g = g.sort_values("station_sequence")
        if len(g) < 2 or g[["latitude", "longitude"]].isna().any().any():
            continue
        codes = g.station_code.tolist()
        J.append(dict(jid=len(J), train_no=tn, journey_date=jd, idx=g.index.values, codes=codes,
                      A=minutes(g.dep_ts), S=minutes(g.sched_dep_ts),
                      lat=g.latitude.values, lon=g.longitude.values,
                      edge=[edges.get(tuple(sorted((a, b))), -1) for a, b in zip(codes, codes[1:])]))
    return J


def locate(j, t):
    """(lat, lon, edge id, overdue min) of journey j at time t, or None if not between stations."""
    A, S = j["A"], j["S"]
    if t < A[0] or t >= A[-1]:
        return None
    i = min(np.searchsorted(A, t, side="right") - 1, len(A) - 2)
    seg_t = max(S[i + 1] - S[i], 1.0)
    f = min((t - A[i]) / seg_t, 1.0)
    return (j["lat"][i] + f * (j["lat"][i + 1] - j["lat"][i]), j["lon"][i] + f * (j["lon"][i + 1] - j["lon"][i]),
            j["edge"][i], max(t - A[i] - seg_t, 0.0))


def snapshots(J, n_edges):
    t0 = np.floor(min(j["A"][0] for j in J) / STEP) * STEP
    grid = np.arange(t0, max(j["A"][-1] for j in J), STEP)
    starts, ends = np.array([j["A"][0] for j in J]), np.array([j["A"][-1] for j in J])
    tabs = {k: np.zeros((len(grid), n_edges), np.float32) for k in ("occ", "clu", "stall")}
    rec = []
    for ti, t in enumerate(grid):
        act = np.where((starts <= t) & (ends > t))[0]
        loc = [(k, locate(J[k], t)) for k in act]
        loc = [(k, p) for k, p in loc if p is not None and p[2] >= 0]
        if not loc:
            continue
        lat = np.array([p[0] for _, p in loc])
        lon = np.array([p[1] for _, p in loc])
        lab, size = cluster_points(lat, lon)
        e = np.array([p[2] for _, p in loc])
        np.add.at(tabs["occ"][ti], e, 1)
        np.add.at(tabs["clu"][ti], e, (size > 1).astype(np.float32))
        np.add.at(tabs["stall"][ti], e, np.minimum([p[3] for _, p in loc], 120))
        for (k, p), lb, sz in zip(loc, lab, size):
            rec.append((ti, k, p[0], p[1], p[2], lb, sz))
    pts = pd.DataFrame(rec, columns=["t_idx", "jid", "lat", "lon", "edge", "cluster", "cluster_size"])
    return grid, tabs, pts


def station_features(df):
    out = pd.DataFrame(index=df.index, columns=["headway_min", "trains_last_30min", "trains_sched_next_30min"],
                       dtype=float)
    for _, g in df.groupby("station_code"):
        t, ts, tn = minutes(g.dep_ts), minutes(g.sched_dep_ts), g.train_no.values
        order = np.argsort(t)
        for i, idx in enumerate(g.index):
            other = tn != tn[i]
            prev = t[other & (t < t[i])]
            out.at[idx, "headway_min"] = min(t[i] - prev.max(), 180.0) if len(prev) else 180.0
            out.at[idx, "trains_last_30min"] = np.sum(other & (t >= t[i] - 30) & (t < t[i]))
            out.at[idx, "trains_sched_next_30min"] = np.sum(other & (ts > t[i]) & (ts <= t[i] + 30))
    return out


def main():
    df = load_data()
    st, node, edges = build_graph(df)
    J = build_journeys(df, edges)
    print(f"graph: {len(st)} stations, {len(edges)} sections; {len(J)} journeys")
    grid, tabs, pts = snapshots(J, len(edges))

    size_at = {(r.jid, r.t_idx): r.cluster_size for r in pts.itertuples()}
    df["dbscan_cluster_size"] = 1
    for j in J:
        ti = ((j["A"] - grid[0]) // STEP).astype(int)
        for idx, k in zip(j["idx"], ti):
            df.at[idx, "dbscan_cluster_size"] = size_at.get((j["jid"], int(k)), 1)
    df[["headway_min", "trains_last_30min", "trains_sched_next_30min"]] = station_features(df)

    meta = pd.DataFrame([(j["jid"], j["train_no"], j["journey_date"]) for j in J],
                        columns=["jid", "train_no", "journey_date"])
    pts.merge(meta, on="jid").to_pickle(STORE / "dbscan_points.pkl")
    elist = sorted(edges.items(), key=lambda x: x[1])
    np.savez_compressed(STORE / "network_tables.npz", grid=grid, **tabs,
                        edge_a=np.array([node[a] for (a, b), _ in elist]),
                        edge_b=np.array([node[b] for (a, b), _ in elist]))
    st.to_csv(STORE / "stations.csv", index=False)
    (STORE / "edges.json").write_text(json.dumps([[a, b] for (a, b), _ in elist]))
    out = df.drop(columns=["dep_ts", "sched_dep_ts"])
    out["journey_date"] = out.journey_date.dt.strftime("%d-%m-%Y")
    out.to_csv(FEATURED_CSV, index=False)

    print(f"snapshots {len(grid)}, positions {len(pts)}, in a DBSCAN cluster: {(pts.cluster_size > 1).mean():.1%}")
    print(f"rows with another train at the station in the last 30 min: {(df.trains_last_30min > 0).mean():.1%}")
    print(df[["headway_min", "trains_last_30min", "trains_sched_next_30min", "dbscan_cluster_size"]]
          .describe().round(2).T[["mean", "50%", "max"]])


if __name__ == "__main__":
    main()
