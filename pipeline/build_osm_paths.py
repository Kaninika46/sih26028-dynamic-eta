"""
Real track paths from OpenStreetMap for every section (pair of consecutive stations).

For each section: take OSM railway=rail ways in a small box around the two stations, build a
graph of the track points, snap both stations to the nearest track point (<= 2 km) and take
the shortest path along the rails. If OSM has no connected track there, or the path is
implausibly long (> 2x the straight line + 3 km), the section falls back to a straight line.

Sections are sent to Overpass in batches (one request per 40 sections) and cached forever
in data/cache, so this runs once (~20 requests for the 816 sections).

Run:  python -m pipeline.build_osm_paths
Out:  data/osm_paths.json   {"JP>FL": [[lat, lon], ...], ...}
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from config import CLEAN_CSV, DATA, STATIONS_CSV
from sources import osm

OUT_JSON = DATA / "osm_paths.json"
PAD = 0.03            # degrees around the two stations (~3 km)
SNAP_KM = 2.0
BATCH = 40


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def sections():
    df = pd.read_csv(CLEAN_CSV, usecols=["train_no", "journey_date", "station_sequence", "station_code"])
    df = df.sort_values(["train_no", "journey_date", "station_sequence"])
    nxt = df.groupby(["train_no", "journey_date"]).station_code.shift(-1)
    pairs = pd.DataFrame({"a": df.station_code, "b": nxt}).dropna()
    pairs = pairs[pairs.a != pairs.b]
    key = pairs.apply(lambda r: tuple(sorted((r.a, r.b))), axis=1)
    return sorted(set(key))


def box(pa, pb):
    return (min(pa[0], pb[0]) - PAD, min(pa[1], pb[1]) - PAD, max(pa[0], pb[0]) + PAD, max(pa[1], pb[1]) + PAD)


def path_on_ways(ways, pa, pb, bx):
    """Shortest rail path between points pa and pb using ways inside box bx."""
    s, w, n, e = bx
    pts, idx, rows, cols, wts = [], {}, [], [], []

    def node(lat, lon):
        k = (round(lat, 6), round(lon, 6))        # ways that share a point connect here
        if k not in idx:
            idx[k] = len(pts)
            pts.append(k)
        return idx[k]

    for way in ways:
        g = [p for p in way["geometry"]]
        if not any(s <= la <= n and w <= lo <= e for la, lo in g):
            continue
        ids = [node(la, lo) for la, lo in g]
        for i, j in zip(ids, ids[1:]):
            if i != j:
                d = hav(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
                rows += [i, j]; cols += [j, i]; wts += [d, d]
    if len(pts) < 2:
        return None
    P = np.array(pts)
    da, db = hav(pa[0], pa[1], P[:, 0], P[:, 1]), hav(pb[0], pb[1], P[:, 0], P[:, 1])
    ia, ib = int(da.argmin()), int(db.argmin())
    if da[ia] > SNAP_KM or db[ib] > SNAP_KM:
        return None
    G = coo_matrix((wts, (rows, cols)), shape=(len(pts), len(pts))).tocsr()
    straight = hav(pa[0], pa[1], pb[0], pb[1])
    dist, pred = dijkstra(G, indices=ia, return_predecessors=True, limit=2 * straight + 3 + 2 * SNAP_KM)
    if not np.isfinite(dist[ib]):
        return None
    seq, k = [], ib
    while k != ia and k >= 0:
        seq.append(k)
        k = pred[k]
    seq.append(ia)
    line = [tuple(pa)] + [pts[i] for i in reversed(seq)] + [tuple(pb)]
    return decimate(line)


def decimate(line, step_km=0.4, cap=80):
    out = [line[0]]
    acc = 0.0
    for p, q in zip(line, line[1:]):
        acc += hav(p[0], p[1], q[0], q[1])
        if acc >= step_km:
            out.append(q)
            acc = 0.0
    if out[-1] != line[-1]:
        out.append(line[-1])
    if len(out) > cap:
        out = [out[int(i)] for i in np.linspace(0, len(out) - 1, cap)]
    return [[round(a, 5), round(b, 5)] for a, b in out]


def main():
    st = pd.read_csv(STATIONS_CSV).set_index("station_code")
    secs = [(a, b) for a, b in sections() if a in st.index and b in st.index]
    print(f"{len(secs)} sections to route along OpenStreetMap track")
    done = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    osm_ok = straight = 0
    for i in range(0, len(secs), BATCH):
        part = [s for s in secs[i:i + BATCH] if f"{s[0]}>{s[1]}" not in done]
        if not part:
            continue
        boxes = [box(st.loc[a, ["lat", "lon"]].values, st.loc[b, ["lat", "lon"]].values) for a, b in part]
        try:
            ways = osm.rail_ways_in_boxes(boxes)
        except Exception as e:
            print(f"batch {i // BATCH}: Overpass failed ({e}); these sections stay straight for now")
            continue
        for (a, b), bx in zip(part, boxes):
            pa, pb = st.loc[a, ["lat", "lon"]].values, st.loc[b, ["lat", "lon"]].values
            line = path_on_ways(ways, pa, pb, bx)
            if line is None:
                line = [[round(pa[0], 5), round(pa[1], 5)], [round(pb[0], 5), round(pb[1], 5)]]
                straight += 1
            else:
                osm_ok += 1
            done[f"{a}>{b}"] = line
        OUT_JSON.write_text(json.dumps(done))
        print(f"batch {i // BATCH + 1}: {len(ways)} OSM ways, total routed {osm_ok}, straight {straight}")
        time.sleep(2)                                   # be polite to the public Overpass server
    print(f"saved {OUT_JSON}: {len(done)} sections ({osm_ok} on OSM track, {straight} straight this run)")


def route_path(codes, paths, coords):
    """Concatenate section paths along a train's stations (either direction)."""
    out = []
    for a, b in zip(codes, codes[1:]):
        seg = paths.get(f"{a}>{b}")
        if seg is None and f"{b}>{a}" in paths:
            seg = paths[f"{b}>{a}"][::-1]
        if seg is None:
            seg = [coords[a], coords[b]] if a in coords and b in coords else []
        out += seg if not out else seg[1:]
    return out


if __name__ == "__main__":
    main()
