"""
2D Kalman filter (space x time) on the network's sections.

State      x_t[e] = stall minutes on section e (from models.dbscan_network)
Transition x_{t+1} = a x_t + b N x_t + c     N = neighbour matrix (sections sharing a station,
           row-normalised): congestion leaks into adjacent sections
Filter     steady-state Kalman gain (computed once), so 800+ sections run in real time
Output     filtered congestion now + 30-minute forecast for every section
Feature    congestion_ahead_30 = forecast stall on the next 3 sections of the train's route,
           congestion_here = filtered stall on its next section (both causal)

Run:  python -m models.kalman_2d
"""
import json

import numpy as np
import pandas as pd

from config import STORE
from models.common import JOURNEY, load_data, split_days
from models.dbscan_network import FEATURED_CSV, STEP

H, AHEAD = 6, 3


def neighbour_matrix(edge_a, edge_b):
    E = len(edge_a)
    inc = {}
    for e, (a, b) in enumerate(zip(edge_a, edge_b)):
        inc.setdefault(a, []).append(e)
        inc.setdefault(b, []).append(e)
    N = np.zeros((E, E), np.float32)
    for es in inc.values():
        for i in es:
            for j in es:
                if i != j:
                    N[i, j] = 1
    s = N.sum(1, keepdims=True)
    return np.divide(N, s, out=np.zeros_like(N), where=s > 0)


def fit(Z, N):
    mu = Z.mean(0)
    X0, X1 = Z[:-1] - mu, Z[1:] - mu
    nb = X0 @ N.T
    D = np.column_stack([X0.ravel(), nb.ravel()])
    (a, b), *_ = np.linalg.lstsq(D, X1.ravel(), rcond=None)
    A = a * np.eye(len(mu), dtype=np.float32) + b * N
    c = mu - A @ mu
    resid = X1.ravel() - D @ np.array([a, b])
    return float(a), float(b), A, c, float(resid.var())


def steady_gain(A, q, r, iters=60):
    E = len(A)
    P = np.eye(E, dtype=np.float64) * q
    for _ in range(iters):
        Pp = A @ P @ A.T + q * np.eye(E)
        K = Pp @ np.linalg.inv(Pp + r * np.eye(E))
        P = (np.eye(E) - K) @ Pp
    return K.astype(np.float32)


def run(Z, A, c, K):
    x = Z[0].copy()
    filt, fc = np.zeros_like(Z), np.zeros_like(Z)
    Ah = np.linalg.matrix_power(A.astype(np.float64), H).astype(np.float32)
    ch = sum(np.linalg.matrix_power(A.astype(np.float64), k) for k in range(H)).astype(np.float32) @ c
    for t in range(len(Z)):
        xp = A @ x + c
        x = xp + K @ (Z[t] - xp)
        filt[t] = x
        fc[t] = np.clip(Ah @ x + ch, 0, None)
    return filt, fc


def main():
    tab = np.load(STORE / "network_tables.npz")
    grid, Z = tab["grid"], tab["stall"].astype(np.float32)
    N = neighbour_matrix(tab["edge_a"], tab["edge_b"])
    tr_end, va_end = split_days()
    day = pd.to_datetime(grid * 60, unit="s").normalize()
    tr, te = day <= tr_end, day > va_end
    a, b, A, c, q = fit(Z[tr], N)
    K = steady_gain(A, q, 0.3 * q)
    filt, fc = run(Z, A, c, K)
    idx = np.where(te[:len(Z) - H])[0]
    truth = Z[idx + H]
    rep = {"a": a, "b_neighbour": b,
           "mse_kalman2d": float(np.mean((fc[idx] - truth) ** 2)),
           "mse_persistence": float(np.mean((Z[idx] - truth) ** 2)),
           "mse_zero": float(np.mean(truth ** 2))}
    print(f"sections {Z.shape[1]}, a={a:.3f}, neighbour coupling b={b:.3f}")
    print("30-min-ahead stall forecast MSE on test days: "
          f"2D Kalman {rep['mse_kalman2d']:.3f} | persistence {rep['mse_persistence']:.3f} | zero {rep['mse_zero']:.3f}")

    # route-ahead features for every row
    df = load_data()
    feat = pd.read_csv(FEATURED_CSV)
    edges = json.loads((STORE / "edges.json").read_text())
    eid = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    ahead, here = np.zeros(len(df)), np.zeros(len(df))
    t_idx = np.clip(((df.dep_ts.values.astype("datetime64[m]").astype("int64") - grid[0]) // STEP).astype(int),
                    0, len(grid) - 1)
    for _, g in df.groupby(JOURNEY):
        g = g.sort_values("station_sequence")
        codes = g.station_code.tolist()
        route = [eid.get(tuple(sorted((x, y))), -1) for x, y in zip(codes, codes[1:])]
        for pos, idx_row in enumerate(g.index):
            nxt = [e for e in route[pos:pos + AHEAD] if e >= 0]
            t = t_idx[idx_row]
            if nxt:
                ahead[idx_row] = fc[t, nxt].sum()
                here[idx_row] = filt[t, nxt[0]]
    feat["congestion_ahead_30"] = np.round(ahead, 3)
    feat["congestion_here"] = np.round(here, 3)
    feat.to_csv(FEATURED_CSV, index=False)
    np.savez_compressed(STORE / "kalman2d.npz", filt=filt, fc=fc)
    (STORE / "kalman2d_report.json").write_text(json.dumps(rep))
    print("added congestion_ahead_30 / congestion_here to", FEATURED_CSV.name)


if __name__ == "__main__":
    main()
