"""
GCN-LSTM branch on the 40-train network.

Graph      491 station nodes, 816 section edges (models.dbscan_network). Edge weight at each
           5-minute snapshot = 1 + DBSCAN-clustered trains on that section.
Node input trains, clustered trains and stall minutes on the station's sections + time of day.
Graph conv symmetric-normalised propagation  A_t X_t  and  A_t^2 X_t  (two hops) for every
           snapshot (SGC-style: propagation precomputed once, weights learned), then a learned
           graph-convolution layer on [X, AX, A^2X].
LSTM       over the last 60 minutes for the train's current station and its next 3 stations.
Head       + the train's own state (delay, distances, schedule, train embedding)
           -> remaining delay with 10th / 90th percentile outputs.
Causal     the window ends at the snapshot before the train departs.

Run:  python -m models.gcn_lstm            (about 5-10 min on a laptop CPU)
      python -m models.gcn_lstm --no-graph (ablation: blank graph input)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import STORE
from models.common import JOURNEY, KEY, TARGET, fmt, load_data, metrics, save_preds, split
from models.dbscan_network import STEP

WINDOW, AHEAD = 12, 3
ROW_FEATS = ["current_delay_min", "distance_remaining_km", "distance_travelled_km",
             "scheduled_remaining_time_min", "station_sequence"]
torch.manual_seed(0)
np.random.seed(0)


def graph_inputs():
    tab = np.load(STORE / "network_tables.npz")
    grid, ea, eb = tab["grid"], tab["edge_a"], tab["edge_b"]
    n = int(max(ea.max(), eb.max())) + 1
    T = len(grid)
    feats = []
    for k in ("occ", "clu", "stall"):
        node = np.zeros((T, n), np.float32)
        np.add.at(node.T, ea, tab[k].T)
        np.add.at(node.T, eb, tab[k].T)
        feats.append(node)
    hour = (grid % 1440) / 1440 * 2 * np.pi
    feats += [np.repeat(np.sin(hour)[:, None], n, 1).astype(np.float32),
              np.repeat(np.cos(hour)[:, None], n, 1).astype(np.float32)]
    X = np.stack(feats, -1)                                            # (T, N, F)
    # dynamic symmetric-normalised adjacency with self loops, applied twice
    src = np.r_[ea, eb, np.arange(n)]
    dst = np.r_[eb, ea, np.arange(n)]
    w_e = 1.0 + tab["clu"]                                             # (T, E)
    out1, out2 = np.zeros_like(X), np.zeros_like(X)
    for s in range(0, T, 500):
        sl = slice(s, min(s + 500, T))
        w = np.concatenate([w_e[sl], w_e[sl], np.ones((w_e[sl].shape[0], n), np.float32)], 1)
        deg = np.zeros((w.shape[0], n), np.float32)
        np.add.at(deg.T, dst, w.T)
        norm = w / np.sqrt(deg[:, src] * deg[:, dst])
        for Xin, Xout in ((X, out1), (out1, out2)):
            msg = Xin[sl][:, src, :] * norm[:, :, None]
            agg = np.zeros((msg.shape[0], n, X.shape[-1]), np.float32)
            np.add.at(agg.transpose(1, 0, 2), dst, msg.transpose(1, 0, 2))
            Xout[sl] = agg
    return grid, np.concatenate([X, out1, out2], -1)                   # (T, N, 3F)


class GCNLSTM(nn.Module):
    def __init__(self, f_in, n_trains, n_row, hid=32):
        super().__init__()
        self.gconv = nn.Sequential(nn.Linear(f_in, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.emb = nn.Embedding(n_trains + 1, 8)
        self.head = nn.Sequential(nn.Linear(2 * hid + n_row + 8, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, G, row, tr):
        """G (B, W, K, F): graph features of the K route nodes over the window."""
        B, W, K, F = G.shape
        h = self.gconv(G).permute(0, 2, 1, 3).reshape(B * K, W, -1)
        _, (hn, _) = self.lstm(h)
        hn = hn[-1].reshape(B, K, -1)
        z = torch.cat([hn[:, 0], hn[:, 1:].mean(1), row, self.emb(tr)], 1)
        o = self.head(z)
        p = o[:, 0]
        return p, p - nn.functional.softplus(o[:, 1]), p + nn.functional.softplus(o[:, 2])


class Batcher:
    def __init__(self, df, grid, G, node_of, scaler=None, train_map=None):
        t = df.dep_ts.values.astype("datetime64[m]").astype("int64")
        self.ti = np.clip(((t - grid[0]) // STEP).astype(int), WINDOW - 1, len(grid) - 1)
        nodes = np.zeros((len(df), 1 + AHEAD), int)
        pos = {i: k for k, i in enumerate(df.index)}
        for _, g in df.groupby(JOURNEY):
            g = g.sort_values("station_sequence")
            ids = [node_of.get(c, 0) for c in g.station_code]
            for p, idx in enumerate(g.index):
                seq = ids[p:p + 1 + AHEAD]
                nodes[pos[idx]] = seq + [seq[-1]] * (1 + AHEAD - len(seq))
        self.nodes = nodes
        hour = df.dep_ts.dt.hour.values / 24 * 2 * np.pi
        row = np.column_stack([df[ROW_FEATS].fillna(0).values, np.sin(hour), np.cos(hour)]).astype(np.float32)
        if scaler is None:
            flat = G.reshape(-1, G.shape[-1])
            scaler = (row.mean(0), row.std(0) + 1e-6, flat.mean(0), flat.std(0) + 1e-6)
        self.scaler = scaler
        self.row = (row - scaler[0]) / scaler[1]
        self.G = G
        if train_map is None:
            train_map = {int(t): i + 1 for i, t in enumerate(sorted(df.train_no.unique()))}
        self.train_map = train_map
        self.tr = df.train_no.map(train_map).fillna(0).astype(int).values
        self.y = df[TARGET].values.astype(np.float32)

    def get(self, idx):
        win = self.ti[idx][:, None] - np.arange(WINDOW)[::-1]                    # (B, W)
        g = self.G[win[:, :, None], self.nodes[idx][:, None, :]]                 # (B, W, K, F)
        g = (g - self.scaler[2]) / self.scaler[3]
        return (torch.from_numpy(g.astype(np.float32)), torch.from_numpy(self.row[idx]),
                torch.from_numpy(self.tr[idx]))


def pinball(y, q, tau):
    e = y - q
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))


def predict(model, bt, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(bt.ti), bs):
            p, lo, hi = model(*bt.get(np.arange(s, min(s + bs, len(bt.ti)))))
            out.append(torch.stack([p, lo, hi], 1).numpy())
    return np.concatenate(out)


def node_index():
    st = pd.read_csv(STORE / "stations.csv")
    return {c: i for i, c in enumerate(st.station_code)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    df = load_data()
    grid, G = graph_inputs()
    if a.no_graph:
        G = np.zeros_like(G)
    node_of = node_index()
    tr, va, te = split(df)
    btr = Batcher(tr, grid, G, node_of)
    bva = Batcher(va, grid, G, node_of, btr.scaler, btr.train_map)
    bte = Batcher(te, grid, G, node_of, btr.scaler, btr.train_map)
    model = GCNLSTM(G.shape[-1], len(btr.train_map), btr.row.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    ytr = torch.from_numpy(btr.y)
    best, state, bad = 1e9, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(btr.ti))
        for s in range(0, len(perm), 256):
            idx = perm[s:s + 256]
            p, lo, hi = model(*btr.get(idx))
            y = ytr[idx]
            loss = nn.functional.huber_loss(p, y, delta=10.0) + pinball(y, lo, .1) + pinball(y, hi, .9)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        mae = metrics(bva.y, predict(model, bva)[:, 0])["MAE"]
        print(f"epoch {ep:2d} val MAE {mae:.2f}", flush=True)
        if mae < best - 0.01:
            best, state, bad = mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(state)
    pv, pt = predict(model, bva), predict(model, bte)
    yt = bte.y
    print("\nTEST 28-31 Aug (minutes)")
    print("  on schedule ", fmt(metrics(yt, np.zeros_like(yt))))
    print("  GCN-LSTM    ", fmt(metrics(yt, pt[:, 0])))
    print(f"  80% range coverage {np.mean((yt >= pt[:, 1]) & (yt <= pt[:, 2])):.1%}")
    if a.no_graph:
        (STORE / "gcn_ablation.json").write_text(json.dumps({"val_mae": best, "test": metrics(yt, pt[:, 0])}))
        print("ablation saved (model not saved)")
        return
    out = pd.concat([va[KEY + ["station_code", TARGET]].assign(split="val", gcn_pred=pv[:, 0], gcn_low=pv[:, 1],
                                                              gcn_high=pv[:, 2]),
                     te[KEY + ["station_code", TARGET]].assign(split="test", gcn_pred=pt[:, 0], gcn_low=pt[:, 1],
                                                              gcn_high=pt[:, 2])])
    save_preds(out, "gcn_predictions.csv")
    torch.save({"state": state, "scaler": [s.tolist() for s in btr.scaler], "train_map": btr.train_map,
                "f_in": G.shape[-1], "n_row": btr.row.shape[1]}, STORE / "gcn_lstm.pt")
    (STORE / "gcn_report.json").write_text(json.dumps({"val_mae": best, "test": metrics(yt, pt[:, 0])}))
    print("saved outputs/gcn_predictions.csv and models_store/gcn_lstm.pt")


if __name__ == "__main__":
    main()
