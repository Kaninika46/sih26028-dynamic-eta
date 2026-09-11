"""
Dynamic inverse-MAE ensemble fusion of XGBoost v2, 1D Kalman and GCN-LSTM.

  w_i = (1 / MAE_i) / sum_j (1 / MAE_j)        MAE_i on the validation days

Per row, only branches with a prediction take part and the weights are re-normalised to
exactly 1 (a branch with no prediction for a row is skipped for that row).
The fused 80% range = weighted branch ranges, rescaled on validation to hit 80%.

Run:  python -m models.fusion
"""
import json

import numpy as np
import pandas as pd

from config import OUT, STORE
from models.common import TARGET, fmt, metrics

KEY = ["train_no", "journey_date", "station_sequence"]
ALL = {"xgb": ("xgb_predictions.csv", "xgb_pred", "xgb_low", "xgb_high"),
       "kf": ("kalman_predictions.csv", "kf_pred", "kf_low", "kf_high"),
       "gcn": ("gcn_predictions.csv", "gcn_pred", "gcn_low", "gcn_high")}


def branches():
    return {k: v for k, v in ALL.items() if (OUT / v[0]).exists()}


def inverse_mae_weights(maes):
    inv = {k: 1.0 / max(v, 1e-6) for k, v in maes.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


def fuse(d, w, B):
    """Row-wise fusion over the branches available in that row."""
    W = np.column_stack([np.where(d[c[1]].notna(), w[n], 0.0) for n, c in B.items()])
    W = W / W.sum(1, keepdims=True)
    P = np.column_stack([d[c[1]].fillna(0).values for c in B.values()])
    Hh = np.column_stack([((d[c[3]] - d[c[2]]) / 2).fillna(0).values for c in B.values()])
    return (W * P).sum(1), (W * Hh).sum(1), W


def main():
    B = branches()
    m = None
    for n, (f, p, lo, hi) in B.items():
        d = pd.read_csv(OUT / f)
        d = d[KEY + ([TARGET, "split", "station_code"] if m is None else []) + [p, lo, hi]]
        m = d if m is None else m.merge(d, on=KEY, how="inner")
    val, test = m[m.split == "val"], m[m.split == "test"].copy()
    maes = {n: metrics(val.dropna(subset=[c[1]])[TARGET], val.dropna(subset=[c[1]])[c[1]])["MAE"]
            for n, c in B.items()}
    w = inverse_mae_weights(maes)
    print("validation MAE:", {k: round(v, 2) for k, v in maes.items()})
    print("weights       :", {k: round(v, 3) for k, v in w.items()}, "sum =", round(sum(w.values()), 6))
    pv, hv, _ = fuse(val, w, B)
    yv = val[TARGET].values
    scale = next((s for s in np.linspace(.5, 3, 51) if np.mean(np.abs(yv - pv) <= s * hv) >= .8), 3.0)
    pt, ht, _ = fuse(test, w, B)
    yt = test[TARGET].values
    lo, hi = pt - scale * ht, pt + scale * ht
    res = {"on_schedule": metrics(yt, np.zeros_like(yt))}
    print("\nTEST 28-31 Aug, all 40 trains (minutes)")
    print(f"  {'on schedule':12s}", fmt(res["on_schedule"]))
    for n, c in B.items():
        ok = test[c[1]].notna().values
        res[n] = metrics(yt[ok], test[c[1]].values[ok])
        print(f"  {n:12s}", fmt(res[n]), "" if ok.all() else f"(only {ok.mean():.0%} of rows: trains it knows)")
    res["fused"] = metrics(yt, pt)
    print(f"  {'FUSED':12s}", fmt(res["fused"]))
    cov = float(np.mean((yt >= lo) & (yt <= hi)))
    print(f"  fused 80% range coverage {cov:.1%}, avg width {np.mean(hi - lo):.1f} min")
    test = test.assign(fused_pred=pt, fused_low=lo, fused_high=hi)
    test.to_csv(OUT / "fused_predictions.csv", index=False)
    (STORE / "fusion.json").write_text(json.dumps({"weights": w, "val_mae": maes, "range_scale": float(scale),
                                                   "test": res, "coverage80": cov}))
    print("saved outputs/fused_predictions.csv")


if __name__ == "__main__":
    main()
