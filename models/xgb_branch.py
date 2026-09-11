"""
XGBoost branch v2: retrained on ALL 40 trains with a stronger setup.

What changed vs the first model (XGBOOST/train_40_train_xgboost.py):
  * data   : the cleaned 40-train master (all 40 trains; v1 lost 13 trains because its
             feature script dropped rows it could not parse)
  * target : remaining DELAY (actual - scheduled remaining time) instead of total remaining
             time, so the model learns the part that is uncertain; converted back for ETAs
  * split  : whole days, same as every other branch (train <= 23 Aug, val 24-27, test 28-31)
  * features:
      running state   current delay, delay change since the previous station, extra minutes
                      on the last section vs schedule, previous section time
      position        station sequence, km travelled / remaining, share of journey done,
                      scheduled remaining time, hour, weekday
      identity        train and station as categories
      history         average remaining delay for this train at this station / this train /
                      this station on training days (out-of-fold for training rows, no leakage)
      network         headway, trains nearby, DBSCAN cluster size, 2D-Kalman congestion ahead
      weather         Open-Meteo columns, if pipeline.add_weather was run
    Feature groups are compared on validation days and the best set is kept.
  * tuning : objective (squared / absolute / pseudo-Huber) and depth chosen on validation
  * range  : 10th / 90th percentile models (quantile regression)

Run:  python -m models.xgb_branch   (after models.dbscan_network and models.kalman_2d)
Out:  models_store/xgb_v2_*.json, outputs/xgb_predictions.csv
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from config import CLEAN_CSV, OUT, STORE, WEATHER_COLS
from models.common import JOURNEY, KEY, TARGET, fmt, load_data, metrics, save_preds, split

FEATURED = OUT / "dataset_with_network_features.csv"
BASE = ["train_no", "station_code", "station_sequence", "journey_frac", "current_delay_min", "delay_change_last",
        "segment_excess_min", "previous_segment_time_min", "scheduled_remaining_time_min",
        "distance_remaining_km", "distance_travelled_km", "hour", "day_of_week"]
PRIORS = ["prior_train_station", "prior_train", "prior_station"]
NETWORK = ["headway_min", "trains_last_30min", "trains_sched_next_30min", "dbscan_cluster_size",
           "congestion_ahead_30", "congestion_here"]
CATS = ["train_no", "station_code"]
GROUPS = {"Running delay carried": ["current_delay_min", "delay_change_last", "segment_excess_min",
                                    "previous_segment_time_min"],
          "Historical pattern": PRIORS + ["train_no", "station_code"],
          "Section congestion": NETWORK,
          "Weather": WEATHER_COLS,
          "Schedule & position": ["station_sequence", "journey_frac", "scheduled_remaining_time_min",
                                  "distance_remaining_km", "distance_travelled_km"],
          "Time of day": ["hour", "day_of_week"]}
PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=10, reg_lambda=2.0, tree_method="hist", enable_categorical=True,
              max_cat_to_onehot=1, early_stopping_rounds=150, random_state=42)


# ------------------------------------------------------------------ features
def load():
    df = load_data(FEATURED) if FEATURED.exists() else load_data()
    if CLEAN_CSV.exists():                                     # weather lives in the clean file
        w = pd.read_csv(CLEAN_CSV)
        cols = [c for c in WEATHER_COLS if c in w.columns]
        if cols:
            w["journey_date"] = pd.to_datetime(w.journey_date, format="%d-%m-%Y")
            df = df.drop(columns=[c for c in cols if c in df]).merge(w[KEY + cols], on=KEY, how="left")
    df = df.sort_values(KEY).reset_index(drop=True)
    g = df.groupby(JOURNEY)
    df["delay_change_last"] = df.current_delay_min - g.current_delay_min.shift()
    sched_seg = (df.sched_dep_ts - g.sched_dep_ts.shift()).dt.total_seconds() / 60
    df["segment_excess_min"] = df.previous_segment_time_min - sched_seg
    tot = df.distance_travelled_km + df.distance_remaining_km
    df["journey_frac"] = np.where(tot > 0, df.distance_travelled_km / tot, 0.0)
    return df


def fit_priors(train):
    return {"ts": train.groupby(["train_no", "station_code"])[TARGET].mean().to_dict(),
            "t": train.groupby("train_no")[TARGET].mean().to_dict(),
            "s": train.groupby("station_code")[TARGET].mean().to_dict(),
            "g": float(train[TARGET].mean())}


def apply_priors(d, enc):
    ts = pd.Series(list(zip(d.train_no, d.station_code)), index=d.index).map(enc["ts"])
    t = d.train_no.map(enc["t"])
    s = d.station_code.map(enc["s"])
    d = d.copy()
    d["prior_train"] = t.fillna(enc["g"]).values
    d["prior_station"] = s.fillna(enc["g"]).values
    d["prior_train_station"] = ts.fillna(d["prior_train"]).values
    return d


def oof_priors(train, k=5, seed=0):
    """Training rows get priors computed without their own journey (no target leakage)."""
    jid = train.groupby(JOURNEY).ngroup().values
    fold = np.random.default_rng(seed).integers(0, k, jid.max() + 1)[jid]
    parts = []
    for f in range(k):
        enc = fit_priors(train[fold != f])
        parts.append(apply_priors(train[fold == f], enc))
    return pd.concat(parts).loc[train.index]


def prep(d, feats, cats):
    X = d[feats].copy()
    for c in CATS:
        if c in X:
            X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    return X


def fit(Xtr, ytr, Xva, yva, **kw):
    m = XGBRegressor(**{**PARAMS, **kw})
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


# ------------------------------------------------------------------ used by the server
_BUNDLE = {}


def load_model():
    cfg = json.loads((STORE / "xgb_v2_config.json").read_text())
    m = XGBRegressor()
    m.load_model(STORE / "xgb_v2_point.json")
    _BUNDLE.update(cfg=cfg, model=m)
    return m


def features_for(rows):
    """Build model features for rows of the featured dataset (uses saved encoders)."""
    cfg = _BUNDLE["cfg"]
    d = apply_priors(rows, {k: ({tuple(json.loads(a)): v for a, v in cfg["priors"]["ts"].items()} if k == "ts"
                                else ({int(a): v for a, v in cfg["priors"]["t"].items()} if k == "t"
                                      else cfg["priors"][k]))
                            for k in ("ts", "t", "s", "g")})
    return prep(d, cfg["features"], cfg["cats"])


def reasons(model, rows):
    X = features_for(rows)
    c = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
    c = pd.DataFrame(c, columns=X.columns, index=rows.index)
    return pd.DataFrame({g: c[[f for f in fs if f in c]].sum(axis=1) for g, fs in GROUPS.items()
                         if any(f in c for f in fs)})


def known_trains():
    return set(int(t) for t in _BUNDLE["cfg"]["cats"]["train_no"]) if _BUNDLE else set()


# ------------------------------------------------------------------ training
def main():
    df = load()
    tr, va, te = split(df)
    tr_f = oof_priors(tr)
    enc = fit_priors(tr)
    va_f, te_f = apply_priors(va, enc), apply_priors(te, enc)
    cats = {c: sorted(df[c].astype(str).unique()) for c in CATS}

    weather = [c for c in WEATHER_COLS if c in df and df[c].notna().mean() > 0.5]
    network = [c for c in NETWORK if c in df]
    cands = {"base": BASE, "base+history": BASE + PRIORS}
    if network:
        cands["base+history+network"] = BASE + PRIORS + network
    if weather:
        cands["all (+weather)"] = BASE + PRIORS + network + weather
    print(f"{df.train_no.nunique()} trains | train {len(tr)} / val {len(va)} / test {len(te)} rows")
    print("Validation MAE per feature set:")
    scores = {}
    for name, feats in cands.items():
        m = fit(prep(tr_f, feats, cats), tr_f[TARGET], prep(va_f, feats, cats), va_f[TARGET])
        scores[name] = metrics(va_f[TARGET], m.predict(prep(va_f, feats, cats)))["MAE"]
        print(f"  {name:24s} {scores[name]:.2f}")
    best = min(scores, key=scores.get)
    feats = cands[best]
    print("chosen:", best)

    Xtr, Xva, Xte = (prep(d, feats, cats) for d in (tr_f, va_f, te_f))
    # small tuning grid, judged on validation days only
    grid = [dict(objective=o, max_depth=d, min_child_weight=w, **({"huber_slope": 10.0} if "huber" in o else {}))
            for o in ("reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror")
            for d, w in ((4, 20), (6, 10))]
    tuned = []
    for kw in grid:
        m = fit(Xtr, tr_f[TARGET], Xva, va_f[TARGET], **kw)
        tuned.append((metrics(va_f[TARGET], m.predict(Xva))["MAE"], kw, m))
    tuned.sort(key=lambda x: x[0])
    print("tuning (val MAE):", "; ".join(f"{k['objective'].split(':')[1]} d{k['max_depth']} {v:.2f}"
                                          for v, k, _ in tuned))
    best_kw, point = tuned[0][1], tuned[0][2]
    print("chosen params:", best_kw)
    q10 = fit(Xtr, tr_f[TARGET], Xva, va_f[TARGET], objective="reg:quantileerror", quantile_alpha=0.1)
    q90 = fit(Xtr, tr_f[TARGET], Xva, va_f[TARGET], objective="reg:quantileerror", quantile_alpha=0.9)

    yt = te_f[TARGET].values
    pt = point.predict(Xte)
    lo, hi = np.minimum(q10.predict(Xte), pt), np.maximum(q90.predict(Xte), pt)
    old = set(pd.read_csv(OUT.parent / "XGBOOST" / "xgboost_40_train_eta_features.csv",
                          usecols=["train_no"]).train_no)
    new13 = ~te_f.train_no.isin(old).values
    print("\nTEST 28-31 Aug (minutes)")
    print("  on schedule      ", fmt(metrics(yt, np.zeros_like(yt))))
    print("  XGBoost v2 (all) ", fmt(metrics(yt, pt)))
    print("  - 27 old trains  ", fmt(metrics(yt[~new13], pt[~new13])))
    print("  - 13 added trains", fmt(metrics(yt[new13], pt[new13])))
    cov = float(np.mean((yt >= lo) & (yt <= hi)))
    print(f"  80% range coverage {cov:.1%}, avg width {np.mean(hi - lo):.1f} min")
    imp = pd.Series(point.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTop features:\n", imp.head(10).round(3).to_string())

    for n, m in (("point", point), ("q10", q10), ("q90", q90)):
        m.save_model(STORE / f"xgb_v2_{n}.json")
    pri = {"ts": {json.dumps([int(a), b]): v for (a, b), v in enc["ts"].items()},
           "t": {str(int(a)): v for a, v in enc["t"].items()}, "s": enc["s"], "g": enc["g"]}
    (STORE / "xgb_v2_config.json").write_text(json.dumps(
        {"features": feats, "cats": cats, "priors": pri, "feature_set": best, "val_mae": scores,
         "params": best_kw,
         "test": metrics(yt, pt), "test_old27": metrics(yt[~new13], pt[~new13]),
         "test_new13": metrics(yt[new13], pt[new13]), "coverage80": cov}))
    keep = KEY + ["station_code", TARGET]
    out = []
    for name, d, X in (("val", va_f, Xva), ("test", te_f, Xte)):
        p = point.predict(X)
        out.append(d[keep].assign(split=name, xgb_pred=p, xgb_low=np.minimum(q10.predict(X), p),
                                  xgb_high=np.maximum(q90.predict(X), p)))
    save_preds(pd.concat(out), "xgb_predictions.csv")
    print("saved models_store/xgb_v2_*.json and outputs/xgb_predictions.csv")


if __name__ == "__main__":
    main()
