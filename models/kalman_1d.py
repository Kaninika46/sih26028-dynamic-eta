"""
1D Kalman filter branch for remaining-delay prediction.

Idea
----
Trains usually RECOVER time before the destination (schedule slack), so a
pure "delay keeps growing at the same rate" filter is wrong. We therefore:

1. Learn a historical profile from the TRAIN days only:
     expected_delay(train, station)       = usual delay at this station
     prior_remaining(train, station)      = usual delay change until destination
2. Run a Kalman filter on the ANOMALY = observed delay - expected delay.
     state = [anomaly level (min), anomaly trend (min per km)]
3. Prediction of remaining delay
     = prior_remaining + k_rate * trend * km_remaining + k_level * level
   (k_rate, k_level are fitted on the validation days.)

Everything is causal: at each station the filter only uses stations already
passed, so it works live, station by station.

Also provides LiveKalman: the same filter for one live train, with dead reckoning
(Telemetry Guard): if no new delay reading arrives (GPS/API dropout), it runs the
predict step only, so the ETA keeps moving and its range widens instead of crashing.

Run:  python -m models.kalman_1d
Out:  outputs/kalman_predictions.csv, models_store/kalman_params.json
"""
import itertools
import json

import numpy as np
import pandas as pd

from config import OUT, STORE
from models.common import JOURNEY, TARGET, fmt, load_data, metrics
from models.common import split as split_by_date

Z80 = 1.2816  # z-score for an 80% interval (10th-90th percentile)


# ----------------------------------------------------------------------
# Historical profile (fit on training days only)
# ----------------------------------------------------------------------
def build_profile(train_df):
    g = train_df.groupby(["train_no", "station_code"])
    prof = pd.DataFrame({
        "exp_delay": g["current_delay_min"].mean(),
        "prior": g[TARGET].mean(),
        "prior_std": g[TARGET].std(),
        "n": g.size(),
    }).reset_index()

    tg = train_df.groupby("train_no")
    train_level = pd.DataFrame({
        "t_exp_delay": tg["current_delay_min"].mean(),
        "t_prior": tg[TARGET].mean(),
        "t_prior_std": tg[TARGET].std(),
    }).reset_index()

    glob = {
        "exp_delay": float(train_df["current_delay_min"].mean()),
        "prior": float(train_df[TARGET].mean()),
        "prior_std": float(train_df[TARGET].std()),
    }
    return prof, train_level, glob


def attach_profile(df, prof, train_level, glob, min_n=3):
    out = df.merge(prof, on=["train_no", "station_code"], how="left")
    out = out.merge(train_level, on="train_no", how="left")
    few = out["n"].fillna(0) < min_n          # too little history -> fall back
    for col, tcol in [("exp_delay", "t_exp_delay"), ("prior", "t_prior"),
                      ("prior_std", "t_prior_std")]:
        out.loc[few, col] = np.nan
        out[col] = out[col].fillna(out[tcol]).fillna(glob[col])
    return out.drop(columns=["n", "t_exp_delay", "t_prior", "t_prior_std"])


# ----------------------------------------------------------------------
# Kalman filter core
# ----------------------------------------------------------------------
class DelayKalman:
    """State x = [level (min), trend (min/km)] of the delay anomaly."""

    def __init__(self, q_level, q_trend, r_obs, p_trend0=0.05):
        self.q_level, self.q_trend, self.r = q_level, q_trend, r_obs
        self.p_trend0 = p_trend0
        self.x = None
        self.P = None
        self.last_km = None

    def update(self, anomaly, km):
        if self.x is None:                        # first station of journey
            self.x = np.array([anomaly, 0.0])
            self.P = np.diag([self.r, self.p_trend0])
            self.last_km = km
            return self.x, self.P

        dk = max(km - self.last_km, 0.0)
        F = np.array([[1.0, dk], [0.0, 1.0]])
        Q = np.diag([self.q_level * dk, self.q_trend * dk])
        # predict
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        # correct with observed anomaly
        H = np.array([1.0, 0.0])
        S = H @ P @ H + self.r
        K = P @ H / S
        x = x + K * (anomaly - H @ x)
        P = (np.eye(2) - np.outer(K, H)) @ P
        self.x, self.P, self.last_km = x, P, km
        return x, P


def run_filter(df, q_level, q_trend, r_obs):
    """Runs the filter over every journey; returns level, trend, trend variance."""
    level = np.zeros(len(df))
    trend = np.zeros(len(df))
    var_trend = np.zeros(len(df))
    df = df.reset_index(drop=True)
    for _, idx in df.groupby(JOURNEY, sort=False).groups.items():
        kf = DelayKalman(q_level, q_trend, r_obs)
        rows = df.loc[idx].sort_values("station_sequence")
        for i, row in rows.iterrows():
            anomaly = row["current_delay_min"] - row["exp_delay"]
            x, P = kf.update(anomaly, row["distance_travelled_km"])
            level[i], trend[i], var_trend[i] = x[0], x[1], P[1, 1]
    return level, trend, var_trend


def design(df, level, trend):
    """Two correction terms added on top of the historical prior."""
    return np.column_stack([trend * df["distance_remaining_km"].values, level])


def predict(df, level, trend, var_trend, k, scale=1.0):
    X = design(df, level, trend)
    pred = df["prior"].values + X @ k
    std = np.sqrt(df["prior_std"].values ** 2
                  + (k[0] * df["distance_remaining_km"].values) ** 2 * var_trend)
    half = Z80 * std * scale
    return pred, pred - half, pred + half


def fit_k(df, level, trend):
    """Least squares for k_rate, k_level on (target - prior)."""
    X = design(df, level, trend)
    y = df[TARGET].values - df["prior"].values
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    return k


# ----------------------------------------------------------------------
# Main: tune on validation, report on test
# ----------------------------------------------------------------------
def main():
    df = load_data()
    train, val, test = split_by_date(df)
    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)} rows")

    prof, train_level, glob = build_profile(train)
    val = attach_profile(val, prof, train_level, glob).reset_index(drop=True)
    test = attach_profile(test, prof, train_level, glob).reset_index(drop=True)

    # --- tune noise parameters on validation ---------------------------------
    # (k is fitted on half the val journeys and scored on the other half so the
    #  grid search does not reward overfitting)
    vj = val[JOURNEY].drop_duplicates().sample(frac=1.0, random_state=0)
    half_a = val.merge(vj.iloc[: len(vj) // 2], on=JOURNEY).index
    mask_a = val.index.isin(half_a)

    grid = itertools.product([0.01, 0.1, 1.0],        # q_level
                             [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # q_trend
                             [1.0, 4.0, 16.0, 64.0])        # r_obs
    best = None
    for ql, qt, r in grid:
        lv, tr, vt = run_filter(val, ql, qt, r)
        k = fit_k(val[mask_a], lv[mask_a], tr[mask_a])
        pred, _, _ = predict(val, lv, tr, vt, k)
        mae = metrics(val[TARGET][~mask_a], pred[~mask_a])["MAE"]
        if best is None or mae < best[0]:
            best = (mae, ql, qt, r)
    _, ql, qt, r = best
    print(f"Best params: q_level={ql}, q_trend={qt}, r_obs={r}")

    # refit k on all validation (used for test / live)
    lv, tr, vt = run_filter(val, ql, qt, r)
    k = fit_k(val, lv, tr)

    # Honest validation predictions for fusion: cross-fit (k from one half of
    # the val journeys predicts the other half), otherwise val MAE is too rosy.
    k_a, k_b = fit_k(val[mask_a], lv[mask_a], tr[mask_a]), fit_k(val[~mask_a], lv[~mask_a], tr[~mask_a])
    y = val[TARGET].values
    scale = 1.0
    for s in np.linspace(0.3, 3.0, 55):
        pa, loa, hia = predict(val, lv, tr, vt, k_b, s)   # half A scored by k from B
        pb, lob, hib = predict(val, lv, tr, vt, k_a, s)   # half B scored by k from A
        pv = np.where(mask_a, pa, pb)
        lo_v = np.where(mask_a, loa, lob)
        hi_v = np.where(mask_a, hia, hib)
        if np.mean((y >= lo_v) & (y <= hi_v)) >= 0.80:
            scale = s
            break
    print(f"k_rate={k[0]:.3f}, k_level={k[1]:.3f}, interval scale={scale:.2f}")

    # --- test -------------------------------------------------------------
    lt, tt, vtt = run_filter(test, ql, qt, r)
    pt, lot, hit = predict(test, lt, tt, vtt, k, scale)
    yt = test[TARGET].values

    print("\n========== TEST RESULTS (remaining delay = ETA error) ==========")
    print("On-schedule baseline   :", fmt(metrics(yt, np.zeros_like(yt))))
    print("Historical prior only  :", fmt(metrics(yt, test["prior"].values)))
    print("Kalman + prior         :", fmt(metrics(yt, pt)))
    cov = np.mean((yt >= lot) & (yt <= hit))
    print(f"80% interval coverage  : {cov:.1%}  (target 80%)")
    print(f"Avg interval width     : {np.mean(hit - lot):.1f} min")

    # --- save -------------------------------------------------------------
    keep = JOURNEY + ["station_code", "station_sequence", TARGET]
    out_v = val[keep].assign(split="val", kf_pred=pv, kf_low=lo_v, kf_high=hi_v)
    out_t = test[keep].assign(split="test", kf_pred=pt, kf_low=lot, kf_high=hit)
    outk = pd.concat([out_v, out_t])
    outk["journey_date"] = pd.to_datetime(outk.journey_date).dt.strftime("%d-%m-%Y")
    outk.to_csv(OUT / "kalman_predictions.csv", index=False)

    params = {"q_level": ql, "q_trend": qt, "r_obs": r,
              "k_rate": float(k[0]), "k_level": float(k[1]), "scale": scale,
              "global": glob,
              "profile": prof.to_dict(orient="records"),
              "train_level": train_level.to_dict(orient="records")}
    with open(STORE / "kalman_params.json", "w") as f:
        json.dump(params, f, default=float)
    print("\nSaved outputs/kalman_predictions.csv and models_store/kalman_params.json")


class LiveKalman:
    """Filter for one running train.

    kf = LiveKalman(train_no)
    kf.update(station_code, delay_min, km_travelled, km_remaining)  -> prediction dict
    kf.dead_reckon(km_travelled_estimate, km_remaining)             -> prediction dict (no reading)
    """

    def __init__(self, train_no, params=None):
        p = params or json.loads((STORE / "kalman_params.json").read_text())
        self.p = p
        self.kf = DelayKalman(p["q_level"], p["q_trend"], p["r_obs"])
        self.k = np.array([p["k_rate"], p["k_level"]])
        self.prof = {(int(r["train_no"]), r["station_code"]): r for r in p["profile"]}
        self.tprof = {int(r["train_no"]): r for r in p["train_level"]}
        self.train_no = int(train_no)
        self.last_station = None
        self.readings_missed = 0

    def _profile(self, station_code):
        r = self.prof.get((self.train_no, station_code))
        if r is not None and r.get("n", 0) >= 3:
            return r["exp_delay"], r["prior"], r["prior_std"] if r["prior_std"] == r["prior_std"] else 10.0
        t = self.tprof.get(self.train_no)
        if t is not None:
            return t["t_exp_delay"], t["t_prior"], t["t_prior_std"] if t["t_prior_std"] == t["t_prior_std"] else 10.0
        g = self.p["global"]
        return g["exp_delay"], g["prior"], g["prior_std"]

    def _predict(self, x, P, km_remaining, prior, prior_std):
        pred = prior + self.k[0] * x[1] * km_remaining + self.k[1] * x[0]
        std = np.sqrt(prior_std ** 2 + (self.k[0] * km_remaining) ** 2 * P[1, 1] + P[0, 0] * self.k[1] ** 2)
        half = Z80 * std * self.p["scale"]
        return {"remaining_delay_pred": float(pred), "low": float(pred - half), "high": float(pred + half),
                "delay_est_min": float(self._exp + x[0]), "delay_trend_per_km": float(x[1]),
                "readings_missed": self.readings_missed}

    def update(self, station_code, delay_min, km_travelled, km_remaining):
        exp_delay, prior, prior_std = self._profile(station_code)
        x, P = self.kf.update(delay_min - exp_delay, km_travelled)
        self.last_station, self.readings_missed, self._exp = station_code, 0, exp_delay
        self._prior = (prior, prior_std)
        return self._predict(x, P, km_remaining, prior, prior_std)

    def dead_reckon(self, km_travelled_est, km_remaining):
        """No new reading: propagate the state forward only (uncertainty grows)."""
        if self.kf.x is None:
            raise RuntimeError("no reading received yet")
        dk = max(km_travelled_est - self.kf.last_km, 0.0)
        F = np.array([[1.0, dk], [0.0, 1.0]])
        Q = np.diag([self.kf.q_level * dk, self.kf.q_trend * dk])
        x = F @ self.kf.x
        P = F @ self.kf.P @ F.T + Q
        self.readings_missed += 1
        prior, prior_std = self._prior
        return self._predict(x, P, km_remaining, prior, prior_std)


if __name__ == "__main__":
    main()
