"""
OLD XGBoost branch (v1 = the first teammate model, kept for reference only).
Running it overwrites outputs/xgb_predictions.csv - use models.xgb_branch (v2) instead.

Loads XGBOOST/xgboost_40_train_eta_model.json (trained by XGBOOST/train_40_train_xgboost.py,
target actual_remaining_time_min, features XGB_FEATURES) and applies it to every row of the
40-train master dataset. Nothing is retrained.

For fusion the prediction is converted to remaining delay:
    xgb_pred = predicted actual_remaining_time - scheduled_remaining_time
The model only knows the 27 trains in its training file; for the other 13 trains its output
is unreliable (train_no is a numeric feature), so those rows get no XGBoost prediction and
fusion uses the remaining branches (weights re-normalised to 1).
The 80% range comes from the model's own validation errors (10th / 90th percentile of the
residuals), again without retraining. Reasons come from XGBoost's pred_contribs.

Run:  python -m models.xgb_branch
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from config import STORE, XGB_FEATURES, XGB_FEATURES_CSV, XGB_MODEL
from models.common import KEY, TARGET, fmt, load_data, metrics, save_preds, split

GROUPS = {"Running delay carried": ["current_delay_min", "previous_segment_time_min"],
          "Historical pattern": ["train_no", "station_sequence"],
          "Time of day": ["hour", "day_of_week"]}


def load_model():
    m = XGBRegressor()
    m.load_model(XGB_MODEL)
    return m


def predict_remaining_delay(model, rows):
    X = rows[XGB_FEATURES].astype(float)
    return model.predict(X) - rows["scheduled_remaining_time_min"].values


def reasons(model, rows):
    """Per-row minutes pushed by each group (sign kept). scheduled_remaining is the baseline, not a cause."""
    X = rows[XGB_FEATURES].astype(float)
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    c = pd.DataFrame(c, columns=XGB_FEATURES, index=rows.index)
    return pd.DataFrame({g: c[f].sum(axis=1) for g, f in GROUPS.items()})


def known_trains():
    return set(pd.read_csv(XGB_FEATURES_CSV, usecols=["train_no"]).train_no)


def main():
    df = load_data()
    tr, va, te = split(df)
    model = load_model()
    known = known_trains()
    pv, pt = predict_remaining_delay(model, va), predict_remaining_delay(model, te)
    kv, kt = va.train_no.isin(known).values, te.train_no.isin(known).values
    q10, q90 = np.quantile((va[TARGET].values - pv)[kv], [0.1, 0.9])   # range from validation errors
    yt = te[TARGET].values
    print(f"Teammate's XGBoost (unchanged). Knows {len(known)} of {df.train_no.nunique()} trains.")
    print("Test days 28-31 Aug, trains XGBoost knows:")
    print("  on schedule ", fmt(metrics(yt[kt], np.zeros(kt.sum()))))
    print("  XGBoost     ", fmt(metrics(yt[kt], pt[kt])))
    cov = np.mean((yt[kt] >= pt[kt] + q10) & (yt[kt] <= pt[kt] + q90))
    print(f"  80% range coverage {cov:.1%}, width {q90 - q10:.1f} min")
    print(f"Trains it never saw ({te[~kt].train_no.nunique()} in test): MAE "
          f"{metrics(yt[~kt], pt[~kt])['MAE']:.2f} -> excluded from fusion for those trains")
    pv[~kv], pt[~kt] = np.nan, np.nan
    out = pd.concat([va[KEY + ["station_code", TARGET]].assign(split="val", xgb_pred=pv,
                                                              xgb_low=pv + q10, xgb_high=pv + q90),
                     te[KEY + ["station_code", TARGET]].assign(split="test", xgb_pred=pt,
                                                              xgb_low=pt + q10, xgb_high=pt + q90)])
    save_preds(out, "xgb_predictions.csv")
    (STORE / "xgb_branch.json").write_text(json.dumps({"q10": float(q10), "q90": float(q90),
                                                      "test_known": metrics(yt[kt], pt[kt])}))
    print("saved outputs/xgb_predictions.csv")


if __name__ == "__main__":
    main()
