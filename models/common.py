"""Shared loading and the common split used by every branch.

Split = the teammate's XGBoost split (70% / 15% / 15% of date-sorted rows) converted to
whole days, so no branch is tested on rows XGBoost trained on:
  train <= 23 Aug   |   validation 24-27 Aug   |   test 28-31 Aug
"""
import numpy as np
import pandas as pd

from config import MASTER_CSV, OUT, XGB_FEATURES_CSV

TARGET = "remaining_delay_min"
KEY = ["train_no", "journey_date", "station_sequence"]
JOURNEY = ["train_no", "journey_date"]


def parse_ts(col, journey_date):
    """Handles both '2026-08-01 06:32:00' and the older '06:32 01-Aug' / '06:32 01-Aug*'."""
    s = col.astype(str).str.replace("*", "", regex=False).str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    old = pd.to_datetime(s + "-" + journey_date.dt.year.astype(str), format="%H:%M %d-%b-%Y", errors="coerce")
    ts = iso.fillna(old)
    wrap = ts < journey_date - pd.Timedelta(days=2)
    ts[wrap] = ts[wrap] + pd.DateOffset(years=1)
    return ts


def split_days():
    """Boundary days of the teammate's XGBoost split."""
    f = pd.read_csv(XGB_FEATURES_CSV, usecols=["train_no", "journey_date", "station_sequence"])
    f["journey_date"] = pd.to_datetime(f.journey_date, dayfirst=True)
    f = f.sort_values(["journey_date", "train_no", "station_sequence"]).reset_index(drop=True)
    n = len(f)
    return f.journey_date.iloc[int(n * 0.70)], f.journey_date.iloc[int(n * 0.85)]


def load_data(path=MASTER_CSV):
    df = pd.read_csv(path)
    df["journey_date"] = pd.to_datetime(df["journey_date"], format="%d-%m-%Y")
    df["dep_ts"] = parse_ts(df["actual_departure"], df["journey_date"])
    df["sched_dep_ts"] = parse_ts(df["scheduled_departure"], df["journey_date"])
    df[TARGET] = df["actual_remaining_time_min"] - df["scheduled_remaining_time_min"]
    df["station_name"] = df["station_name"].fillna(df["station_code"])
    # coordinates: 0/NaN -> station median from other rows -> interpolate along the journey
    for c in ("latitude", "longitude"):
        df.loc[df[c] == 0, c] = np.nan
        df[c] = df[c].fillna(df.groupby("station_code")[c].transform("median"))
    df = df.sort_values(KEY).reset_index(drop=True)
    for c in ("latitude", "longitude"):
        df[c] = df.groupby(JOURNEY, group_keys=False).apply(
            lambda g: g[c].interpolate(limit_direction="both") if g[c].notna().any() else g[c])
    df = df.dropna(subset=["dep_ts", "sched_dep_ts"]).reset_index(drop=True)
    return df


def split(df):
    tr_end, va_end = split_days()
    return (df[df.journey_date <= tr_end],
            df[(df.journey_date > tr_end) & (df.journey_date <= va_end)],
            df[df.journey_date > va_end])


def metrics(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}


def fmt(m):
    return f"MAE {m['MAE']:7.2f} | RMSE {m['RMSE']:7.2f}"


def save_preds(frame, name):
    out = frame.copy()
    out["journey_date"] = out.journey_date.dt.strftime("%d-%m-%Y")
    out.to_csv(OUT / name, index=False)
