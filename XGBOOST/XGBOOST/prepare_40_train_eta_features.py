import pandas as pd
import numpy as np
from datetime import datetime, timedelta


INPUT_FILE = "xgboost_40_trains_dataset.csv"
OUTPUT_FILE = "xgboost_40_train_eta_features.csv"


df = pd.read_csv(INPUT_FILE)


def parse_datetime(value, journey_date):

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if value == "":
        return pd.NaT

    try:
        # Format: 06:10 01-Aug
        dt = datetime.strptime(
            value,
            "%H:%M %d-%b"
        )

        year = pd.to_datetime(journey_date).year

        return dt.replace(year=year)

    except:
        return pd.NaT


rows = []


for (train_no, journey_date), group in df.groupby(
    ["train_no", "journey_date"]
):

    group = group.sort_values("station_sequence").copy()

    # Parse scheduled and actual departure
    group["scheduled_dt"] = group.apply(
        lambda x: parse_datetime(
            x["scheduled_departure"],
            x["journey_date"]
        ),
        axis=1
    )

    group["actual_dt"] = group.apply(
        lambda x: parse_datetime(
            x["actual_departure"],
            x["journey_date"]
        ),
        axis=1
    )

    # Final station scheduled/actual arrival
    final_station = group.iloc[-1]

    final_scheduled_arrival = parse_datetime(
        final_station["scheduled_arrival"],
        journey_date
    )

    final_actual_arrival = parse_datetime(
        final_station["actual_arrival"],
        journey_date
    )

    previous_actual_time = None

    for _, station in group.iterrows():

        scheduled_current = station["scheduled_dt"]
        actual_current = station["actual_dt"]

        # Skip if scheduled time is unavailable
        if pd.isna(scheduled_current):
            continue

        # Scheduled remaining time
        if pd.notna(final_scheduled_arrival):

            scheduled_remaining = (
                final_scheduled_arrival - scheduled_current
            ).total_seconds() / 60

        else:
            scheduled_remaining = np.nan

        # Actual remaining time
        if (
            pd.notna(actual_current)
            and pd.notna(final_actual_arrival)
        ):

            actual_remaining = (
                final_actual_arrival - actual_current
            ).total_seconds() / 60

        else:
            actual_remaining = np.nan

                # Current delay = actual departure - scheduled departure
        if pd.notna(actual_current) and pd.notna(scheduled_current):

            current_delay = (
                actual_current - scheduled_current
            ).total_seconds() / 60

        else:
            current_delay = np.nan
        # Previous segment travel time
        if (
            previous_actual_time is not None
            and pd.notna(actual_current)
        ):

            previous_segment_time = (
                actual_current - previous_actual_time
            ).total_seconds() / 60

        else:
            previous_segment_time = np.nan

        # Save current actual time for next station
        if pd.notna(actual_current):
            previous_actual_time = actual_current

        rows.append({
            "train_no": train_no,
            "train_name": station["train_name"],
            "journey_date": journey_date,
            "station_code": station["station_code"],
            "station_name": station["station_name"],
            "station_sequence": station["station_sequence"],
            "current_delay_min": current_delay,
            "scheduled_remaining_time_min": scheduled_remaining,
            "previous_segment_time_min": previous_segment_time,
            "hour": scheduled_current.hour,
            "day_of_week": scheduled_current.weekday(),
            "actual_remaining_time_min": actual_remaining
        })


eta_df = pd.DataFrame(rows)


# Keep only rows where we know the actual remaining time
eta_df = eta_df[
    eta_df["actual_remaining_time_min"].notna()
].copy()


eta_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("==============================")
print("40-TRAIN ETA DATASET CREATED")
print("==============================")
print("Rows:", len(eta_df))
print("Columns:", len(eta_df.columns))
print("Trains:", eta_df["train_no"].nunique())
print("Saved as:", OUTPUT_FILE)