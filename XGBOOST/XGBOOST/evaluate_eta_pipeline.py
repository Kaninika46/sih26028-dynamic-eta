import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("xgboost_phase1_delay_dataset.csv")

df["journey_date"] = pd.to_datetime(
    df["journey_date"],
    format="%d-%m-%Y"
)

df = df.sort_values("journey_date").reset_index(drop=True)

# ==========================================
# FEATURES
# ==========================================

features = [
    "train_no",
    "station_sequence",
    "distance_travelled_km",
    "distance_remaining_km",
    "current_delay_min",
    "scheduled_remaining_time_min",
    "previous_segment_time_min",
    "hour",
    "day_of_week",

    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure"
]

# ==========================================
# TARGET
# ==========================================

target = "remaining_delay_min"

X = df[features]
y = df[target]

# ==========================================
# SAME CHRONOLOGICAL SPLIT
# ==========================================

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

test_df = df.iloc[val_end:].copy()

print("Test rows:", len(X_test))

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = XGBRegressor()

model.load_model(
    "xgboost_remaining_delay_model.json"
)

# ==========================================
# PREDICT REMAINING DELAY
# ==========================================

predicted_delay = model.predict(X_test)

# ==========================================
# CONVERT DELAY → REMAINING TRAVEL TIME
# ==========================================

predicted_remaining_time = (
    test_df["scheduled_remaining_time_min"].values
    + predicted_delay
)

actual_remaining_time = (
    test_df["actual_remaining_time_min"].values
)

# ==========================================
# ETA ERROR
# ==========================================

eta_error = (
    predicted_remaining_time
    - actual_remaining_time
)

mae = mean_absolute_error(
    actual_remaining_time,
    predicted_remaining_time
)

rmse = np.sqrt(
    mean_squared_error(
        actual_remaining_time,
        predicted_remaining_time
    )
)

# ==========================================
# RESULTS
# ==========================================

print()
print("==============================")
print("END-TO-END ETA PIPELINE")
print("==============================")

print("ETA MAE :", round(mae, 2), "minutes")
print("ETA RMSE:", round(rmse, 2), "minutes")

print()
print("==============================")
print("SAMPLE PREDICTIONS")
print("==============================")

results = pd.DataFrame({
    "train_no": test_df["train_no"].values,
    "station_code": test_df["station_code"].values,
    "scheduled_remaining_min":
        test_df["scheduled_remaining_time_min"].values,
    "predicted_delay_min":
        predicted_delay,
    "predicted_remaining_min":
        predicted_remaining_time,
    "actual_remaining_min":
        actual_remaining_time,
    "error_min":
        eta_error
})

print(
    results.head(10).to_string(index=False)
)

# ==========================================
# SAVE RESULTS
# ==========================================

results.to_csv(
    "eta_pipeline_test_results.csv",
    index=False
)

print()
print("Results saved as:")
print("eta_pipeline_test_results.csv")