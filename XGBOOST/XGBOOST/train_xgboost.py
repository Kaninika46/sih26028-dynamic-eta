import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load Phase 1 weather dataset
df = pd.read_csv("xgboost_phase1_weather_merged_august_2026.csv")
# Convert date
df["journey_date"] = pd.to_datetime(
    df["journey_date"],
    format="%d-%m-%Y"
)

# Sort chronologically
df = df.sort_values("journey_date").reset_index(drop=True)

# Features
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

    # Weather
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure"
]

target = "actual_remaining_time_min"

X = df[features]
y = df[target]

# Chronological split
train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# XGBoost
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    early_stopping_rounds=50,
    random_state=42
)

# Train
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# Test
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print()
print("==============================")
print("XGBOOST V2 RESULTS")
print("==============================")
print("MAE :", round(mae, 2), "minutes")
print("RMSE:", round(rmse, 2), "minutes")

# Feature importance
print()
print("==============================")
print("FEATURE IMPORTANCE")
print("==============================")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance)

# Save model
model.save_model("xgboost_eta_weather_model.json")

print()
print("Model saved as: xgboost_eta_weather_model.json")