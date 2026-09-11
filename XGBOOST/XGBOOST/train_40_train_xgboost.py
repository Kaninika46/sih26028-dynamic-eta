import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


INPUT_FILE = "xgboost_40_train_eta_features.csv"
MODEL_FILE = "xgboost_40_train_eta_model.json"


# Load dataset
df = pd.read_csv(INPUT_FILE)


# Sort chronologically
df["journey_date"] = pd.to_datetime(
    df["journey_date"],
    dayfirst=True
)

df = df.sort_values(
    ["journey_date", "train_no", "station_sequence"]
).reset_index(drop=True)


# Features
FEATURES = [
    "train_no",
    "station_sequence",
    "current_delay_min",
    "scheduled_remaining_time_min",
    "previous_segment_time_min",
    "hour",
    "day_of_week"
]


TARGET = "actual_remaining_time_min"


X = df[FEATURES]
y = df[TARGET]


# Chronological split
n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)


X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]


print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))


# XGBoost model
model = xgb.XGBRegressor(
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
    eval_set=[
        (X_validation, y_validation)
    ],
    verbose=False
)


# Test prediction
predictions = model.predict(X_test)


# Metrics
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


print()
print("==============================")
print("40-TRAIN XGBOOST RESULTS")
print("==============================")
print("Test MAE:", round(mae, 2), "minutes")
print("Test RMSE:", round(rmse, 2), "minutes")
print("Best iteration:", model.best_iteration)


# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)


print()
print("FEATURE IMPORTANCE")
print("==============================")
print(importance)


# Save model
model.save_model(MODEL_FILE)

print()
print("Model saved as:", MODEL_FILE)