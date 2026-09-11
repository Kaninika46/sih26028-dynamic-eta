import pandas as pd

# Load merged dataset
df = pd.read_csv("xgboost_phase1_weather_merged_august_2026.csv")

# Calculate remaining delay
df["remaining_delay_min"] = (
    df["actual_remaining_time_min"]
    - df["scheduled_remaining_time_min"]
)

# Save new dataset
df.to_csv(
    "xgboost_phase1_delay_dataset.csv",
    index=False
)

print("==============================")
print("DELAY DATASET CREATED")
print("==============================")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()
print("Remaining delay statistics:")
print(df["remaining_delay_min"].describe())
print()
print("Saved as:")
print("xgboost_phase1_delay_dataset.csv")