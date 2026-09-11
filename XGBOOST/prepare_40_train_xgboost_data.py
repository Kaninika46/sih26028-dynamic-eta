import os
import json
import pandas as pd
from datetime import datetime


DATA_FOLDER = r"C:\Users\91826\AppData\Local\Programs\Microsoft VS Code\railkit_all_trains"
OUTPUT_FILE = "xgboost_40_trains_dataset.csv"


rows = []


for train_number in os.listdir(DATA_FOLDER):

    train_folder = os.path.join(DATA_FOLDER, train_number)

    if not os.path.isdir(train_folder):
        continue

    for filename in os.listdir(train_folder):

        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(train_folder, filename)

        try:

            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

            if data.get("success") is not True:
                continue

            train_info = data.get("data", {})

            train_no = train_info.get("trainNo")
            train_name = train_info.get("trainName")
            journey_date = train_info.get("journeyDate")

            stations = train_info.get("stations", [])

            for sequence, station in enumerate(stations, start=1):

                arrival = station.get("arrival", {})
                departure = station.get("departure", {})

                arrival_delay = arrival.get("delay")
                departure_delay = departure.get("delay")

                rows.append({
                    "train_no": train_no,
                    "train_name": train_name,
                    "journey_date": journey_date,
                    "station_code": station.get("stationCode"),
                    "station_name": station.get("stationName"),
                    "station_sequence": sequence,
                    "platform": station.get("platform"),

                    "scheduled_arrival": arrival.get("scheduled"),
                    "actual_arrival": arrival.get("actual"),
                    "arrival_delay": arrival_delay,

                    "scheduled_departure": departure.get("scheduled"),
                    "actual_departure": departure.get("actual"),
                    "departure_delay": departure_delay
                })

        except Exception as error:

            print("Error:", filepath)
            print(error)


df = pd.DataFrame(rows)


if len(df) > 0:

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("==============================")
    print("40-TRAIN DATASET CREATED")
    print("==============================")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Trains:", df["train_no"].nunique())
    print("Saved as:", OUTPUT_FILE)

else:

    print()
    print("==============================")
    print("NO DATA FOUND")
    print("==============================")