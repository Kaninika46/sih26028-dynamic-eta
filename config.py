"""Central configuration for the 40-train project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, OUT, STORE = ROOT / "data", ROOT / "outputs", ROOT / "models_store"
CACHE = DATA / "cache"
RAW = DATA / "raw"
for p in (DATA, OUT, STORE, CACHE, RAW):
    p.mkdir(parents=True, exist_ok=True)

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

RAILRADAR_KEY = os.getenv("RAILRADAR_KEY", "")
RAILKIT_KEY = os.getenv("RAILKIT_KEY", "")
RAILKIT_MODE = os.getenv("RAILKIT_MODE", "sdk")

MASTER_RAW = DATA / "master_40_trains.csv"            # team's 40-train master dataset
CLEAN_CSV = DATA / "master_40_clean.csv"              # after pipeline.prepare_data
STATIONS_CSV = DATA / "station_coords.csv"
# models read the cleaned file when it exists (fixed coordinates + distances)
MASTER_CSV = CLEAN_CSV if CLEAN_CSV.exists() else MASTER_RAW
XGB_DIR = ROOT / "XGBOOST"                            # first XGBoost package (v1, kept for reference + split days)
XGB_FEATURES_CSV = XGB_DIR / "xgboost_40_train_eta_features.csv"
XGB_MODEL = XGB_DIR / "xgboost_40_train_eta_model.json"
XGB_FEATURES = ["train_no", "station_sequence", "current_delay_min", "scheduled_remaining_time_min",
                "previous_segment_time_min", "hour", "day_of_week"]   # exactly as in train_40_train_xgboost.py

# bounding box used to keep live RailRadar trains near this network (Rajasthan + approaches)
CORRIDOR_BBOX = (22.0, 69.0, 31.5, 80.5)
HEADWAY_MIN = 6.0          # minimum spacing of trains leaving a station towards the same next station

WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain",
                "weather_code", "wind_speed_10m", "cloud_cover", "surface_pressure"]
