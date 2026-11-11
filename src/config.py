
from pathlib import Path

# Thu muc goc cua du an
BASE_DIR = Path(__file__).resolve().parent.parent

# Duong dan du lieu
RAW_DATA_PATH = (
    BASE_DIR / "data" /  "Crop_recommendation.csv"
)

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# dac trung dau vao
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# Nhan can du doan
TARGET = "label"

RANDOM_STATE = 42
