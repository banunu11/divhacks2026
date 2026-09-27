"""Shared constants and paths for the delay-prediction pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "delay_model.joblib"
TIMEZONE = "America/New_York"

# --- data.ny.gov (Socrata) datasets -------------------------------------
SOCRATA = "https://data.ny.gov/resource"
ALERTS_ID = "7kct-peq7"     # MTA Service Alerts: Beginning April 2020 (our labels)
RIDERSHIP_ID = "wujg-7c2s"  # MTA Subway Hourly Ridership: 2020-2024
STATIONS_ID = "39hk-dx4f"   # MTA Subway Stations (station complex -> routes)

# --- live feeds (used at prediction time) --------------------------------
LIVE_ALERTS_URL = (
    "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fsubway-alerts.json"
)
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NYC_LAT, NYC_LON = 40.75, -73.98  # Midtown
WEATHER_VARS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]

# --- modeling --------------------------------------------------------------
# Lines we predict for. Shuttles / SIR are skipped to keep things simple.
LINES = [
    "1", "2", "3", "4", "5", "6", "7",
    "A", "B", "C", "D", "E", "F", "G",
    "J", "L", "M", "N", "Q", "R", "W", "Z",
]
# Express variants that should be folded into their parent line.
LINE_ALIASES = {"6X": "6", "7X": "7", "FX": "F"}

TRAIN_START = "2021-01-01"   # skip the worst of the COVID ridership collapse
MAX_HORIZON_HOURS = 6        # how far ahead we forecast
# Recent-delay lookback windows. Kept <= 24h because at prediction time we
# only know recent history from MTA's live alert feed (the open-data archive
# lags ~6 weeks behind).
LAG_WINDOWS_HOURS = [1, 3, 6, 24]

# Risk buckets shown in the app. ~17% of line-hours have a delay alert, so
# "high" is roughly 2x the norm.
RISK_HIGH = 0.35
RISK_MEDIUM = 0.18

EXPERIMENTS_DIR = MODELS_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
# Everything from here on is held out for testing experiments.
TEST_START = "2025-08-01"
