# src/config.py

CITY = "Islamabad"

LATITUDE = 33.6844
LONGITUDE = 73.0479

TIMEZONE = "Asia/Karachi"

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

HISTORICAL_WEATHER_API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

AIR_QUALITY_API_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

RAW_DATA_PATH = "data/raw/islamabad_hourly.csv"

PROCESSED_DATA_PATH = (
    "data/processed/islamabad_features.csv"
)