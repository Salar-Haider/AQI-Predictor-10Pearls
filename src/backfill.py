# src/backfill.py

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    AIR_QUALITY_API_URL,
    HISTORICAL_WEATHER_API_URL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
)

from src.features import create_features
from src.feature_store import upload_dataframe


BACKFILL_DAYS = 90

BACKFILL_RAW_PATH = (
    "data/raw/islamabad_backfill_raw.csv"
)

BACKFILL_PROCESSED_PATH = (
    "data/processed/islamabad_backfill_features.csv"
)


def get_backfill_dates():
    """Get the start and end dates for the 90-day backfill."""

    end_date = date.today() - timedelta(days=1)

    start_date = end_date - timedelta(
        days=BACKFILL_DAYS - 1
    )

    return start_date, end_date


def fetch_historical_weather():
    """Fetch historical Islamabad weather from Open-Meteo."""

    start_date, end_date = get_backfill_dates()

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "surface_pressure,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_direction_10m,"
            "wind_gusts_10m"
        ),
        "timezone": TIMEZONE,
    }

    response = requests.get(
        HISTORICAL_WEATHER_API_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def fetch_historical_air_quality():
    """Fetch the previous 90 days of air-quality data."""

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone,"
            "us_aqi"
        ),
        "past_days": BACKFILL_DAYS,
        "forecast_days": 0,
        "timezone": TIMEZONE,
    }

    response = requests.get(
        AIR_QUALITY_API_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def json_to_dataframe(data):
    """Convert Open-Meteo hourly JSON to a DataFrame."""

    df = pd.DataFrame(
        data["hourly"]
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df


def create_backfill_dataset():
    """Create the historical Islamabad dataset."""

    start_date, end_date = get_backfill_dates()

    print(
        f"Backfill period: "
        f"{start_date} to {end_date}"
    )

    print(
        "Fetching historical weather data..."
    )

    weather_json = fetch_historical_weather()

    print(
        "Fetching historical air-quality data..."
    )

    air_quality_json = (
        fetch_historical_air_quality()
    )

    weather_df = json_to_dataframe(
        weather_json
    )

    air_quality_df = json_to_dataframe(
        air_quality_json
    )

    weather_df = weather_df.rename(
        columns={
            "time": "timestamp"
        }
    )

    air_quality_df = air_quality_df.rename(
        columns={
            "time": "timestamp"
        }
    )

    print(
        f"Weather rows: {len(weather_df)}"
    )

    print(
        f"Air-quality rows: {len(air_quality_df)}"
    )

    df = pd.merge(
        weather_df,
        air_quality_df,
        on="timestamp",
        how="inner",
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    raw_path = Path(
        BACKFILL_RAW_PATH
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        raw_path,
        index=False,
    )

    print(
        f"Saved {len(df)} raw historical rows."
    )

    features_df = create_features(
        df
    )

    processed_path = Path(
        BACKFILL_PROCESSED_PATH
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_csv(
        processed_path,
        index=False,
    )

    print(
        f"Saved {len(features_df)} "
        f"processed historical rows."
    )

    return features_df


def backfill_to_feature_store():
    """Create historical data and upload it to Hopsworks."""

    features_df = (
        create_backfill_dataset()
    )

    print(
        "Uploading historical features "
        "to Hopsworks..."
    )

    upload_dataframe(
        features_df
    )

    print(
        "Historical backfill completed."
    )


if __name__ == "__main__":
    backfill_to_feature_store()