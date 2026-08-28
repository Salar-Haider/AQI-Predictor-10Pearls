# src/fetch_data.py

from pathlib import Path

import pandas as pd
import requests

from src.config import (
    AIR_QUALITY_API_URL,
    LATITUDE,
    LONGITUDE,
    RAW_DATA_PATH,
    TIMEZONE,
    WEATHER_API_URL,
)


def fetch_weather_data():
    """Fetch hourly weather data for Islamabad."""

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
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
        "forecast_days": 3,
        "timezone": TIMEZONE,
    }

    response = requests.get(
        WEATHER_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def fetch_air_quality_data():
    """Fetch hourly air-quality data for Islamabad."""

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
        "forecast_days": 3,
        "timezone": TIMEZONE,
    }

    response = requests.get(
        AIR_QUALITY_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def json_to_dataframe(data):
    """Convert Open-Meteo hourly JSON data into a DataFrame."""

    hourly_data = data["hourly"]

    df = pd.DataFrame(hourly_data)

    df["time"] = pd.to_datetime(df["time"])

    return df


def fetch_and_save_data():
    """Fetch weather and air-quality data and save them together."""

    weather_json = fetch_weather_data()
    air_quality_json = fetch_air_quality_data()

    weather_df = json_to_dataframe(weather_json)
    air_quality_df = json_to_dataframe(air_quality_json)

    weather_df = weather_df.rename(columns={"time": "timestamp"})
    air_quality_df = air_quality_df.rename(columns={"time": "timestamp"})

    df = pd.merge(
        weather_df,
        air_quality_df,
        on="timestamp",
        how="inner",
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    current_hour = (
        pd.Timestamp.now(
            tz=TIMEZONE
        )
        .floor("h")
        .tz_localize(None)
    )

    # Keep current and historical rows only.
    df = df[
        df["timestamp"] <= current_hour
    ].copy()

    # Calculate AQI change while previous
    # hourly AQI is still available.
    df["aqi_change"] = (
        df["us_aqi"].diff()
    )

    # Keep only the latest hourly row.
    df = (
        df.tail(1)
        .reset_index(drop=True)
    )

    df = df.reset_index(drop=True)

    output_path = Path(RAW_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    fetch_and_save_data()