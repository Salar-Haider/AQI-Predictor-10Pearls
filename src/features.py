from pathlib import Path

import math

import pandas as pd

from src.config import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)


def add_time_features(df):
    """Add time-based features."""

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_rush_hour"] = (
        df["hour"].isin([7, 8, 9, 17, 18, 19])
    ).astype(int)

    return df


def add_season_feature(df):
    """Add a simple four-season encoding."""

    def get_season(month):
        if month in [12, 1, 2]:
            return 0
        if month in [3, 4, 5]:
            return 1
        if month in [6, 7, 8]:
            return 2

        return 3

    df["season"] = df["month"].apply(get_season)

    return df


def add_wind_features(df):
    """Convert wind speed and direction into U/V components."""

    direction_radians = (
        df["wind_direction_10m"] * math.pi / 180
    )

    df["wind_u"] = (
        df["wind_speed_10m"]
        * direction_radians.apply(math.sin)
    )

    df["wind_v"] = (
        df["wind_speed_10m"]
        * direction_radians.apply(math.cos)
    )

    df["is_stagnant"] = (
        df["wind_speed_10m"] < 2
    ).astype(int)

    return df


def add_weather_interaction(df):
    """Add simple interaction between temperature and humidity."""

    df["temperature_humidity"] = (
        df["temperature_2m"]
        * df["relative_humidity_2m"]
    )

    return df


def add_aqi_change(df):
    """Add AQI change from the previous hour."""

    df["aqi_change"] = (
        df["us_aqi"].diff()
    )

    df["aqi_change"] = df["aqi_change"].fillna(0)

    return df


def create_features(df):
    """Run all feature-engineering steps."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

    df = add_time_features(df)
    df = add_season_feature(df)
    df = add_wind_features(df)
    df = add_weather_interaction(df)
    df = add_aqi_change(df)

    return df


def load_and_create_features():
    """Load raw data and create the complete feature dataset."""

    df = pd.read_csv(RAW_DATA_PATH)

    df = create_features(df)

    output_path = Path(PROCESSED_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    features = load_and_create_features()

    print(features.head())
    print("\nColumns:")
    print(features.columns.tolist())