# src/inference.py

import json
import math
import os
import pickle
from pathlib import Path

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

from src import model_registry
from src.config import (
    AIR_QUALITY_API_URL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    WEATHER_API_URL,
)
from src.features import (
    add_season_feature,
    add_time_features,
    add_weather_interaction,
    add_wind_features,
)


MODEL_NAME = "islamabad_aqi_model"

FEATURE_GROUP_NAME = "aqi_hourly_features"
FEATURE_GROUP_VERSION = 4

FORECAST_HOURS = 72

FORECAST_PATH = Path(
    "data/processed/islamabad_72_hour_forecast.csv"
)


def connect_to_hopsworks():
    """Connect to the Hopsworks project."""

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is missing")

    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT is missing")

    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key,
        engine="python",
    )

    return project


def fetch_future_weather():
    """Fetch the next 3 days of Islamabad weather."""

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
        "forecast_days": 4,
        "timezone": TIMEZONE,
    }

    response = requests.get(
        WEATHER_API_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def fetch_future_air_quality():
    """Fetch pollutant forecasts for the next 3 days."""

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "forecast_days": 4,
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
    """Convert Open-Meteo hourly JSON into a DataFrame."""

    df = pd.DataFrame(
        data["hourly"]
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df


def create_future_features():
    """Create base features for the next 72 hours."""

    print("Fetching future weather...")

    weather_json = fetch_future_weather()

    print("Fetching future pollutant data...")

    air_json = fetch_future_air_quality()

    weather_df = json_to_dataframe(
        weather_json
    )

    air_df = json_to_dataframe(
        air_json
    )

    weather_df = weather_df.rename(
        columns={"time": "timestamp"}
    )

    air_df = air_df.rename(
        columns={"time": "timestamp"}
    )

    df = pd.merge(
        weather_df,
        air_df,
        on="timestamp",
        how="inner",
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    current_hour = pd.Timestamp.now(
        tz=TIMEZONE
    ).floor("h")

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            TIMEZONE
        )

    df = df[
        df["timestamp"] >= current_hour
    ].reset_index(drop=True)

    df = df.head(
        FORECAST_HOURS
    )

    if len(df) < FORECAST_HOURS:
        raise ValueError(
            f"Expected 72 future hours, "
            f"but received only {len(df)}."
        )

    df = add_time_features(df)
    df = add_season_feature(df)
    df = add_wind_features(df)
    df = add_weather_interaction(df)

    return df



def download_best_model(project):
    """Download the best AQI model and metadata from Hopsworks."""

    model_registry = project.get_model_registry()

    models = model_registry.get_models(
        name=MODEL_NAME
    )

    if not models:
        raise ValueError(
            f"No registered model found: {MODEL_NAME}"
        )

    model = max(
        models,
        key=lambda registered_model:
            registered_model.version,
    )

    if model is None:
        raise ValueError(
            f"No registered model found: {MODEL_NAME}"
        )

    print(
        f"Using model {model.name} "
        f"version {model.version}"
    )

    download_path = Path(
        model.download()
    )

    model_files = list(
        download_path.rglob("best_model.pkl")
    )

    feature_files = list(
        download_path.rglob(
            "feature_columns.json"
        )
    )

    metrics_files = list(
        download_path.rglob(
            "metrics.json"
        )
    )

    shap_files = list(
        download_path.rglob(
            "shap_importance.json"
        )
    )

    best_info_files = list(
        download_path.rglob(
            "best_model_info.json"
        )
    )

    if not model_files:
        raise FileNotFoundError(
            "best_model.pkl not found "
            "in downloaded model."
        )

    if not feature_files:
        raise FileNotFoundError(
            "feature_columns.json not found "
            "in downloaded model."
        )

    with open(
        model_files[0],
        "rb",
    ) as file:
        trained_model = pickle.load(file)

    with open(
        feature_files[0],
        "r",
    ) as file:
        feature_columns = json.load(file)

    metrics = None

    if metrics_files:
        with open(
            metrics_files[0],
            "r",
        ) as file:
            metrics = json.load(file)

    shap_importance = None

    if shap_files:
        with open(
            shap_files[0],
            "r",
        ) as file:
            shap_importance = json.load(file)

    best_model_info = None

    if best_info_files:
        with open(
            best_info_files[0],
            "r",
        ) as file:
            best_model_info = json.load(file)

    metadata = {
        "model_name": model.name,
        "model_version": model.version,
        "metrics": metrics,
        "shap_importance": shap_importance,
        "best_model_info": best_model_info,
    }

    return (
        trained_model,
        feature_columns,
        metadata,
    )

def get_recent_aqi_history(project):
    """Get recent known AQI values from the Feature Store."""

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    df = feature_group.read(
        dataframe_type="pandas"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        "timestamp"
    )

    current_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    df = df[
        df["timestamp"] <= current_hour
    ]

    df = df.dropna(
        subset=["us_aqi"]
    )

    recent_aqi = (
        df["us_aqi"]
        .tail(24)
        .astype(float)
        .tolist()
    )

    if len(recent_aqi) < 24:
        raise ValueError(
            "At least 24 historical AQI "
            "values are required."
        )

    print(
        f"Loaded {len(recent_aqi)} "
        "recent AQI values."
    )

    return recent_aqi


def add_recursive_aqi_features(row, history):
    """Add AQI lag and rolling features to one future row."""

    row["aqi_lag_1"] = history[-1]

    row["aqi_lag_3"] = history[-3]

    row["aqi_lag_6"] = history[-6]

    row["aqi_lag_12"] = history[-12]

    row["aqi_lag_24"] = history[-24]

    row["aqi_rolling_mean_6"] = (
        sum(history[-6:]) / 6
    )

    row["aqi_rolling_mean_24"] = (
        sum(history[-24:]) / 24
    )

    return row

def make_72_hour_forecast(
    model,
    feature_columns,
    future_df,
    history,
):
    """Predict AQI recursively for the next 72 hours."""

    predictions = []

    history = history.copy()

    for index in range(len(future_df)):

        row = future_df.iloc[index].copy()

        row = add_recursive_aqi_features(
            row,
            history,
        )

        input_data = pd.DataFrame(
            [row]
        )

        input_data = input_data[
            feature_columns
        ]

        prediction = float(
            model.predict(input_data)[0]
        )

        prediction = max(
            0,
            prediction,
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    result = future_df.copy()

    result["predicted_aqi"] = predictions

    return result


def create_daily_summary(forecast_df):
    """Create daily AQI summaries."""

    df = forecast_df.copy()

    df["date"] = (
        df["timestamp"]
        .dt.date
    )

    daily = (
        df.groupby("date")
        .agg(
            average_aqi=(
                "predicted_aqi",
                "mean",
            ),
            minimum_aqi=(
                "predicted_aqi",
                "min",
            ),
            maximum_aqi=(
                "predicted_aqi",
                "max",
            ),
        )
        .reset_index()
    )

    return daily


def run_forecast():
    """Run the complete 3-day AQI forecast."""

    print("Starting Islamabad AQI forecast...")

    project = connect_to_hopsworks()

    (model,feature_columns,model_metadata,) = download_best_model(project)



    history = get_recent_aqi_history(
        project
    )

    future_df = create_future_features()

    forecast_df = make_72_hour_forecast(
        model=model,
        feature_columns=feature_columns,
        future_df=future_df,
        history=history,
    )

    FORECAST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_df.to_csv(
        FORECAST_PATH,
        index=False,
    )

    daily = create_daily_summary(
        forecast_df
    )

    print("\n3-Day AQI Forecast:")
    print(daily)

    print(
        f"\nSaved hourly forecast to "
        f"{FORECAST_PATH}"
    )

    return forecast_df, model_metadata


if __name__ == "__main__":
    run_forecast()