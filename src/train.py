# src/train.py

from pathlib import Path
import json
import math
import os
import pickle

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from src.features import add_aqi_lag_features


FEATURE_GROUP_NAME = "aqi_hourly_features"
FEATURE_GROUP_VERSION = 4

MODEL_DIR = Path("models")

TEST_SIZE = 0.20


EXCLUDED_COLUMNS = [
    "timestamp",
    "us_aqi",
    "aqi_change",
]


LAG_COLUMNS = [
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_rolling_mean_6",
    "aqi_rolling_mean_24",
]


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


def load_data():
    """Load historical data from Hopsworks Feature Store."""

    print("Connecting to Hopsworks...")

    project = connect_to_hopsworks()

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    print(
        f"Reading {FEATURE_GROUP_NAME} "
        f"version {FEATURE_GROUP_VERSION}..."
    )

    df = feature_group.read(
        dataframe_type="pandas"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Loaded {len(df)} rows from Hopsworks."
    )

    # Add training-time AQI history features
    df = add_aqi_lag_features(df)

    # First 24 rows cannot have complete lag history
    df = df.dropna(
        subset=LAG_COLUMNS
    ).reset_index(drop=True)

    return df


def prepare_data(df):
    """Create chronological training and testing datasets."""

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns]
    y = df["us_aqi"]

    split_index = int(
        len(df) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
    )


def create_models():
    """Create regression models."""

    models = {
        "ridge": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(alpha=1.0),
                ),
            ]
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
        ),

        "xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        ),
    }

    return models


def evaluate_model(model, X_test, y_test):
    """Calculate RMSE, MAE and R²."""

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = math.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def train_models():
    """Train, evaluate and save the best AQI model."""

    print("Loading historical data...")

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
    ) = prepare_data(df)

    print(f"\nTotal usable rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    print("\nTraining AQI:")
    print(y_train.describe())

    print("\nTesting AQI:")
    print(y_test.describe())

    print("\nTraining period:")
    print(
        df["timestamp"].iloc[0],
        "to",
        df["timestamp"].iloc[len(X_train) - 1],
    )

    print("\nTesting period:")
    print(
        df["timestamp"].iloc[len(X_train)],
        "to",
        df["timestamp"].iloc[-1],
    )

    models = create_models()

    results = {}
    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results[name] = metrics
        trained_models[name] = model

        print(
            f"RMSE: {metrics['rmse']:.3f}"
        )

        print(
            f"MAE:  {metrics['mae']:.3f}"
        )

        print(
            f"R²:   {metrics['r2']:.3f}"
        )

    best_model_name = min(
        results,
        key=lambda name: results[name]["rmse"],
    )

    best_model = trained_models[
        best_model_name
    ]

    print("\n---------------------------")
    print(
        f"Best model: {best_model_name}"
    )
    print("---------------------------")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "best_model.pkl"

    with open(
        model_path,
        "wb",
    ) as file:
        pickle.dump(
            best_model,
            file,
        )

    feature_path = (
        MODEL_DIR / "feature_columns.json"
    )

    with open(
        feature_path,
        "w",
    ) as file:
        json.dump(
            feature_columns,
            file,
            indent=4,
        )

    results_path = (
        MODEL_DIR / "metrics.json"
    )

    with open(
        results_path,
        "w",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    best_model_info = {
        "name": best_model_name,
        "metrics": results[best_model_name],
    }

    best_info_path = (
        MODEL_DIR / "best_model_info.json"
    )

    with open(
        best_info_path,
        "w",
    ) as file:
        json.dump(
            best_model_info,
            file,
            indent=4,
        )

    print(
        f"\nSaved best model to {model_path}"
    )

    print(
        f"Saved metrics to {results_path}"
    )

    print(
        f"Saved feature list to {feature_path}"
    )

    print(
        f"Saved best model info to "
        f"{best_info_path}"
    )

    return results


if __name__ == "__main__":
    train_models()