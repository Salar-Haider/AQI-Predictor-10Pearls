# src/model_registry.py

import json
import os
from pathlib import Path

import hopsworks
from dotenv import load_dotenv


MODEL_DIR = Path("models")

MODEL_NAME = "aqi_gradient_boosting"
MODEL_VERSION = 1


def connect_to_hopsworks():
    """Connect to Hopsworks."""

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


def register_model():
    """Register the trained AQI model in Hopsworks."""

    model_path = MODEL_DIR / "best_model.pkl"
    metrics_path = MODEL_DIR / "metrics.json"
    features_path = MODEL_DIR / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "models/best_model.pkl not found. "
            "Run the training pipeline first."
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            "models/metrics.json not found."
        )

    if not features_path.exists():
        raise FileNotFoundError(
            "models/feature_columns.json not found."
        )

    with open(metrics_path, "r") as file:
        all_metrics = json.load(file)

    best_metrics = all_metrics[
        "gradient_boosting"
    ]

    project = connect_to_hopsworks()

    model_registry = project.get_model_registry()

    model = model_registry.python.create_model(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        metrics={
            "rmse": best_metrics["rmse"],
            "mae": best_metrics["mae"],
            "r2": best_metrics["r2"],
        },
        description=(
            "Gradient Boosting model for Islamabad "
            "AQI prediction using weather, pollutant, "
            "temporal, lag and rolling features."
        ),
    )

    model.save(str(MODEL_DIR))

    print(
        f"Registered {MODEL_NAME} "
        f"version {MODEL_VERSION}"
    )


if __name__ == "__main__":
    register_model()