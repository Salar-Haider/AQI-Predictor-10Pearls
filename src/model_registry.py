import json
import os
from pathlib import Path

import hopsworks
from dotenv import load_dotenv


MODEL_DIR = Path("models")

MODEL_NAME = "islamabad_aqi_model"


def connect_to_hopsworks():
    """Connect to Hopsworks."""

    load_dotenv()

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    project_name = os.getenv(
        "HOPSWORKS_PROJECT"
    )

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is missing"
        )

    if not project_name:
        raise ValueError(
            "HOPSWORKS_PROJECT is missing"
        )

    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key,
        engine="python",
    )

    return project


def register_model():
    """Register the dynamically selected best AQI model."""

    model_path = (
        MODEL_DIR
        / "best_model.pkl"
    )

    features_path = (
        MODEL_DIR
        / "feature_columns.json"
    )

    best_info_path = (
        MODEL_DIR
        / "best_model_info.json"
    )

    shap_path = (
        MODEL_DIR
        / "shap_importance.json"
    )

    required_files = [
        model_path,
        features_path,
        best_info_path,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"{file_path} not found. "
                "Run the training pipeline first."
            )


    with open(
        best_info_path,
        "r",
    ) as file:

        best_info = json.load(
            file
        )


    best_model_name = (
        best_info["name"]
    )

    best_metrics = (
        best_info["metrics"]
    )


    print(
        f"Best trained model: "
        f"{best_model_name}"
    )

    print(
        f"RMSE: "
        f"{best_metrics['rmse']:.3f}"
    )

    print(
        f"MAE: "
        f"{best_metrics['mae']:.3f}"
    )

    print(
        f"R²: "
        f"{best_metrics['r2']:.3f}"
    )


    project = (
        connect_to_hopsworks()
    )

    model_registry = (
        project.get_model_registry()
    )


    model = (
        model_registry.python.create_model(

            name=MODEL_NAME,

            metrics={
                "rmse":
                    best_metrics["rmse"],

                "mae":
                    best_metrics["mae"],

                "r2":
                    best_metrics["r2"],
            },

            description=(
                "Best AQI forecasting model "
                "for Islamabad. "
                f"Selected model: "
                f"{best_model_name}. "
                "Uses weather, pollutant, "
                "temporal, AQI lag and "
                "rolling features."
            ),
        )
    )


    registered_model = (
        model.save(
            str(MODEL_DIR)
        )
    )


    print(
        "\nModel registered successfully."
    )

    print(
        f"Name: "
        f"{registered_model.name}"
    )

    print(
        f"Version: "
        f"{registered_model.version}"
    )

    if shap_path.exists():

        print(
            "SHAP artifact included."
        )


if __name__ == "__main__":
    register_model()