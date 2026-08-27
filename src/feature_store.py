# src/feature_store.py

import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from src.config import PROCESSED_DATA_PATH


FEATURE_GROUP_NAME = "aqi_hourly_features"
FEATURE_GROUP_VERSION = 1


def connect_to_hopsworks():
    """Connect to the Hopsworks project."""

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is missing from .env")

    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT is missing from .env")

    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key,
        engine="python",
    )

    return project


def upload_dataframe(df):
    """Upload a DataFrame to the Hopsworks Feature Store."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    project = connect_to_hopsworks()

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            "Hourly weather and air-quality features "
            "for Islamabad"
        ),
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
    )

    feature_group.insert(
        df,
        wait=True,
    )

    print(
        f"Uploaded {len(df)} rows to "
        f"{FEATURE_GROUP_NAME} version "
        f"{FEATURE_GROUP_VERSION}"
    )
    
    
    
def upload_features():
    """Upload normal processed feature data."""

    df = pd.read_csv(PROCESSED_DATA_PATH)

    upload_dataframe(df)
    

if __name__ == "__main__":
    upload_features()