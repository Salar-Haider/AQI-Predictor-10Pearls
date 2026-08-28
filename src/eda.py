import os

import hopsworks
import pandas as pd

from dotenv import load_dotenv


FEATURE_GROUP_NAME = "aqi_hourly_features"
FEATURE_GROUP_VERSION = 4


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


def load_historical_data():
    """Load historical AQI data from Feature Store."""

    project = connect_to_hopsworks()

    feature_store = (
        project.get_feature_store()
    )

    feature_group = (
        feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )
    )

    df = feature_group.read(
        dataframe_type="pandas"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    return df