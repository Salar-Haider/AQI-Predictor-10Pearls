import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.inference import run_forecast


def get_aqi_category(aqi):
    """Return AQI category."""

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def configure_secrets():
    """
    Make Streamlit Cloud secrets available to
    the existing inference code as environment variables.
    """

    try:
        if "HOPSWORKS_API_KEY" in st.secrets:
            os.environ["HOPSWORKS_API_KEY"] = (
                st.secrets["HOPSWORKS_API_KEY"]
            )

        if "HOPSWORKS_PROJECT" in st.secrets:
            os.environ["HOPSWORKS_PROJECT"] = (
                st.secrets["HOPSWORKS_PROJECT"]
            )

    except FileNotFoundError:
        # Local development can still use .env
        pass


@st.cache_data(ttl=3600)
def generate_forecast():
    """
    Run inference and cache the result for one hour.

    This prevents Hopsworks/model/API calls every time
    Streamlit reruns the page.
    """

    configure_secrets()

    forecast_df = run_forecast()

    return forecast_df


def create_daily_summary(df):
    """Split the 72-hour forecast into 3 x 24-hour periods."""

    daily_df = df.copy().reset_index(drop=True)

    daily_df["forecast_day"] = (
        daily_df.index // 24
    ) + 1

    daily = (
        daily_df.groupby("forecast_day")
        .agg(
            date=(
                "timestamp",
                "first",
            ),
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

    daily["date"] = (
        daily["date"].dt.date
    )

    return daily


st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)


st.title("Islamabad AQI Predictor")

st.write(
    "72-hour air quality forecast for Islamabad "
    "using weather, pollutant and historical AQI features."
)


# --------------------------------------------------
# Forecast
# --------------------------------------------------

with st.spinner(
    "Generating latest 72-hour AQI forecast..."
):

    try:

        forecast_df = generate_forecast()

    except Exception as error:

        st.error(
            "Unable to generate the live forecast."
        )

        st.write(
            "The cloud inference service could not "
            "complete the prediction."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(str(error))

        st.stop()


forecast_df["timestamp"] = pd.to_datetime(
    forecast_df["timestamp"]
)

daily_df = create_daily_summary(
    forecast_df
)


# --------------------------------------------------
# Refresh button
# --------------------------------------------------

col1, col2 = st.columns(
    [1, 5]
)

with col1:

    if st.button(
        "Refresh Forecast"
    ):

        generate_forecast.clear()

        st.rerun()


# --------------------------------------------------
# Daily forecast
# --------------------------------------------------

st.subheader(
    "3-Day AQI Forecast"
)

columns = st.columns(
    len(daily_df)
)


for index, row in daily_df.iterrows():

    average_aqi = round(
        row["average_aqi"]
    )

    category = get_aqi_category(
        average_aqi
    )

    with columns[index]:

        st.metric(
            label=(
                f"Day {int(row['forecast_day'])} "
                f"— {row['date']}"
            ),
            value=f"{average_aqi} AQI",
        )

        st.write(
            f"**{category}**"
        )

        st.write(
            f"Min: "
            f"{row['minimum_aqi']:.0f}"
        )

        st.write(
            f"Max: "
            f"{row['maximum_aqi']:.0f}"
        )


# --------------------------------------------------
# 72-hour chart
# --------------------------------------------------

st.subheader(
    "72-Hour AQI Forecast"
)


chart = px.line(
    forecast_df,
    x="timestamp",
    y="predicted_aqi",
    markers=True,
    title=(
        "Predicted AQI Over "
        "the Next 72 Hours"
    ),
)


chart.update_layout(
    xaxis_title="Time",
    yaxis_title="Predicted AQI",
)


st.plotly_chart(
    chart,
    use_container_width=True,
)


# --------------------------------------------------
# Health alert
# --------------------------------------------------

st.subheader(
    "AQI Health Alert"
)


maximum_aqi = (
    forecast_df[
        "predicted_aqi"
    ].max()
)


if maximum_aqi > 300:

    st.error(
        "Hazardous AQI is predicted "
        "during the next 3 days."
    )

elif maximum_aqi > 200:

    st.error(
        "Very unhealthy AQI is predicted "
        "during the next 3 days."
    )

elif maximum_aqi > 150:

    st.warning(
        "Unhealthy AQI is predicted "
        "during the next 3 days."
    )

elif maximum_aqi > 100:

    st.warning(
        "AQI may be unhealthy "
        "for sensitive groups."
    )

else:

    st.success(
        "AQI is expected to remain "
        "in the Good to Moderate range."
    )


# --------------------------------------------------
# Hourly table
# --------------------------------------------------

st.subheader(
    "Hourly Forecast Data"
)


display_df = forecast_df[
    [
        "timestamp",
        "predicted_aqi",
        "temperature_2m",
        "relative_humidity_2m",
        "pm2_5",
        "pm10",
    ]
].copy()


display_df["AQI Category"] = (
    display_df[
        "predicted_aqi"
    ].apply(
        get_aqi_category
    )
)


display_df = display_df.rename(
    columns={
        "timestamp": "Time",
        "predicted_aqi": "Predicted AQI",
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "pm2_5": "PM2.5",
        "pm10": "PM10",
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
)