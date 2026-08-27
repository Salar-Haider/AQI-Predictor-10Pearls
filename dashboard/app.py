# dashboard/app.py

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


FORECAST_PATH = Path(
    "data/processed/islamabad_72_hour_forecast.csv"
)


def get_aqi_category(aqi):
    """Return AQI category text."""

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


def load_forecast():
    """Load the generated 72-hour forecast."""

    if not FORECAST_PATH.exists():
        st.error(
            "Forecast file not found. "
            "Run the inference pipeline first."
        )
        st.stop()

    df = pd.read_csv(
        FORECAST_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


def create_daily_summary(df):
    """Create daily AQI summary."""

    daily_df = df.copy()

    daily_df["date"] = (
        daily_df["timestamp"].dt.date
    )

    daily = (
        daily_df.groupby("date")
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


forecast_df = load_forecast()

daily_df = create_daily_summary(
    forecast_df
)


st.subheader("3-Day AQI Forecast")


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
            label=str(row["date"]),
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


st.subheader("72-Hour AQI Forecast")


chart = px.line(
    forecast_df,
    x="timestamp",
    y="predicted_aqi",
    markers=True,
    title="Predicted AQI Over the Next 72 Hours",
)

chart.update_layout(
    xaxis_title="Time",
    yaxis_title="Predicted AQI",
)

st.plotly_chart(
    chart,
    use_container_width=True,
)


st.subheader("AQI Health Alert")


maximum_aqi = forecast_df[
    "predicted_aqi"
].max()


if maximum_aqi > 300:

    st.error(
        "Hazardous AQI is predicted during "
        "the next 3 days."
    )

elif maximum_aqi > 200:

    st.error(
        "Very unhealthy AQI is predicted during "
        "the next 3 days."
    )

elif maximum_aqi > 150:

    st.warning(
        "Unhealthy AQI is predicted during "
        "the next 3 days."
    )

elif maximum_aqi > 100:

    st.warning(
        "AQI may be unhealthy for sensitive groups."
    )

else:

    st.success(
        "AQI is expected to remain in the "
        "Good to Moderate range."
    )


st.subheader("Hourly Forecast Data")


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
    display_df["predicted_aqi"]
    .apply(get_aqi_category)
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