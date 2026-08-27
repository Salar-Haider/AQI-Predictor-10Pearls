
import os
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.config import (
    AIR_QUALITY_API_URL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
)

from src.inference import run_forecast


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background:
            linear-gradient(
                180deg,
                #0b1120 0px,
                #111827 460px,
                #f5f7fb 460px,
                #f5f7fb 100%
            );
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit default header decoration */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .dashboard-title {
        font-size: 1.05rem;
        letter-spacing: 0.22rem;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }

    .hero-card {
        padding: 2.2rem 2.5rem;
        border-radius: 28px;
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(249,115,22,0.15),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #111827,
                #1f2937
            );

        border: 1px solid rgba(255,255,255,0.08);
        box-shadow:
            0 18px 60px rgba(0,0,0,0.30);
        margin-bottom: 1.4rem;
    }

    .hero-aqi {
        font-size: 7rem;
        font-weight: 700;
        line-height: 1;
        margin: 0;
    }

    .hero-category {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    .hero-message {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.7rem;
    }

    .hero-time {
        color: #94a3b8;
        font-size: 0.85rem;
        letter-spacing: 0.12rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: white;
        padding: 1.4rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow:
            0 7px 25px rgba(15,23,42,0.06);
        min-height: 180px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
    }

    .metric-value {
        color: #111827;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    .forecast-card {
        background: white;
        padding: 1.6rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow:
            0 7px 25px rgba(15,23,42,0.06);
        min-height: 220px;
    }

    .forecast-day {
        color: #64748b;
        font-size: 0.9rem;
    }

    .forecast-aqi {
        color: #111827;
        font-size: 2.8rem;
        font-weight: 700;
        margin-top: 0.6rem;
    }

    .forecast-category {
        margin-top: 0.7rem;
        font-weight: 600;
        color: #334155;
    }

    .section-title {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# AQI helper functions
# --------------------------------------------------

def get_aqi_category(aqi):
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


def get_aqi_message(aqi):
    if aqi <= 50:
        return (
            "Air quality is satisfactory and poses "
            "little or no health risk."
        )

    if aqi <= 100:
        return (
            "Air quality is acceptable for most people."
        )

    if aqi <= 150:
        return (
            "Sensitive groups may experience health effects."
        )

    if aqi <= 200:
        return (
            "Some members of the general public may "
            "experience health effects."
        )

    if aqi <= 300:
        return (
            "Health risk is increased for everyone."
        )

    return (
        "Health warning of emergency conditions."
    )


def get_aqi_color(aqi):
    if aqi <= 50:
        return "#22c55e"

    if aqi <= 100:
        return "#eab308"

    if aqi <= 150:
        return "#f97316"

    if aqi <= 200:
        return "#ef4444"

    if aqi <= 300:
        return "#a855f7"

    return "#7f1d1d"


# --------------------------------------------------
# Secrets
# --------------------------------------------------

def configure_secrets():
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
        pass


# --------------------------------------------------
# Current AQI
# --------------------------------------------------

@st.cache_data(ttl=900)
def fetch_current_aqi():
    """
    Fetch current Islamabad AQI and pollutant values
    directly from Open-Meteo.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "timezone": TIMEZONE,
    }

    response = requests.get(
        AIR_QUALITY_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    current = data["current"]

    return {
        "time": current["time"],
        "aqi": float(current["us_aqi"]),
        "pm2_5": float(current["pm2_5"]),
        "pm10": float(current["pm10"]),
        "carbon_monoxide": float(
            current["carbon_monoxide"]
        ),
        "nitrogen_dioxide": float(
            current["nitrogen_dioxide"]
        ),
        "sulphur_dioxide": float(
            current["sulphur_dioxide"]
        ),
        "ozone": float(current["ozone"]),
    }


# --------------------------------------------------
# Forecast
# --------------------------------------------------

@st.cache_data(ttl=3600)
def generate_forecast():
    configure_secrets()

    return run_forecast()


def create_daily_summary(df):
    """
    Split the 72-hour forecast into
    exactly 3 x 24-hour periods.
    """

    daily_df = (
        df.copy()
        .reset_index(drop=True)
    )

    daily_df["forecast_day"] = (
        daily_df.index // 24
    ) + 1

    daily = (
        daily_df.groupby(
            "forecast_day"
        )
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


# --------------------------------------------------
# Load current AQI
# --------------------------------------------------

try:
    current = fetch_current_aqi()

except Exception as error:
    st.error(
        "Unable to retrieve the current Islamabad AQI."
    )

    with st.expander(
        "Technical details"
    ):
        st.code(str(error))

    st.stop()


current_aqi = round(
    current["aqi"]
)

current_category = get_aqi_category(
    current_aqi
)

current_message = get_aqi_message(
    current_aqi
)

current_color = get_aqi_color(
    current_aqi
)


# --------------------------------------------------
# Hero / Current AQI
# --------------------------------------------------

st.markdown(
    dedent(
        f"""
        <div class="hero-card">
            <div class="dashboard-title">
                ISLAMABAD, PAKISTAN · CURRENT AIR QUALITY
            </div>

            <div class="hero-time">
                UPDATED {current["time"]}
            </div>

            <div
                class="hero-aqi"
                style="color:{current_color};"
            >
                {current_aqi}
            </div>

            <div class="hero-category">
                {current_category}
            </div>

            <div class="hero-message">
                {current_message}
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Current pollutant cards
# --------------------------------------------------
metric_columns = st.columns(
    4
)


with metric_columns[0]:
    st.markdown(
        dedent(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    PM2.5
                </div>

                <div class="metric-value">
                    {current["pm2_5"]:.1f}
                </div>

                <div class="metric-label">
                    μg/m³
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


with metric_columns[1]:
    st.markdown(
        dedent(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    PM10
                </div>

                <div class="metric-value">
                    {current["pm10"]:.1f}
                </div>

                <div class="metric-label">
                    μg/m³
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


with metric_columns[2]:
    st.markdown(
        dedent(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Nitrogen Dioxide
                </div>

                <div class="metric-value">
                    {current["nitrogen_dioxide"]:.1f}
                </div>

                <div class="metric-label">
                    μg/m³
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


with metric_columns[3]:
    st.markdown(
        dedent(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Ozone
                </div>

                <div class="metric-value">
                    {current["ozone"]:.1f}
                </div>

                <div class="metric-label">
                    μg/m³
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# Forecast loading
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '3-Day AQI Forecast'
    '</div>',
    unsafe_allow_html=True,
)


with st.spinner(
    "Generating latest 72-hour forecast..."
):

    try:
        forecast_df = generate_forecast()

    except Exception as error:

        st.error(
            "Unable to generate the forecast."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(str(error))

        st.stop()


forecast_df["timestamp"] = (
    pd.to_datetime(
        forecast_df["timestamp"]
    )
)

daily_df = create_daily_summary(
    forecast_df
)


# --------------------------------------------------
# Refresh
# --------------------------------------------------

refresh_col, _ = st.columns(
    [1, 5]
)

with refresh_col:

    if st.button(
        "↻ Refresh Data",
        use_container_width=True,
    ):
        generate_forecast.clear()
        fetch_current_aqi.clear()
        st.rerun()


# --------------------------------------------------
# 3 forecast cards
# --------------------------------------------------

forecast_columns = st.columns(
    3
)


for index, row in daily_df.iterrows():

    average_aqi = round(
        row["average_aqi"]
    )

    category = get_aqi_category(
        average_aqi
    )

    color = get_aqi_color(
        average_aqi
    )

    with forecast_columns[index]:

        st.markdown(
            f"""
            <div class="forecast-card">

                <div class="forecast-day">
                    DAY {int(row["forecast_day"])}
                    · {row["date"]}
                </div>

                <div
                    class="forecast-aqi"
                    style="color:{color};"
                >
                    {average_aqi} AQI
                </div>

                <div class="forecast-category">
                    {category}
                </div>

                <br>

                <div>
                    Minimum:
                    <strong>
                        {row["minimum_aqi"]:.0f}
                    </strong>
                </div>

                <div>
                    Maximum:
                    <strong>
                        {row["maximum_aqi"]:.0f}
                    </strong>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# 72-hour forecast chart
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '72-Hour AQI Trend'
    '</div>',
    unsafe_allow_html=True,
)


chart = px.line(
    forecast_df,
    x="timestamp",
    y="predicted_aqi",
    markers=False,
)


chart.update_traces(
    line=dict(
        width=3
    )
)


chart.update_layout(
    height=430,
    xaxis_title=None,
    yaxis_title="AQI",
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20,
    ),
)


st.plotly_chart(
    chart,
    use_container_width=True,
)


# --------------------------------------------------
# Health alert
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Forecast Health Alert'
    '</div>',
    unsafe_allow_html=True,
)


maximum_aqi = (
    forecast_df[
        "predicted_aqi"
    ].max()
)


if maximum_aqi > 300:

    st.error(
        "Hazardous AQI is predicted "
        "during the next 72 hours."
    )

elif maximum_aqi > 200:

    st.error(
        "Very unhealthy AQI is predicted "
        "during the next 72 hours."
    )

elif maximum_aqi > 150:

    st.warning(
        "Unhealthy AQI is predicted during "
        "part of the next 72 hours."
    )

elif maximum_aqi > 100:

    st.warning(
        "AQI may be unhealthy for sensitive "
        "groups during the next 72 hours."
    )

else:

    st.success(
        "AQI is expected to remain in the "
        "Good to Moderate range."
    )


# --------------------------------------------------
# Hourly data
# --------------------------------------------------

with st.expander(
    "View Hourly Forecast Data"
):

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

    display_df[
        "AQI Category"
    ] = (
        display_df[
            "predicted_aqi"
        ].apply(
            get_aqi_category
        )
    )

    display_df = display_df.rename(
        columns={
            "timestamp": "Time",
            "predicted_aqi":
                "Predicted AQI",
            "temperature_2m":
                "Temperature °C",
            "relative_humidity_2m":
                "Humidity %",
            "pm2_5":
                "PM2.5",
            "pm10":
                "PM10",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Current air-quality data: Open-Meteo / CAMS. "
    "72-hour AQI forecast: Pearls AQI Predictor model."
)