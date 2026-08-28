import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    AIR_QUALITY_API_URL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
)

from src.inference import run_forecast


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


minimum_aqi = row["minimum_aqi"]
maximum_aqi = row["maximum_aqi"]


# ==================================================
# PROFESSIONAL STYLING
# ==================================================

st.html(
    """
    <style>

    /* ==============================
       GLOBAL
    ============================== */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #07111f 0px,
                #0b1727 560px,
                #f4f7fb 560px,
                #f4f7fb 100%
            );
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ==============================
       TOP BRAND
    ============================== */

    .brand-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
    }

    .brand-title {
        color: white;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02rem;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 0.2rem;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #d1fae5;
        background: rgba(16, 185, 129, 0.13);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
    }


    /* ==============================
       HERO
    ============================== */

    .hero-card {
        border-radius: 30px;
        padding: 2.5rem 2.8rem;

        background:
            radial-gradient(
                circle at 85% 10%,
                rgba(56, 189, 248, 0.14),
                transparent 32%
            ),
            radial-gradient(
                circle at 95% 85%,
                rgba(249, 115, 22, 0.14),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #111c2e,
                #162235
            );

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 28px 80px rgba(0,0,0,0.28);

        margin-bottom: 1.6rem;
    }

    .hero-location {
        color: #94a3b8;
        font-size: 0.83rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.18rem;
    }

    .hero-time {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 0.55rem;
    }

    .hero-main {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        margin-top: 1.6rem;
        gap: 2rem;
    }

    .hero-aqi {
        font-size: 7.4rem;
        line-height: 0.95;
        font-weight: 800;
        letter-spacing: -0.35rem;
    }

    .aqi-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        letter-spacing: 0.08rem;
    }

    .hero-status {
        max-width: 620px;
        padding-bottom: 0.5rem;
    }

    .hero-category {
        color: white;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.2;
    }

    .hero-message {
        color: #cbd5e1;
        margin-top: 0.8rem;
        font-size: 1rem;
        line-height: 1.65;
    }


    /* ==============================
       KPI CARDS
    ============================== */

    .metric-card {
        background: white;

        padding:
            1.4rem
            1.45rem;

        border-radius: 20px;

        border:
            1px solid
            #e6ebf1;

        box-shadow:
            0 10px 35px
            rgba(15,23,42,0.06);

        min-height: 150px;

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);

        box-shadow:
            0 15px 40px
            rgba(15,23,42,0.10);
    }

    .metric-icon {
        font-size: 1.25rem;
        margin-bottom: 0.8rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 2.3rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }

    .metric-unit {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }


    /* ==============================
       SECTIONS
    ============================== */

    .section-header {
        margin-top: 2.7rem;
        margin-bottom: 1.1rem;
    }

    .section-kicker {
        color: #64748b;
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12rem;
    }

    .section-title {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    .section-description {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }


    /* ==============================
       FORECAST CARDS
    ============================== */

    .forecast-card {
        background: white;
        padding: 1.7rem;
        border-radius: 22px;

        border:
            1px solid
            #e6ebf1;

        box-shadow:
            0 10px 35px
            rgba(15,23,42,0.06);

        min-height: 245px;

        position: relative;
        overflow: hidden;
    }

    .forecast-accent {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
    }

    .forecast-day {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
    }

    .forecast-date {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }

    .forecast-aqi {
        font-size: 3.1rem;
        font-weight: 800;
        margin-top: 1rem;
    }

    .forecast-category {
        color: #334155;
        font-size: 0.9rem;
        font-weight: 650;
        min-height: 42px;
    }

    .forecast-range {
        display: flex;
        justify-content: space-between;
        margin-top: 1.1rem;
        padding-top: 1rem;
        border-top: 1px solid #eef2f7;
        color: #64748b;
        font-size: 0.83rem;
    }

    .forecast-range strong {
        color: #0f172a;
    }


    /* ==============================
       CHART AREA
    ============================== */

    .chart-card {
        background: white;
        padding: 0.6rem 1rem 1rem 1rem;
        border-radius: 22px;
        border: 1px solid #e6ebf1;

        box-shadow:
            0 10px 35px
            rgba(15,23,42,0.06);
    }


    /* ==============================
       FOOTER
    ============================== */

    .custom-footer {
        margin-top: 3rem;
        padding-top: 1.4rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.78rem;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
    }


    /* ==============================
       BUTTON
    ============================== */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #dbe3ec;
        background: white;
        color: #0f172a;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #94a3b8;
        color: #0f172a;
    }


    /* ==============================
       MOBILE
    ============================== */

    @media (max-width: 800px) {

        .hero-main {
            flex-direction: column;
            align-items: flex-start;
        }

        .hero-aqi {
            font-size: 5.5rem;
        }

        .hero-category {
            font-size: 1.5rem;
        }

        .hero-card {
            padding: 1.8rem;
        }
    }

    </style>
    """
)


# ==================================================
# AQI HELPERS
# ==================================================

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
            "Air quality is satisfactory. "
            "Outdoor activities can be enjoyed normally."
        )

    if aqi <= 100:
        return (
            "Air quality is acceptable for most people."
        )

    if aqi <= 150:
        return (
            "Sensitive groups may experience health effects. "
            "Consider reducing prolonged outdoor activity."
        )

    if aqi <= 200:
        return (
            "Some members of the general public may experience "
            "health effects. Sensitive groups should limit exposure."
        )

    if aqi <= 300:
        return (
            "Health risk is increased for everyone. "
            "Outdoor exposure should be reduced."
        )

    return (
        "Emergency-level air pollution conditions. "
        "Avoid outdoor exposure where possible."
    )


def get_aqi_color(aqi):

    if aqi <= 50:
        return "#10b981"

    if aqi <= 100:
        return "#eab308"

    if aqi <= 150:
        return "#f97316"

    if aqi <= 200:
        return "#ef4444"

    if aqi <= 300:
        return "#8b5cf6"

    return "#991b1b"


# ==================================================
# SECRETS
# ==================================================

def configure_secrets():

    try:

        if "HOPSWORKS_API_KEY" in st.secrets:

            os.environ[
                "HOPSWORKS_API_KEY"
            ] = st.secrets[
                "HOPSWORKS_API_KEY"
            ]

        if "HOPSWORKS_PROJECT" in st.secrets:

            os.environ[
                "HOPSWORKS_PROJECT"
            ] = st.secrets[
                "HOPSWORKS_PROJECT"
            ]

    except FileNotFoundError:
        pass


# ==================================================
# CURRENT AQI
# ==================================================

@st.cache_data(ttl=900)
def fetch_current_aqi():

    params = {

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "current": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),

        "timezone":
            TIMEZONE,
    }

    response = requests.get(
        AIR_QUALITY_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    current = response.json()[
        "current"
    ]

    return {

        "time":
            current["time"],

        "aqi":
            float(
                current["us_aqi"]
            ),

        "pm2_5":
            float(
                current["pm2_5"]
            ),

        "pm10":
            float(
                current["pm10"]
            ),

        "carbon_monoxide":
            float(
                current[
                    "carbon_monoxide"
                ]
            ),

        "nitrogen_dioxide":
            float(
                current[
                    "nitrogen_dioxide"
                ]
            ),

        "sulphur_dioxide":
            float(
                current[
                    "sulphur_dioxide"
                ]
            ),

        "ozone":
            float(
                current["ozone"]
            ),
    }


# ==================================================
# MODEL FORECAST
# ==================================================

@st.cache_data(ttl=3600)
def generate_forecast():

    configure_secrets()

    return run_forecast()


def create_daily_summary(df):

    daily_df = (
        df.copy()
        .reset_index(drop=True)
    )

    daily_df[
        "forecast_day"
    ] = (
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
        daily["date"]
        .dt
        .date
    )

    return daily


# ==================================================
# LOAD CURRENT DATA
# ==================================================

try:

    current = fetch_current_aqi()

except Exception as error:

    st.error(
        "Unable to retrieve current Islamabad air-quality data."
    )

    with st.expander(
        "Technical details"
    ):
        st.code(
            str(error)
        )

    st.stop()


current_aqi = round(
    current["aqi"]
)

current_category = (
    get_aqi_category(
        current_aqi
    )
)

current_message = (
    get_aqi_message(
        current_aqi
    )
)

current_color = (
    get_aqi_color(
        current_aqi
    )
)


# ==================================================
# TOP BRAND
# ==================================================

st.html(
    """
    <div class="brand-row">

        <div>

            <div class="brand-title">
                Pearls AQI Predictor
            </div>

            <div class="brand-subtitle">
                AI-powered air quality monitoring and forecasting
            </div>

        </div>

        <div class="live-badge">
            <span class="live-dot"></span>
            LIVE DATA
        </div>

    </div>
    """
)


# ==================================================
# CURRENT AQI HERO
# ==================================================

st.html(
    f"""
    <div class="hero-card">

        <div class="hero-location">
            Islamabad, Pakistan
            · Current Air Quality
        </div>

        <div class="hero-time">
            Last updated:
            {current["time"]}
        </div>

        <div class="hero-main">

            <div>

                <div
                    class="hero-aqi"
                    style="
                        color:
                        {current_color};
                    "
                >
                    {current_aqi}
                </div>

                <div class="aqi-label">
                    US AQI
                </div>

            </div>

            <div class="hero-status">

                <div
                    class="hero-category"
                >
                    {current_category}
                </div>

                <div
                    class="hero-message"
                >
                    {current_message}
                </div>

            </div>

        </div>

    </div>
    """
)


# ==================================================
# POLLUTANT KPI CARDS
# ==================================================

metric_columns = st.columns(
    4,
    gap="medium",
)


metric_data = [

    (
        "PM2.5",
        current["pm2_5"],
        "Fine particles",
        "◉",
    ),

    (
        "PM10",
        current["pm10"],
        "Coarse particles",
        "◌",
    ),

    (
        "Nitrogen Dioxide",
        current[
            "nitrogen_dioxide"
        ],
        "NO₂ concentration",
        "△",
    ),

    (
        "Ozone",
        current["ozone"],
        "O₃ concentration",
        "◇",
    ),
]


for column, item in zip(
    metric_columns,
    metric_data,
):

    name, value, description, icon = item

    with column:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-icon">
                    {icon}
                </div>

                <div class="metric-label">
                    {name}
                </div>

                <div class="metric-value">
                    {value:.1f}
                </div>

                <div class="metric-unit">
                    μg/m³
                    · {description}
                </div>

            </div>
            """
        )


# ==================================================
# LOAD MODEL FORECAST
# ==================================================

st.html(
    """
    <div class="section-header">

        <div class="section-kicker">
            Machine Learning Forecast
        </div>

        <div class="section-title">
            Next 3 Days
        </div>

        <div class="section-description">
            Rolling 72-hour AQI forecast generated
            using the trained Islamabad AQI model.
        </div>

    </div>
    """
)


with st.spinner(
    "Generating latest 72-hour forecast..."
):

    try:

        forecast_df = (
            generate_forecast()
        )

    except Exception as error:

        st.error(
            "Unable to generate the forecast."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(
                str(error)
            )

        st.stop()


forecast_df[
    "timestamp"
] = pd.to_datetime(
    forecast_df[
        "timestamp"
    ]
)


daily_df = (
    create_daily_summary(
        forecast_df
    )
)


# ==================================================
# REFRESH BUTTON
# ==================================================

button_col, _ = st.columns(
    [1, 5]
)

with button_col:

    if st.button(
        "↻ Refresh data",
        use_container_width=True,
    ):

        generate_forecast.clear()

        fetch_current_aqi.clear()

        st.rerun()


# ==================================================
# 3-DAY FORECAST CARDS
# ==================================================

forecast_columns = st.columns(
    3,
    gap="medium",
)


for index, row in daily_df.iterrows():

    average_aqi = round(
        row["average_aqi"]
    )

    minimum_aqi = row[
        "minimum_aqi"
    ]

    maximum_aqi = row[
        "maximum_aqi"
    ]

    forecast_day = int(
        row["forecast_day"]
    )

    forecast_date = row[
        "date"
    ]

    category = get_aqi_category(
        average_aqi
    )

    color = get_aqi_color(
        average_aqi
    )

    with forecast_columns[index]:

        st.html(
            f"""
            <div class="forecast-card">

                <div
                    class="forecast-accent"
                    style="background:{color};"
                ></div>

                <div class="forecast-day">
                    Forecast Day {forecast_day}
                </div>

                <div class="forecast-date">
                    {forecast_date}
                </div>

                <div
                    class="forecast-aqi"
                    style="color:{color};"
                >
                    {average_aqi}
                </div>

                <div class="forecast-category">
                    {category}
                </div>

                <div class="forecast-range">

                    <span>
                        Minimum
                        <strong>
                            {minimum_aqi:.0f}
                        </strong>
                    </span>

                    <span>
                        Maximum
                        <strong>
                            {maximum_aqi:.0f}
                        </strong>
                    </span>

                </div>

            </div>
            """
        )

# ==================================================
# 72-HOUR TREND
# ==================================================

st.html(
    """
    <div class="section-header">

        <div class="section-kicker">
            Forecast Trend
        </div>

        <div class="section-title">
            72-Hour AQI Trend
        </div>

        <div class="section-description">
            Hour-by-hour AQI predictions for
            the next three days.
        </div>

    </div>
    """
)


chart = px.area(
    forecast_df,
    x="timestamp",
    y="predicted_aqi",
)


chart.update_traces(
    line=dict(
        width=3,
    ),
)


chart.update_layout(

    height=450,

    xaxis_title=None,

    yaxis_title="Predicted AQI",

    hovermode="x unified",

    plot_bgcolor="#ffffff",

    paper_bgcolor="#ffffff",

    font=dict(
        color="#475569"
    ),

    margin=dict(
        l=25,
        r=25,
        t=30,
        b=20,
    ),

    xaxis=dict(
        showgrid=False,
    ),

    yaxis=dict(
        gridcolor="#eef2f7",
        zeroline=False,
    ),
)


st.plotly_chart(
    chart,
    use_container_width=True,
)


# ==================================================
# HEALTH ADVISORY
# ==================================================

st.html(
    """
    <div class="section-header">

        <div class="section-kicker">
            Health Information
        </div>

        <div class="section-title">
            Forecast Advisory
        </div>

    </div>
    """
)


maximum_aqi = (
    forecast_df[
        "predicted_aqi"
    ].max()
)


if maximum_aqi > 300:

    st.error(
        "Hazardous AQI is predicted during "
        "the next 72 hours. Avoid outdoor exposure."
    )

elif maximum_aqi > 200:

    st.error(
        "Very unhealthy air quality is predicted. "
        "Outdoor activity should be significantly reduced."
    )

elif maximum_aqi > 150:

    st.warning(
        "Unhealthy AQI levels are predicted during "
        "part of the next 72 hours. Sensitive groups "
        "should reduce prolonged outdoor exposure."
    )

elif maximum_aqi > 100:

    st.warning(
        "AQI may become unhealthy for sensitive "
        "groups during the next 72 hours."
    )

else:

    st.success(
        "AQI is expected to remain within the "
        "Good to Moderate range."
    )


# ==================================================
# HOURLY DATA
# ==================================================

with st.expander(
    "View detailed hourly forecast"
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
        ]
        .apply(
            get_aqi_category
        )
    )


    display_df = (
        display_df.rename(
            columns={

                "timestamp":
                    "Time",

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
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# FOOTER
# ==================================================

st.html(
    """
    <div class="custom-footer">

        <div>
            Pearls AQI Predictor
        </div>

        <div>
            Current data: Open-Meteo / CAMS
            · Forecast: ML model
        </div>

    </div>
    """
)