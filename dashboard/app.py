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
   LIGHT CONTENT SECTIONS
============================== */

.info-panel {
    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f8fafc
        );

    border:
        1px solid #e2e8f0;

    border-radius: 22px;

    padding:
        1.6rem
        1.8rem;

    margin-top: 1.4rem;
    margin-bottom: 1rem;

    box-shadow:
        0 10px 35px
        rgba(15,23,42,0.06);
}

.info-panel-kicker {
    color: #64748b;

    font-size: 0.72rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.12rem;
}

.info-panel-title {
    color: #0f172a;

    font-size: 1.65rem;

    font-weight: 750;

    margin-top: 0.25rem;
}

.info-panel-description {
    color: #64748b;

    font-size: 0.9rem;

    margin-top: 0.35rem;

    line-height: 1.55;
}


/* Streamlit metric cards */

[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fafc
        );

    border:
        1px solid #e2e8f0;

    padding: 1.25rem;

    border-radius: 18px;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,0.05);
}


/* Metric label */

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 650 !important;
}


/* Metric value */

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 750 !important;
}


/* Information message */

[data-testid="stAlert"] {
    border-radius: 16px;
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
        border-radius: 28px;
        padding: 2.4rem 2.7rem;

        background:
            radial-gradient(
                circle at 88% 12%,
                rgba(249,115,22,0.13),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #0f172a,
                #172033
            );

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 24px 70px rgba(0,0,0,0.28);

        margin-bottom: 1.6rem;

        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 2.1rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 0.35rem;
    }

    .hero-live {
        color: #86efac;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.22);
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04rem;
    }

    .hero-content {
        display: grid;
        grid-template-columns: 0.8fr 1.5fr;
        gap: 3rem;
        align-items: center;
    }

    .aqi-number-wrap {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .hero-aqi {
        font-size: 7rem;
        line-height: 0.9;
        font-weight: 800;
        letter-spacing: -0.28rem;
    }

    .aqi-caption {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.09rem;
        margin-top: 0.8rem;
    }

    .status-block {
        border-left: 1px solid rgba(255,255,255,0.10);
        padding-left: 2.2rem;
    }

    .status-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.09rem;
        font-weight: 700;
    }

    .hero-category {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.2;
        margin-top: 0.55rem;
    }

    .hero-message {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 0.8rem;
        max-width: 650px;
    }

    .hero-updated {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 1.1rem;
    }

    @media (max-width: 800px) {

        .hero-content {
            grid-template-columns: 1fr;
            gap: 1.8rem;
        }

        .status-block {
            border-left: none;
            padding-left: 0;
            border-top: 1px solid rgba(255,255,255,0.10);
            padding-top: 1.5rem;
        }

        .hero-aqi {
            font-size: 5.2rem;
        }

        .hero-category {
            font-size: 1.55rem;
        }
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
            " Sensitive individuals should consider reducing prolonged. "
            "or strenuous outdoor activity."
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

        <div class="hero-top">

            <div>
                <div class="hero-title">
                    Islamabad Air Quality
                </div>

                <div class="hero-subtitle">
                    Live atmospheric conditions
                    · Islamabad, Pakistan
                </div>
            </div>

            <div class="hero-live">
                ● LIVE
            </div>

        </div>

        <div class="hero-content">

            <div class="aqi-number-wrap">

                <div
                    class="hero-aqi"
                    style="color:{current_color};"
                >
                    {current_aqi}
                </div>

                <div class="aqi-caption">
                    Current US AQI
                </div>

            </div>

            <div class="status-block">

                <div class="status-label">
                    Air Quality Status
                </div>

                <div class="hero-category">
                    {current_category}
                </div>

                <div class="hero-message">
                    {current_message}
                </div>

                <div class="hero-updated">
                    Updated {current["time"]} · PKT
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
        (
            forecast_df,
            model_metadata,
        ) = generate_forecast()


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
# MODEL PERFORMANCE
# ==================================================

st.html(
    """
    <div class="info-panel">

        <div class="info-panel-kicker">
            Model Performance
        </div>

        <div class="info-panel-title">
            Prediction Accuracy
        </div>

        <div class="info-panel-description">
            Performance of the latest trained AQI
            forecasting model on the chronological
            holdout dataset.
        </div>

    </div>
    """
)

best_info = model_metadata.get(
    "best_model_info"
)


if best_info:

    model_name = best_info[
        "name"
    ]

    model_metrics = best_info[
        "metrics"
    ]

    perf_cols = st.columns(
        4,
        gap="medium",
    )


    with perf_cols[0]:

        st.metric(
            "Selected Model",
            model_name.replace(
                "_",
                " "
            ).title(),
        )


    with perf_cols[1]:

        st.metric(
            "R² Score",
            f"{model_metrics['r2']:.3f}",
        )


    with perf_cols[2]:

        st.metric(
            "RMSE",
            f"{model_metrics['rmse']:.2f}",
        )


    with perf_cols[3]:

        st.metric(
            "MAE",
            f"{model_metrics['mae']:.2f}",
        )


else:

    st.info(
        "Model performance metadata "
        "is not available."
    )
    
# ==================================================
# SHAP EXPLAINABILITY
# ==================================================

st.html(
    """
    <div class="info-panel">

        <div class="info-panel-kicker">
            Model Explainability
        </div>

        <div class="info-panel-title">
            What Drives the Forecast?
        </div>

        <div class="info-panel-description">
            SHAP analysis measures how strongly each
            feature influences the AQI prediction model.
            Larger values indicate greater overall impact.
        </div>

    </div>
    """
)


shap_data = model_metadata.get(
    "shap_importance"
)


if shap_data:

    shap_df = pd.DataFrame(
        shap_data
    )

    shap_df = (
        shap_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(10)
    )

    shap_df["feature"] = (
        shap_df["feature"]
        .str.replace(
            "_",
            " "
        )
        .str.title()
    )


    shap_chart = px.bar(
        shap_df,
        x="importance",
        y="feature",
        orientation="h",
    )


    shap_chart.update_traces(
        marker_line_width=0,
    )


    shap_chart.update_layout(

        height=460,

        xaxis_title=(
            "Average impact on model output"
        ),

        yaxis_title=None,

        plot_bgcolor="#ffffff",

        paper_bgcolor="#ffffff",

        font=dict(
            color="#334155",
        ),

        margin=dict(
            l=25,
            r=25,
            t=25,
            b=25,
        ),

        xaxis=dict(
            gridcolor="#edf2f7",
            zeroline=False,
        ),

        yaxis=dict(
            categoryorder="total ascending",
        ),
    )


    st.plotly_chart(
        shap_chart,
        use_container_width=True,
    )


    top_feature = (
        shap_df.iloc[0]["feature"]
    )


    st.info(
        f"The most influential feature "
        f"in the current model is "
        f"**{top_feature}**."
    )


else:

    st.info(
        "SHAP explainability data "
        "is not available for this model version."
    )

# ==================================================
# HEALTH ADVISORY
# ==================================================

st.html(
    """
    <div class="info-panel">

        <div class="info-panel-kicker">
            Health Guidance
        </div>

        <div class="info-panel-title">
            72-Hour Health Advisory
        </div>

        <div class="info-panel-description">
            Health guidance based on the highest
            AQI level predicted during the next
            three days.
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