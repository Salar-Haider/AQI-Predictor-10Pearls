
# 🌍 Islamabad AQI Predictor

> **A 100% Serverless, End-to-End Air Quality Index Forecasting System for Islamabad, Pakistan**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-orange)

## 📌 Overview

The **Islamabad AQI Predictor** is a fully automated machine learning application that monitors current air quality and forecasts hourly AQI values for the next 72 hours.

The system combines data collection, feature engineering, model training, model versioning, explainability, automation, and interactive visualization in a serverless workflow.

### Key Highlights

- 🌫️ **Current AQI Monitoring** for Islamabad
- 📊 **72-Hour AQI Forecast** with hourly predictions
- ⚡ **100% Serverless Architecture**
- 🔄 **Hourly Feature Pipeline**
- 🤖 **Daily Model Retraining**
- 🧠 **SHAP Model Explainability**
- 📈 **Historical AQI Analysis**
- 🎨 **Interactive Streamlit Dashboard**
- ☁️ **Hopsworks Feature Store & Model Registry**
- 🔁 **GitHub Actions Automation**

---

## ✨ Features

### Data Pipeline

- Hourly weather and air-quality data collection from Open-Meteo
- Historical data backfill
- Automated feature engineering
- Hopsworks Feature Store integration
- Timestamp-based duplicate prevention

### Model Training

The following regression models are evaluated:

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The best-performing model is automatically selected and stored in the Hopsworks Model Registry.

### Inference & Forecasting

- 72-hour hourly AQI prediction
- Recursive AQI lag generation
- Three 24-hour forecast summaries
- Daily minimum, maximum, and average AQI
- AQI category classification
- Health advisory generation

### Interactive Dashboard

- Current Islamabad AQI
- PM2.5, PM10, NO₂ and O₃ concentrations
- 3-day AQI forecast cards
- 72-hour AQI trend
- Model performance metrics
- SHAP feature importance
- Historical AQI analysis
- Detailed hourly forecast table

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│        Historical Backfill + Hourly Data Ingestion                     │
│ GitHub Actions → Open-Meteo Weather API + Air Quality API              │
│             → Weather, Pollutants and Current AQI                      │
└───────────────────────────────┬────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────────┐
│                 Feature Engineering & Storage                          │
│ Temporal Features → Seasonal Features → Wind Features                  │
│ Weather Interactions → Hopsworks Feature Store                         │
└───────────────────────────────┬────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────────┐
│                    Model Training Pipeline                             │
│ Feature Retrieval → AQI Lag Features → Chronological Train/Test Split  │
│ Ridge → Random Forest → Gradient Boosting → XGBoost                    │
└───────────────────────────────┬────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────────┐
│                Model Evaluation & Explainability                       │
│           RMSE + MAE + R² → Best Model Selection → SHAP                │
└───────────────────────────────┬────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────────┐
│                    Hopsworks Model Registry                            │
│       Model Versioning → Metrics → Features → SHAP Artifacts           │
└───────────────────────────────┬────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────────┐
│                    72-Hour Inference Pipeline                          │
│ Future Weather & Pollutants → Feature Engineering → Recursive AQI Lags │
│                    → Best Model Prediction                             │
└───────────────────────────────┬────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────────┐
│                 Interactive Streamlit Dashboard                        │
│ Current AQI → 3-Day Forecast → Trend Charts → Model Metrics            │
│          → SHAP → EDA → Health Advisory → Hourly Data                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| **Data Source**      | Open-Meteo API / CAMS     |
| **Language**         | Python 3.11               |
| **Data Processing**  | Pandas, NumPy             |
| **Machine Learning** | Scikit-learn, XGBoost     |
| **Feature Store**    | Hopsworks                 |
| **Model Registry**   | Hopsworks                 |
| **Explainability**   | SHAP                      |
| **Visualization**    | Plotly                    |
| **Frontend**         | Streamlit                 |
| **Automation**       | GitHub Actions            |
| **Deployment**       | Streamlit Community Cloud |

---

## 📊 Feature Engineering

The system creates temporal, environmental, and historical AQI features.

### Temporal Features

* Hour
* Day of week
* Month
* Day of year
* Weekend indicator
* Rush-hour indicator
* Season

### Weather Features

* Wind U component
* Wind V component
* Stagnant-air indicator
* Temperature-humidity interaction

### AQI History Features

* AQI lag 1 hour
* AQI lag 3 hours
* AQI lag 6 hours
* AQI lag 12 hours
* AQI lag 24 hours
* 6-hour rolling AQI mean
* 24-hour rolling AQI mean

---

## 📈 Model Performance

A chronological train-test split is used because AQI forecasting is a time-dependent problem.

| Model                 |      RMSE |       MAE |        R² |
| --------------------- | --------: | --------: | --------: |
| Ridge Regression      |     6.482 |     4.627 |     0.931 |
| Random Forest         |     5.745 |     3.839 |     0.946 |
| **Gradient Boosting** | **5.330** | **3.597** | **0.953** |
| XGBoost               |     5.819 |     4.074 |     0.945 |

### Best Model

**Gradient Boosting Regressor**

* **R²:** 0.953
* **RMSE:** 5.33
* **MAE:** 3.60

---

## 🧠 Model Explainability

SHAP is used to analyze how strongly each feature influences the trained AQI model.

The dashboard displays the most important features using mean absolute SHAP values, helping explain the behavior of the model.

---

## 📊 Exploratory Data Analysis

The Streamlit dashboard includes historical AQI analysis such as:

* Historical AQI trend
* AQI distribution
* Average AQI by hour
* Average AQI by weekday
* Pollutant correlation with AQI

---

## 🔄 Automation

### Feature Pipeline

Runs automatically every hour using GitHub Actions.

```text
Open-Meteo APIs
       ↓
Latest Weather & Pollution Data
       ↓
Feature Engineering
       ↓
Hopsworks Feature Store


### Training Pipeline

Runs automatically every day.

```text
Hopsworks Feature Store
       ↓
Historical Feature Retrieval
       ↓
AQI Lag Generation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
SHAP Explainability
       ↓
Hopsworks Model Registry
```

---

## 🌫️ Current AQI vs Forecast AQI

The dashboard separates current air-quality data from machine-learning forecasts.

```text
Current Islamabad AQI
        ↓
Open-Meteo / CAMS


Next 72-Hour AQI Forecast
        ↓
Pearls AQI Predictor ML Model
```

This ensures that future AQI values shown on the dashboard are generated by the trained model rather than directly copied from the API.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11
* Git
* Hopsworks account
* GitHub account

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Salar-Haider/AQI-Predictor-10Pearls.git
cd AQI-Predictor-10Pearls
```

2. **Create a Conda environment**

```bash
conda create -n aqi-python311 python=3.11
conda activate aqi-python311
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Hopsworks credentials**

Create a `.env` file:

```env
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT=your_project_name
```

Do not commit `.env` or API keys to GitHub.

5. **Run the dashboard**

```bash
streamlit run dashboard/app.py
```

---

## 📂 Project Structure

```text
AQI-Predictor-10Pearls/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── config.py
│   ├── fetch_data.py
│   ├── features.py
│   ├── inference.py
│   ├── train.py
│   ├── model_registry.py
│   └── eda.py
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚠️ Limitations

* Currently supports Islamabad only
* Historical training data is limited
* Current AQI is provided by Open-Meteo/CAMS rather than a local physical sensor
* Recursive 72-hour forecasting can accumulate prediction error over time
* Forecast quality depends partly on future weather and pollutant forecasts

---

## 🔮 Future Enhancements

* [ ] Increase historical training data
* [ ] Support additional cities
* [ ] Experiment with LSTM and other deep-learning models
* [ ] Add prediction uncertainty intervals
* [ ] Add hazardous AQI email/SMS alerts
* [ ] Add automatic model drift monitoring

---




