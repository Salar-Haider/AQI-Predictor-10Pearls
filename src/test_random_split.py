import math
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_PATH = "data/processed/islamabad_backfill_features.csv"

EXCLUDED_COLUMNS = [
    "timestamp",
    "us_aqi",
    "aqi_change",
]


def run_random_split_test():
    df = pd.read_csv(DATA_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns]
    y = df["us_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    rmse = math.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("Random split diagnostic")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")


if __name__ == "__main__":
    run_random_split_test()