# forecasting_xgb.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta


def daily_quantity_series(df_med: pd.DataFrame) -> pd.DataFrame:
    """
    Groups sales data by day and sums the quantity sold per day.
    """
    series = df_med.groupby(df_med["DateTime"].dt.date)["Quantity"].sum().to_frame()
    series.index = pd.to_datetime(series.index)
    return series


def make_features(ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Creates time series features like lags, date parts, and rolling stats.
    """
    df_feat = ts.copy()
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["day"] = df_feat.index.day
    df_feat["is_weekend"] = df_feat.index.dayofweek.isin([5, 6]).astype(int)

    for lag in (1, 2, 3, 7):
        df_feat[f"lag_{lag}"] = df_feat["Quantity"].shift(lag)

    df_feat["rolling_mean_3"] = df_feat["Quantity"].shift(1).rolling(3).mean()
    df_feat.dropna(inplace=True)

    X = df_feat.drop("Quantity", axis=1)
    y = df_feat["Quantity"]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """
    Trains an XGBoost model on the prepared features.
    """
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model


def iterative_forecast(
    model: XGBRegressor, history: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """
    Performs recursive multi-day forecasting using the trained model.
    """
    hist = history.copy()
    last_date = hist.index[-1]
    preds = []

    for i in range(1, horizon + 1):
        next_date = last_date + timedelta(days=i)
        row = {
            "dayofweek": next_date.dayofweek,
            "month": next_date.month,
            "day": next_date.day,
            "is_weekend": int(next_date.dayofweek in [5, 6]),
            "lag_1": hist["Quantity"].iloc[-1],
            "lag_2": hist["Quantity"].iloc[-2] if len(hist) > 1 else np.nan,
            "lag_3": hist["Quantity"].iloc[-3] if len(hist) > 2 else np.nan,
            "lag_7": hist["Quantity"].iloc[-7] if len(hist) > 6 else np.nan,
            "rolling_mean_3": (
                hist["Quantity"].iloc[-3:].mean()
                if len(hist) >= 3
                else hist["Quantity"].mean()
            ),
        }

        # Handle missing values
        for k, v in row.items():
            if pd.isna(v):
                row[k] = hist["Quantity"].mean()

        pred_val = model.predict(pd.DataFrame([row]))[0]
        preds.append(pred_val)
        hist.loc[next_date] = pred_val  # Append prediction to history

    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon)
    return pd.DataFrame({"Date": future_dates.date, "PredictedQty": preds})


def get_metrics(model, X_train, y_train):
    """
    Returns MAE and R² for model performance on training data.
    """
    if len(y_train) < 2:
        return np.nan, np.nan

    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    return mae, r2
