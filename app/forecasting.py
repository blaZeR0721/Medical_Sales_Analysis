"""
Core time-series forecasting module for pharmacy medicine demand prediction.

Implements a weekday-seasonal naive model with exponential smoothing, selected
for datasets spanning fewer than 30 days where autoregressive models cannot
generalise meaningfully. Each future day is forecast as the exponentially
weighted mean of all historical observations sharing the same weekday, giving
more recent occurrences higher influence.

The module also exposes leave-one-out cross-validation metrics that reflect
genuine held-out performance rather than in-sample fit.

Example
-------
>>> ts = daily_quantity_series(df[df["DrugName"] == "Aspirin"])
>>> model = fit_seasonal_naive(ts)
>>> forecast_df = forecast(model, horizon=14)
>>> metrics = cross_validate(ts)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exponential smoothing decay factor applied to same-weekday observations.
# Higher values weight recent occurrences more aggressively (range: 0 < α ≤ 1).
_SMOOTHING_ALPHA: float = 0.7

# Minimum number of historical days required to produce a forecast.
MIN_HISTORY_DAYS: int = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonalNaiveModel:
    """Fitted weekday-seasonal naive model.

    Stores per-weekday exponentially weighted means derived from the training
    series. Instances are immutable; re-fit to incorporate new observations.

    Attributes
    ----------
    weekday_means : dict[int, float]
        Mapping of weekday index (0 = Monday, 6 = Sunday) to the smoothed
        demand estimate for that weekday.
    global_mean : float
        Mean of all observed daily quantities, used as a fallback when a
        weekday has no historical observations.
    last_date : pd.Timestamp
        Final date in the training series; forecasts begin from the
        following day.
    """

    weekday_means: dict[int, float]
    global_mean: float
    last_date: pd.Timestamp


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def daily_quantity_series(df_med: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-transaction records into a daily quantity series.

    Groups all transactions for a single medicine by calendar date and sums
    units sold, producing a ``DatetimeIndex``-aligned DataFrame with no
    intra-day granularity.

    Parameters
    ----------
    df_med : pd.DataFrame
        Raw sales records filtered to a single ``DrugName``. Must contain
        columns ``DateTime`` (datetime64) and ``Quantity`` (numeric).

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with a ``DatetimeIndex`` at daily frequency
        and column ``Quantity`` holding total units sold per day.

    Notes
    -----
    Dates with no recorded transactions are absent from the index. The
    seasonal naive model handles sparse weekdays via the global mean fallback,
    so explicit gap-filling is not required.
    """
    df_daily = (
        df_med.groupby(df_med["DateTime"].dt.date)["Quantity"]
        .sum()
        .to_frame()
        .rename_axis("Date")
    )

    df_daily = df_daily.set_index(pd.to_datetime(df_daily.index)).sort_index()
    return df_daily


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def _exponential_weights(n: int, alpha: float) -> np.ndarray:
    """Compute normalised exponential weights for ``n`` observations.

    Weights are ordered oldest-to-newest so the most recent observation
    receives the highest weight.

    Parameters
    ----------
    n : int
        Number of observations to weight.
    alpha : float
        Smoothing factor in the range ``(0, 1]``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n,)`` summing to 1.0.
    """
    exponents = np.arange(n - 1, -1, -1, dtype=float)
    raw = (1 - alpha) ** exponents
    return raw / raw.sum()


def fit_seasonal_naive(ts: pd.DataFrame) -> SeasonalNaiveModel:
    """Fit a weekday-seasonal naive model to a daily quantity series.

    For each weekday present in the history, computes an exponentially
    weighted mean that emphasises recent observations. Weekdays absent from
    the history fall back to the global mean at inference time.

    Parameters
    ----------
    ts : pd.DataFrame
        Output of :func:`daily_quantity_series`. Must contain a ``Quantity``
        column and a ``DatetimeIndex``.

    Returns
    -------
    SeasonalNaiveModel
        Fitted model encapsulating per-weekday smoothed estimates.

    Raises
    ------
    ValueError
        If ``ts`` contains fewer than ``MIN_HISTORY_DAYS`` observations or
        is missing the ``Quantity`` column.
    """
    if "Quantity" not in ts.columns:
        raise ValueError("Input DataFrame must contain a 'Quantity' column.")
    if len(ts) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"At least {MIN_HISTORY_DAYS} days of history are required; "
            f"got {len(ts)}."
        )

    ts_sorted = ts.sort_index()
    qty = ts_sorted["Quantity"]
    global_mean = float(qty.mean())

    weekday_means: dict[int, float] = {}
    weekday_indices = ts_sorted.index.dayofweek

    for wd in range(7):
        obs = qty.iloc[np.where(weekday_indices == wd)[0]].values
        if len(obs) == 0:
            continue
        weights = _exponential_weights(len(obs), _SMOOTHING_ALPHA)
        weekday_means[wd] = float(np.dot(weights, obs))

    return SeasonalNaiveModel(
        weekday_means=weekday_means,
        global_mean=global_mean,
        last_date=ts_sorted.index[-1],
    )


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def _predict_single(model: SeasonalNaiveModel, target_date: pd.Timestamp) -> float:
    """Return the demand estimate for a single future date.

    Parameters
    ----------
    model : SeasonalNaiveModel
        Fitted model from :func:`fit_seasonal_naive`.
    target_date : pd.Timestamp
        The calendar date to predict.

    Returns
    -------
    float
        Predicted quantity; always non-negative.
    """
    return max(0.0, model.weekday_means.get(target_date.dayofweek, model.global_mean))


def forecast(model: SeasonalNaiveModel, horizon: int) -> pd.DataFrame:
    """Generate a multi-day demand forecast from a fitted seasonal naive model.

    Parameters
    ----------
    model : SeasonalNaiveModel
        Fitted model from :func:`fit_seasonal_naive`.
    horizon : int
        Number of calendar days to forecast beyond the last training date.

    Returns
    -------
    pd.DataFrame
        Columns:

        * ``Date`` — ``datetime.date`` of the forecast day.
        * ``PredictedQty`` — smoothed demand estimate; non-negative float.

    Raises
    ------
    ValueError
        If ``horizon`` is less than 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    future_dates = pd.date_range(
        start=model.last_date + timedelta(days=1), periods=horizon
    )
    predictions = [_predict_single(model, d) for d in future_dates]

    return pd.DataFrame({"Date": future_dates.date, "PredictedQty": predictions})


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def cross_validate(ts: pd.DataFrame) -> dict[str, float]:
    """Evaluate forecast accuracy via leave-one-out cross-validation.

    For each observation beyond the minimum training window, fits the model
    on all preceding observations and predicts the held-out day. Reports
    Mean Absolute Error and Mean Absolute Percentage Error over all
    held-out predictions, producing genuinely out-of-sample metrics.

    Parameters
    ----------
    ts : pd.DataFrame
        Output of :func:`daily_quantity_series`.

    Returns
    -------
    dict[str, float]
        Keys:

        * ``mae`` — Mean Absolute Error in units sold.
        * ``mape`` — Mean Absolute Percentage Error (0–100 scale).
          Observations with zero actual quantity are excluded from MAPE.

        Both values are ``nan`` when fewer than ``MIN_HISTORY_DAYS + 1``
        observations are available.
    """
    nan_result: dict[str, float] = {"mae": np.nan, "mape": np.nan}

    if len(ts) < MIN_HISTORY_DAYS + 1:
        return nan_result

    ts_sorted = ts.sort_index()
    qty_values = ts_sorted["Quantity"].values
    errors = np.empty(len(ts_sorted) - MIN_HISTORY_DAYS)
    pct_errors: list[float] = []

    for i in range(MIN_HISTORY_DAYS, len(ts_sorted)):
        train = ts_sorted.iloc[:i]
        actual = float(qty_values[i])
        target = ts_sorted.index[i]

        try:
            model = fit_seasonal_naive(train)
            pred = _predict_single(model, target)
        except ValueError:
            errors[i - MIN_HISTORY_DAYS] = np.nan
            continue

        abs_err = abs(actual - pred)
        errors[i - MIN_HISTORY_DAYS] = abs_err
        if actual > 0:
            pct_errors.append(abs_err / actual * 100)

    valid_errors = errors[~np.isnan(errors)]
    if len(valid_errors) == 0:
        return nan_result

    return {
        "mae": float(np.mean(valid_errors)),
        "mape": float(np.mean(pct_errors)) if pct_errors else np.nan,
    }
