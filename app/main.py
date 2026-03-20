"""
Streamlit dashboard for interactive pharmacy medicine demand forecasting.

Provides a two-tab interface:

* **Forecast** — Fits a weekday-seasonal naive model on historical daily sales
  for a selected medicine and renders a multi-day demand forecast alongside
  observed history. Leave-one-out cross-validation metrics are shown in the
  sidebar. Forecast results are exportable as CSV.

* **Historical Trends** — Displays raw daily unit sales and a weekday-level
  average bar chart for exploratory analysis.

Usage
-----
Run from the project root::

    streamlit run app.py

Configuration
-------------
``_DATA_PATH`` is hard-coded for local development. Replace with an
environment variable or Streamlit secrets for production deployments.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    SeasonalNaiveModel,
    cross_validate,
    daily_quantity_series,
    fit_seasonal_naive,
    forecast,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Medicine Demand Forecast", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_PATH = "D:\\7th sem\\project\\Medical_Sales_Analysis\\data\\pharmacy_sales.csv"
_DATE_FORMAT = "%d-%m-%Y %H:%M"
_HORIZON_OPTIONS: list[int] = [7, 14]
_WEEKDAY_ORDER: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
_HISTORY_COLOR = "#4C72B0"
_FORECAST_COLOR = "#DD8452"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess the raw pharmacy sales CSV.

    Parses ``DateTime`` using the project-specific format and derives a
    ``Date`` column used for daily aggregation. Results are cached by
    Streamlit so the file is read only once per session.

    Parameters
    ----------
    path : str
        Absolute or relative filesystem path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Transactions DataFrame containing at minimum ``DateTime`` (datetime64),
        ``Date`` (date), ``DrugName`` (str), and ``Quantity`` (numeric).
    """
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format=_DATE_FORMAT)
    df["Date"] = df["DateTime"].dt.date
    return df


@st.cache_data(show_spinner=False)
def get_medicine_series(path: str, drug_name: str) -> pd.DataFrame:
    """Load data and return the daily quantity series for a single medicine.

    Wraps :func:`load_data` and :func:`daily_quantity_series` with a combined
    cache key so switching medicines does not re-read the CSV.

    Parameters
    ----------
    path : str
        Filesystem path passed through to :func:`load_data`.
    drug_name : str
        The ``DrugName`` value to filter on.

    Returns
    -------
    pd.DataFrame
        Output of :func:`daily_quantity_series` for the requested medicine.
    """
    df = load_data(path)
    return daily_quantity_series(df[df["DrugName"] == drug_name])


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full Streamlit dashboard.

    Orchestrates data loading, sidebar controls, model fitting, and the
    two-tab layout (Forecast / Historical Trends).
    """
    df = load_data(_DATA_PATH)

    # ------------------------------------------------------------------
    # Sidebar — user controls
    # ------------------------------------------------------------------
    st.sidebar.header("Controls")

    medicine_list: list[str] = sorted(df["DrugName"].unique())
    selected_med: str = st.sidebar.selectbox(
        "Select Medicine", medicine_list, index=0
    )
    horizon_days: int = st.sidebar.select_slider(
        "Prediction window (days)", options=_HORIZON_OPTIONS, value=7
    )

    # ------------------------------------------------------------------
    # Data preparation and model fitting
    # ------------------------------------------------------------------
    ts_med = get_medicine_series(_DATA_PATH, selected_med)

    try:
        model = fit_seasonal_naive(ts_med)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    metrics = cross_validate(ts_med)

    # ------------------------------------------------------------------
    # Tab layout
    # ------------------------------------------------------------------
    tab_forecast, tab_hist = st.tabs(["Forecast", "Historical Trends"])

    with tab_forecast:
        _render_forecast_tab(ts_med, model, selected_med, horizon_days, metrics)

    with tab_hist:
        _render_history_tab(ts_med, selected_med)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_forecast_tab(
    ts_med: pd.DataFrame,
    model: SeasonalNaiveModel,
    selected_med: str,
    horizon_days: int,
    metrics: dict[str, float],
) -> None:
    """Render the Forecast tab.

    Displays a combined history + forecast line chart, a formatted forecast
    table, and a CSV download button. Leave-one-out cross-validation metrics
    are surfaced in the sidebar.

    Parameters
    ----------
    ts_med : pd.DataFrame
        Daily quantity series for the selected medicine.
    model : SeasonalNaiveModel
        Fitted forecasting model.
    selected_med : str
        Medicine name used as the section heading.
    horizon_days : int
        Number of days to forecast.
    metrics : dict[str, float]
        Cross-validation metrics with keys ``mae`` and ``mape``.
    """
    st.markdown(f"## Forecast of Units Sold — {selected_med}")

    _render_sidebar_metrics(metrics)

    forecast_df = forecast(model, horizon_days)

    x_start = ts_med.index.min() - pd.Timedelta(days=1)
    x_end = pd.to_datetime(forecast_df["Date"].iloc[-1]) + pd.Timedelta(days=1)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=ts_med.index,
                y=ts_med["Quantity"],
                mode="lines+markers",
                name="Historical Units",
                line=dict(color=_HISTORY_COLOR),
            ),
            go.Scatter(
                x=pd.to_datetime(forecast_df["Date"]),
                y=forecast_df["PredictedQty"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color=_FORECAST_COLOR, dash="dash"),
            ),
        ]
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        xaxis=dict(
            range=[x_start, x_end],
            tickformat="%b %d",
            tickangle=-45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Forecast Table (Units)")
    display_df = (
        forecast_df.copy()
        .assign(PredictedQty=lambda df: df["PredictedQty"].round().astype(int).apply(lambda x: f"{x:,}"))
        .set_index("Date")
    )
    st.dataframe(display_df, width='stretch')

    st.download_button(
        label="Download Forecast Data",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_data.csv",
        mime="text/csv",
    )


def _render_sidebar_metrics(metrics: dict[str, float]) -> None:
    """Render cross-validation metrics and caveat in the sidebar.

    Parameters
    ----------
    metrics : dict[str, float]
        Cross-validation metrics with keys ``mae`` and ``mape``.
    """
    mae_val = metrics.get("mae", float("nan"))
    mape_val = metrics.get("mape", float("nan"))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Performance (cross-validated)**")
    st.sidebar.metric("MAE", f"{mae_val:.2f}" if not pd.isna(mae_val) else "N/A")
    st.sidebar.metric("MAPE", f"{mape_val:.1f}%" if not pd.isna(mape_val) else "N/A")
    st.sidebar.caption(
        "Metrics vary per medicine due to limited history (13 days). "
        "Treat as indicative, not definitive."
    )


def _render_history_tab(ts_med: pd.DataFrame, selected_med: str) -> None:
    """Render the Historical Trends tab.

    Displays a daily units-sold line chart and a weekday-level average bar
    chart for the selected medicine.

    Parameters
    ----------
    ts_med : pd.DataFrame
        Daily quantity series for the selected medicine.
    selected_med : str
        Medicine name used as the section heading.
    """
    st.markdown(f"## Historical Quantity Trends — {selected_med}")

    ts_plot = ts_med.assign(Date=ts_med.index)

    st.subheader("Daily Units Sold")
    fig_hist = px.line(
        ts_plot,
        x="Date",
        y="Quantity",
        labels={"Quantity": "Units Sold", "Date": "Date"},
    )
    fig_hist.update_layout(
        xaxis=dict(tickformat="%b %d", tickangle=-45, tickfont=dict(size=10)),
        yaxis_title="Units Sold",
        hovermode="x unified",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_hist, width='stretch')

    st.subheader("Average Units Sold by Weekday")
    weekday_avg = (
        ts_plot.assign(Weekday=ts_plot["Date"].dt.day_name())
        .groupby("Weekday", sort=False)["Quantity"]
        .mean()
        .reindex(_WEEKDAY_ORDER)
    )

    fig_bar = px.bar(
        x=weekday_avg.index,
        y=weekday_avg.values,
        labels={"x": "Weekday", "y": "Avg Units Sold"},
        text_auto=".1f",
        color=weekday_avg.values,
        color_continuous_scale="Blues",
    )
    fig_bar.update_layout(
        xaxis_title="Weekday",
        yaxis_title="Average Units Sold",
        coloraxis_showscale=False,
        margin=dict(t=20),
    )
    st.plotly_chart(fig_bar, width='stretch')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

    