import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from forecasting import (
    daily_quantity_series,
    make_features,
    train_model,
    iterative_forecast,
    get_metrics,
)

st.set_page_config(page_title="Medicine Demand Forecast", layout="wide")


########################################################################
# 1 Load & cache data
########################################################################
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d-%m-%Y %H:%M")
    df["Date"] = df["DateTime"].dt.date
    return df


df = load_data("D:\\7th sem\\project\\Medical_Sales_Analysis\\data\\pharmacy_sales.csv")

########################################################################
# 2 Sidebar controls
########################################################################
st.sidebar.header("🔧 Controls")
medicine_list = sorted(df["DrugName"].unique())
selected_med = st.sidebar.selectbox("Select Medicine", medicine_list, index=0)
horizon_days = st.sidebar.select_slider("Prediction window (days)", [7, 14], 7)

########################################################################
# 3 Prepare Data & Model
########################################################################
ts_med = daily_quantity_series(df[df["DrugName"] == selected_med])
if len(ts_med) < 10:
    st.warning("⚠️ Very little historical quantity data – forecasts may be unreliable.")

X_train, y_train = make_features(ts_med)
if X_train.empty:
    st.error("Not enough historical data to train model. Try another medicine.")
    st.stop()

model = train_model(X_train, y_train)
mae, r2 = get_metrics(model, X_train, y_train)

########################################################################
# 4 Tabs UI (Forecast + History)
########################################################################
tab_forecast, tab_hist = st.tabs(["📈 Forecast", "📊 Historical Trends"])

with tab_forecast:
    st.markdown(f"## 💡 Forecast of Units Sold – {selected_med}")
    forecast_df = iterative_forecast(model, ts_med, horizon_days)

    # Fixed range for x-axis
    x_start = pd.to_datetime("2023-03-31")
    x_end = pd.to_datetime("2023-04-28")

    # Plot history + forecast
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts_med.index,
            y=ts_med["Quantity"],
            mode="lines+markers",
            name="Historical Units",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(forecast_df["Date"]),
            y=forecast_df["PredictedQty"],
            mode="lines+markers",
            name="Forecast",
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        xaxis=dict(
            range=[x_start, x_end],
            tickmode="array",
            tickvals=pd.date_range(start=x_start, end=x_end, freq="D"),
            tickformat="%b %d",
            tickangle=-60,
            tickfont=dict(size=10),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Table
    st.markdown("### 📋 Forecast Table (Units)")
    display_df = forecast_df.copy()
    display_df["PredictedQty"] = (
        display_df["PredictedQty"].round().astype(int).apply(lambda x: f"{x:,}")
    )
    display_df.set_index("Date", inplace=True)
    st.dataframe(display_df, use_container_width=True)

    # Download button
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Forecast Data",
        data=csv,
        file_name="forecast_data.csv",
        mime="text/csv",
    )

with tab_hist:
    st.markdown(f"## 📊 Historical Quantity Trends – {selected_med}")
    st.subheader("📈 Daily Units Sold")

    import plotly.express as px

    # Fix index to have a proper "Date" column
    ts_med_fixed = ts_med.copy()
    ts_med_fixed["Date"] = ts_med_fixed.index

    fig_hist = px.line(
        ts_med_fixed,
        x="Date",
        y="Quantity",
        labels={"Quantity": "Units Sold", "Date": "Date"},
    )

    fig_hist.update_layout(
        xaxis=dict(
            tickformat="%b %d",  # Format like Apr 01
            tickangle=-45,
            tickfont=dict(size=10),
            tickmode="array",
            tickvals=pd.date_range(start="2023-04-01", end="2023-04-13", freq="D"),
        ),
        yaxis_title="Units Sold",
        hovermode="x unified",
        margin=dict(t=20),
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    # Weekday average bar chart
    st.subheader("📊 Average Units Sold by Weekday")

    weekday_avg = (
        ts_med_fixed.assign(Weekday=ts_med_fixed["Date"].dt.day_name())
        .groupby("Weekday")["Quantity"]
        .mean()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
    )

    fig_bar = px.bar(
        x=weekday_avg.index,
        y=weekday_avg.values,
        labels={"x": "Weekday", "y": "Avg Units Sold"},
        text_auto=".1f",
        color=weekday_avg.values,
        color_continuous_scale="Blues",
        title=None,
    )

    fig_bar.update_layout(
        xaxis_title="Weekday",
        yaxis_title="Average Units Sold",
        coloraxis_showscale=False,
        margin=dict(t=20),
    )

    st.plotly_chart(fig_bar, use_container_width=True)
