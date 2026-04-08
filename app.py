
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.llms import Ollama
llm = Ollama(model="gemma:2b")
import os

from modules.forecasting.preprocess import load_and_prepare_data
from modules.forecasting.prophet_model import train_prophet, forecast_prophet
from modules.decision_engine import get_sales_trend
from modules.recommendation.model import recommend_action
from modules.chatbot import ask_ai

# -------------------------------
# PAGE CONFIG (Professional UI)
# -------------------------------
st.set_page_config(
    page_title="Sales AI Dashboard",
    layout="wide"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 AI Sales Forecasting & Recommendation System")
st.caption("Forecast • Insights • Recommendations • AI Assistant")


# -------------------------------
# LOAD DATA
# -------------------------------
df = load_and_prepare_data("data/sales_history.csv")
raw_df = pd.read_csv("data/sales_history.csv")

# -------------------------------
# KPI METRICS (Top Section)
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Total Revenue", f"${int(raw_df['revenue_usd'].sum()):,}")

with col2:
    st.metric("📦 Units Sold", int(raw_df["units_sold"].sum()))

with col3:
    st.metric("📊 Avg Deal Size", round(raw_df["avg_deal_size"].mean(), 2))

# -------------------------------
# TRAIN FORECAST MODEL (ONCE)
# -------------------------------
model = train_prophet(df)
forecast = forecast_prophet(model)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Forecast",
    "📉 Model Comparison",
    "📞 Recommendations",
    "🤖 AI Assistant"
])

with tab1:

    # ==============================
    # 📈 SALES FORECAST
    # ==============================
    st.subheader("📈 Sales Forecast")

    fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Sales Forecast"
    )

    fig.update_layout(template="plotly_dark", title_x=0.3)
    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # 🚨 ANOMALY DETECTION
    # ==============================
    st.markdown("---")
    st.subheader("🚨 Anomaly Detection")

    forecast["change"] = forecast["yhat"].pct_change()
    threshold = 0.15

    spikes = forecast[forecast["change"] > threshold]
    drops = forecast[forecast["change"] < -threshold]

    if not spikes.empty:
        st.error("📈 Sales Spikes Detected")
        st.dataframe(spikes[["ds", "yhat", "change"]])

    if not drops.empty:
        st.warning("📉 Sales Drops Detected")
        st.dataframe(drops[["ds", "yhat", "change"]])

    if spikes.empty and drops.empty:
        st.success("✅ No major anomalies detected")

    # ==============================
    # 📊 TREND
    # ==============================
    st.markdown("---")
    st.subheader("📊 Trend")

    fig_trend = px.line(
        forecast,
        x="ds",
        y="trend",
        title="Overall Trend"
    )

    fig_trend.update_layout(template="plotly_dark", title_x=0.3)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ==============================
    # 📅 WEEKLY SEASONALITY
    # ==============================
    st.markdown("---")
    st.subheader("📅 Weekly Sales Impact")

    if "weekly" in forecast.columns:

        weekly_df = forecast.copy()
        weekly_df["day"] = weekly_df["ds"].dt.day_name()

        weekly_avg = weekly_df.groupby("day")["weekly"].mean().reset_index()

        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekly_avg["day"] = pd.Categorical(weekly_avg["day"], categories=order, ordered=True)
        weekly_avg = weekly_avg.sort_values("day")

        fig_weekly = px.bar(
            weekly_avg,
            x="day",
            y="weekly",
            title="Weekly Sales Impact"
        )

        fig_weekly.update_layout(template="plotly_dark", title_x=0.3)
        st.plotly_chart(fig_weekly, use_container_width=True)

    # ==============================
    # 🥧 SALES CHANNEL DISTRIBUTION (FIXED)
    # ==============================
    st.markdown("---")
    st.subheader("📊 Sales Channel Distribution")

    # 🔥 FIX UNDEFINED ISSUE
    raw_df["sales_channel"] = raw_df["sales_channel"].fillna("Unknown")

    fig_pie = px.pie(
        raw_df,
        names="sales_channel",
        values="revenue_usd",
        hole=0.4,
        title="Sales Channel Distribution"
    )

    fig_pie.update_layout(template="plotly_dark", title_x=0.3)
    fig_pie.update_traces(textinfo="percent+label")

    st.plotly_chart(fig_pie, use_container_width=True)

    # ==============================
    # 🤖 AI INSIGHTS (OLLAMA FIXED)
    # ==============================
    st.markdown("---")
    st.subheader("🤖 AI Insights")

    latest_value = forecast["yhat"].iloc[-1]
    first_value = forecast["yhat"].iloc[0]

    trend_direction = "increasing 📈" if latest_value > first_value else "decreasing 📉"

    weekly_summary = forecast.groupby(forecast["ds"].dt.day_name())["yhat"].mean()
    top_day = weekly_summary.idxmax()
    worst_day = weekly_summary.idxmin()

    channel_summary = raw_df.groupby("sales_channel")["revenue_usd"].sum()
    top_channel = channel_summary.idxmax()
    worst_channel = channel_summary.idxmin()

    prompt = f"""
    You are a business analyst.

    Sales trend is {trend_direction}.
    Best sales day: {top_day}
    Worst sales day: {worst_day}
    Top channel: {top_channel}
    Weakest channel: {worst_channel}

    Give 3 insights and 2 recommendations.
    """

    try:
        response = llm.invoke(prompt)

        st.success("✅ AI Insights Generated")
        st.write(response)

    except Exception as e:
        st.error("❌ AI insight failed. Make sure Ollama is running.")
        st.exception(e)
with tab2:
    st.subheader("📉 Model Comparison")

    try:
        from modules.forecasting.lstm_model import train_lstm
        import numpy as np

        st.write("Running Prophet vs LSTM...")

        # Prophet values
        prophet_df = forecast.copy()
        prophet_values = prophet_df["yhat"].values

        # 🔥 FIXED: correct column
        lstm_pred = train_lstm(df["y"])

        # 🔥 CLEAN output
        lstm_pred = [
            float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x)
            for x in lstm_pred
        ]

        # Match length
        min_len = min(len(prophet_values), len(lstm_pred))

        # Create dataframe
        comparison_df = pd.DataFrame({
            "date": prophet_df["ds"][:min_len],
            "Prophet": prophet_values[:min_len],
            "LSTM": lstm_pred[:min_len]
        })

        # Plot
        fig = px.line(
            comparison_df,
            x="date",
            y=["Prophet", "LSTM"],
            title="Prophet vs LSTM Forecast"
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Error in Model Comparison: {e}")
  
# ===============================
# TAB 3: RECOMMENDATION ENGINE
# ===============================
with tab3:
    st.subheader("📞 Smart Follow-up Recommendations")

    interaction_df = pd.read_csv("data/interaction_history.csv")

    # Safety (avoid errors)
    interaction_df.fillna(0, inplace=True)

    # Get trend from forecast
    sales_trend = get_sales_trend(forecast)

    st.write(f"📉 Current Sales Trend: **{sales_trend.upper()}**")

    # Apply recommendation logic
    interaction_df["recommendation"] = interaction_df.apply(
        lambda row: recommend_action(row, sales_trend),
        axis=1
    )

    st.dataframe(
        interaction_df.head(20),
        use_container_width=True
    )

# ===============================
# TAB 4: AI CHATBOT
# ===============================
with tab4:
    st.subheader("💬 Ask Your Data")

    question = st.text_input("Ask anything")

    if question:
        # 🔥 CREATE SUMMARY FOR PIE CHART
        summary = raw_df.groupby("sales_channel")["revenue_usd"].sum().to_string()

        # 🔥 CREATE SMART PROMPT
        prompt = f"""
        You are a business analyst.

        User question: {question}

        Here is sales channel revenue data (used in pie chart):
        {summary}

        Explain insights from this distribution:
        - Which channel performs best?
        - Which is weakest?
        - What should the business do?

        Answer in simple business language.
        """

        # 🔥 SEND TO AI
        response = ask_ai(prompt)

        # 🔥 SHOW RESULT
        st.success(response)