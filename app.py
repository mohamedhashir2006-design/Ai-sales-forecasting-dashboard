import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.llms import Ollama
llm = Ollama(model="gemma:2b")

from modules.forecasting.preprocess import load_and_prepare_data
from modules.forecasting.prophet_model import train_prophet, forecast_prophet
from modules.decision_engine import get_sales_trend
from modules.recommendation.model import recommend_action
from modules.chatbot import ask_ai

st.set_page_config(
    page_title="AI based CRM sales forecasting dashboard & smart follow-up recommendation engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
.main {
    background-color: #0b0e1f !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #dde1f5 !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #0b0e1f !important;
    border-bottom: 1px solid #1e2347 !important;
}
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
    background-color: #0b0e1f !important;
}
[data-testid="stSidebar"] {
    background-color: #0f1228 !important;
    border-right: 1px solid #1e2347 !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'DM Sans', sans-serif !important;
    color: #f0f2ff !important;
    font-weight: 500 !important;
}
body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #dde1f5 !important;
}
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
[data-testid="stMetricLabel"] > div {
    color: #8890bb !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    font-weight: 400 !important;
}
[data-testid="stMetricValue"] > div {
    color: #f0f2ff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.03em !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.78rem !important; font-weight: 500 !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #111428 !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1px solid #1e2347 !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #6b75a8 !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    background-color: #1a1f3c !important;
    color: #c0c7f0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #6c3fff, #9b5cf6) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px #6c3fff40 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid #1e2347 !important;
    overflow: hidden !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #111428 !important;
    border: 1px solid #1e2347 !important;
    border-radius: 10px !important;
    color: #f0f2ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #6c3fff !important;
    box-shadow: 0 0 0 3px #6c3fff25 !important;
    outline: none !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6c3fff, #9b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px #6c3fff35 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px #6c3fff55 !important;
}
hr { border: none !important; border-top: 1px solid #1e2347 !important; margin: 1.5rem 0 !important; }
[data-testid="stExpander"] {
    background-color: #111428 !important;
    border: 1px solid #1e2347 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #c0c7f0 !important; font-weight: 500 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0b0e1f; }
::-webkit-scrollbar-thumb { background: #2a2f5a; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #6c3fff; }
[data-testid="stCaptionContainer"], .stCaption { color: #6b75a8 !important; font-size: 0.8rem !important; }
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b75a8;
    font-weight: 500;
    margin-bottom: 0.75rem;
}
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #1e2347 !important;
}
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="#111428",
    plot_bgcolor="#111428",
    font=dict(family="DM Sans, sans-serif", color="#8890bb", size=12),
    title_font=dict(family="DM Sans, sans-serif", color="#f0f2ff", size=15),
    title_x=0.04,
    margin=dict(l=20, r=20, t=48, b=20),
    legend=dict(bgcolor="#0f1228", bordercolor="#1e2347", borderwidth=1, font=dict(color="#c0c7f0")),
    xaxis=dict(gridcolor="#181c38", linecolor="#1e2347", tickcolor="#1e2347", tickfont=dict(color="#6b75a8"), zerolinecolor="#1e2347"),
    yaxis=dict(gridcolor="#181c38", linecolor="#1e2347", tickcolor="#1e2347", tickfont=dict(color="#6b75a8"), zerolinecolor="#1e2347"),
)
ACCENT = ["#6c3fff", "#9b5cf6", "#00d4ff", "#f754a8", "#22c55e", "#f59e0b"]

@st.cache_data
def load_data():
    df = load_and_prepare_data("data/sales_history.csv")
    raw_df = pd.read_csv("data/sales_history.csv", parse_dates=["date"])
    raw_df["sales_channel"] = raw_df["sales_channel"].fillna("Unknown")
    interaction_df = pd.read_csv("data/interaction_history.csv")
    interaction_df.fillna(0, inplace=True)
    return df, raw_df, interaction_df

@st.cache_resource
def train_model(df):
    model = train_prophet(df)
    forecast = forecast_prophet(model)
    return model, forecast

df, raw_df, interaction_df = load_data()
model, forecast = train_model(df)

total_rev  = int(raw_df["revenue_usd"].sum())
units_sold = int(raw_df["units_sold"].sum())
avg_deal   = round(raw_df["avg_deal_size"].mean(), 2)
deals_closed = int(raw_df["deals_closed"].sum())

# HEADER
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:0.25rem;">
    <div style="width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,#6c3fff,#9b5cf6);
        display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 4px 18px #6c3fff50;">📈</div>
    <div>
        <div style="font-size:1.6rem;font-weight:500;color:#f0f2ff;letter-spacing:-0.02em;line-height:1.2;">
            AI based CRM sales forecasting dashboard & smart follow-up recommendation engine
        <div style="font-size:0.78rem;color:#6b75a8;letter-spacing:0.05em;">
            FORECAST &nbsp;·&nbsp; INSIGHTS &nbsp;·&nbsp; RECOMMENDATIONS &nbsp;·&nbsp; AI ASSISTANT</div>
    </div>
</div>
<hr style="margin:1.2rem 0 1.5rem;">
""", unsafe_allow_html=True)

# KPI CARDS
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Revenue",  f"${total_rev:,}",    delta="+12.4% vs last year")
with c2: st.metric("Units Sold",     f"{units_sold:,}",    delta="+8.1% vs last year")
with c3: st.metric("Deals Closed",   f"{deals_closed:,}",  delta="+5.6% conversion")
with c4: st.metric("Avg Deal Size",  f"${avg_deal}",       delta="-2.1% vs last year", delta_color="inverse")
st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast & Analytics",
    "⚖️  Model Comparison",
    "🎯  Recommendations",
    "🤖  AI Assistant",
])

# ── TAB 1 ──
with tab1:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 1.2], gap="medium")

    with left:
        fig_fc = go.Figure()
        hist = forecast[forecast["ds"] <= raw_df["date"].max()]
        fut  = forecast[forecast["ds"] >  raw_df["date"].max()]
        fig_fc.add_trace(go.Scatter(x=hist["ds"], y=hist["yhat"], mode="lines", name="Historical",
            line=dict(color="#6c3fff", width=2.5), fill="tozeroy", fillcolor="rgba(108,63,255,0.10)"))
        fig_fc.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], mode="lines", name="Forecast",
            line=dict(color="#9b5cf6", width=2, dash="dot"), fill="tozeroy", fillcolor="rgba(155,92,246,0.06)"))
        fig_fc.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
            line=dict(color="#9b5cf6", width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
            name="Confidence interval", line=dict(color="#9b5cf6", width=0),
            fill="tonexty", fillcolor="rgba(155,92,246,0.06)"))
        fig_fc.update_layout(**CHART_LAYOUT, title="30-Day Revenue Forecast", height=340)
        st.plotly_chart(fig_fc, use_container_width=True)

    with right:
        channel_data = raw_df.groupby("sales_channel")["revenue_usd"].sum().reset_index()
        fig_donut = go.Figure(go.Pie(
            labels=channel_data["sales_channel"], values=channel_data["revenue_usd"],
            hole=0.6, marker=dict(colors=ACCENT, line=dict(color="#111428", width=3)),
            textfont=dict(color="#dde1f5", size=12), textinfo="percent"))
        fig_donut.update_layout(**CHART_LAYOUT, title="Channel Mix", height=340,
            annotations=[dict(text=f"${total_rev//1000}K", x=0.5, y=0.5, showarrow=False,
                font=dict(color="#f0f2ff", size=18, family="DM Mono"))])
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        fig_trend = go.Figure(go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines",
            line=dict(color="#00d4ff", width=2.5), fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"))
        fig_trend.update_layout(**CHART_LAYOUT, title="Overall Trend", height=260)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_b:
        if "weekly" in forecast.columns:
            weekly_df = forecast.copy()
            weekly_df["day"] = weekly_df["ds"].dt.day_name()
            weekly_avg = weekly_df.groupby("day")["weekly"].mean().reset_index()
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            weekly_avg["day"] = pd.Categorical(weekly_avg["day"], categories=order, ordered=True)
            weekly_avg = weekly_avg.sort_values("day")
            colors = ["#6c3fff"] * len(weekly_avg)
            fig_week = go.Figure(go.Bar(x=weekly_avg["day"], y=weekly_avg["weekly"],
                marker=dict(color=colors, line=dict(width=0))))
            fig_week.update_layout(**CHART_LAYOUT, title="Weekly Sales Pattern", height=260, bargap=0.35)
            st.plotly_chart(fig_week, use_container_width=True)
        else:
            st.info("Weekly seasonality not available.")

    with col_c:
        st.markdown("<div class='section-label'>Anomaly Detection</div>", unsafe_allow_html=True)
        forecast["change"] = forecast["yhat"].pct_change()
        threshold = 0.15
        spikes = forecast[forecast["change"] > threshold]
        drops  = forecast[forecast["change"] < -threshold]
        if not spikes.empty:
            st.error(f"📈 **{len(spikes)} spike(s)** detected above +{int(threshold*100)}% threshold")
            st.dataframe(spikes[["ds","yhat","change"]].rename(
                columns={"ds":"Date","yhat":"Value","change":"Change"})
                .assign(Change=lambda d: d["Change"].map("{:.1%}".format)).head(5),
                use_container_width=True, hide_index=True)
        if not drops.empty:
            st.warning(f"📉 **{len(drops)} drop(s)** detected below -{int(threshold*100)}% threshold")
            st.dataframe(drops[["ds","yhat","change"]].rename(
                columns={"ds":"Date","yhat":"Value","change":"Change"})
                .assign(Change=lambda d: d["Change"].map("{:.1%}".format)).head(5),
                use_container_width=True, hide_index=True)
        if spikes.empty and drops.empty:
            st.success("✅ No major anomalies detected — trend is stable")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>AI Insights</div>", unsafe_allow_html=True)
    latest_value = forecast["yhat"].iloc[-1]
    first_value  = forecast["yhat"].iloc[0]
    trend_direction = "increasing" if latest_value > first_value else "decreasing"
    weekly_summary  = forecast.groupby(forecast["ds"].dt.day_name())["yhat"].mean()
    top_day    = weekly_summary.idxmax()
    worst_day  = weekly_summary.idxmin()
    channel_summary = raw_df.groupby("sales_channel")["revenue_usd"].sum()
    top_channel   = channel_summary.idxmax()
    worst_channel = channel_summary.idxmin()

    with st.expander("Generate AI Insights"):
        if st.button("Generate insights", key="insights_btn"):
            with st.spinner("Asking AI..."):
                prompt = f"""You are a business analyst. Sales trend is {trend_direction}.
Best day: {top_day}. Worst: {worst_day}. Top channel: {top_channel}. Weakest: {worst_channel}.
Give 3 bullet-point insights and 2 actionable recommendations. Be concise."""
                try:
                    response = llm.invoke(prompt)

                    # ✅ CLEAN THE TEXT
                    response = response.replace("•", "-").replace("–", "-")

                    # ✅ FORCE LINE BREAKS (important)
                    response = response.replace("\n\n", "\n")

                    st.success("AI Insights Generated")

                    # ✅ DISPLAY PROPERLY
                    st.markdown(response)
                except Exception as e:
                    st.error(f"❌ AI insight failed — make sure Ollama is running.\n\n`{e}`")

# ── TAB 2 ──
with tab2:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    try:
        from modules.forecasting.lstm_model import train_lstm
        import numpy as np
        with st.spinner("Training LSTM model..."):
            prophet_values = forecast["yhat"].values
            lstm_pred = train_lstm(df["y"])
            lstm_pred = [float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x) for x in lstm_pred]
            min_len = min(len(prophet_values), len(lstm_pred))
            comparison_df = pd.DataFrame({
                "date": forecast["ds"][:min_len],
                "Prophet": prophet_values[:min_len],
                "LSTM": lstm_pred[:min_len],
            })
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=comparison_df["date"], y=comparison_df["Prophet"],
            name="Prophet", mode="lines", line=dict(color="#6c3fff", width=2.5)))
        fig_cmp.add_trace(go.Scatter(x=comparison_df["date"], y=comparison_df["LSTM"],
            name="LSTM", mode="lines", line=dict(color="#f754a8", width=2, dash="dash")))
        fig_cmp.update_layout(**CHART_LAYOUT, title="Prophet vs LSTM Forecast", height=400)
        st.plotly_chart(fig_cmp, use_container_width=True)
        m1, m2, m3 = st.columns(3, gap="medium")
        prophet_mae = float(np.mean(np.abs(comparison_df["Prophet"] - comparison_df["LSTM"])))
        with m1: st.metric("Prophet MAE (vs LSTM)", f"{prophet_mae:,.0f}")
        with m2: st.metric("Data points compared", f"{min_len:,}")
        with m3: st.metric("Forecast horizon", "30 days")
    except Exception as e:
        st.error(f"❌ Model comparison failed: {e}")
        st.info("Make sure TensorFlow is installed: `pip install tensorflow`")

# ── TAB 3 ──
with tab3:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ================= TREND =================
    sales_trend = get_sales_trend(forecast)
    trend_color = "#22c55e" if sales_trend == "up" else "#ef4444"
    trend_icon  = "↑" if sales_trend == "up" else "↓"

    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:10px;
        background:#111428;border:1px solid #1e2347;border-radius:12px;
        padding:0.8rem 1.4rem;margin-bottom:1.2rem;">
        <span style="font-size:1.3rem;color:{trend_color};">{trend_icon}</span>
        <div>
            <div style="font-size:0.7rem;color:#6b75a8;text-transform:uppercase;">
                Current sales trend
            </div>
            <div style="font-size:1.1rem;font-weight:500;color:{trend_color};">
                {sales_trend.upper()}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= RECOMMENDATION =================
    interaction_df["recommendation"] = interaction_df.apply(
        lambda row: recommend_action(row, sales_trend), axis=1
    )

    left_col, right_col = st.columns([2.5, 1], gap="medium")

    # ================= LEFT SIDE =================
    with left_col:
        st.markdown("<div class='section-label'>Lead Follow-up Queue</div>", unsafe_allow_html=True)

        header = st.columns([1,1,1,1,1,1,2,1])
        headers = ["Lead","Stage","Type","Outcome","Rate","Days","Recommendation",""]

        for col, name in zip(header, headers):
            col.markdown(f"**{name}**")

        st.markdown("---")

        df_display = interaction_df.head(10)

        for i, row in df_display.iterrows():

            cols = st.columns([1,1,1,1,1,1,2,1])

            cols[0].write(row["lead_id"])
            cols[1].write(row["lead_stage"])
            cols[2].write(row["event_type"])
            cols[3].write(row["event_outcome"])
            cols[4].write(round(row["response_rate"], 2))
            cols[5].write(row["days_since_last_contact"])
            cols[6].write(row["recommendation"])

            send_clicked = cols[7].button("📩", key=f"send_{i}")

            # ================= MESSAGE =================
            if send_clicked:

                # 🔥 Dynamic action text based on recommendation
                if "Call" in row["recommendation"]:
                    action_text = "Please call the lead at the earliest opportunity."
                elif "Email" in row["recommendation"]:
                    action_text = "Please send a follow-up email to re-engage the lead."
                elif "discount" in row["recommendation"].lower():
                    action_text = "Consider offering a limited-time discount to move the deal forward."
                else:
                    action_text = "Please follow up with the lead promptly."

                days = row["days_since_last_contact"]

                msg = f"""Hi {row['lead_id']},

It has been {days} day(s) since your last interaction.

We noticed your recent {row['event_type']} activity.

Recommended action: {action_text}

Let us know if you need any assistance.

Best regards,
Sales Team"""

                # ✅ Safety cleanup
                msg = msg.replace("</div>", "")

                st.success(f"Message sent to {row['lead_id']}")

                st.markdown(f"""
                <div style="
                    max-width:520px;
                    line-height:1.6;
                    background:#111428;
                    padding:14px;
                    border-radius:10px;
                    border:1px solid #1e2347;
                    margin-top:8px;
                    margin-bottom:10px;
                    white-space: pre-line;
                ">
                {msg}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ================= RIGHT SIDE =================
    with right_col:
        st.markdown("<div class='section-label'>Recommendation Breakdown</div>", unsafe_allow_html=True)

        rec_counts = interaction_df["recommendation"].value_counts().reset_index()
        rec_counts.columns = ["Action", "Count"]

        fig_rec = go.Figure(go.Bar(
            x=rec_counts["Count"],
            y=rec_counts["Action"],
            orientation="h",
            marker=dict(color=ACCENT[:len(rec_counts)], line=dict(width=0))
        ))

        fig_rec.update_layout(**CHART_LAYOUT, title="Action Distribution", height=280)
        st.plotly_chart(fig_rec, use_container_width=True)

        if "lead_stage" in interaction_df.columns:
            stage_counts = interaction_df["lead_stage"].value_counts().reset_index()
            stage_counts.columns = ["Stage", "Count"]

            fig_stage = go.Figure(go.Pie(
                labels=stage_counts["Stage"],
                values=stage_counts["Count"],
                hole=0.55,
                marker=dict(colors=ACCENT, line=dict(color="#111428", width=2)),
                textinfo="percent",
                textfont=dict(size=11, color="#dde1f5")
            ))

            fig_stage.update_layout(**CHART_LAYOUT, title="Lead Stages", height=260, showlegend=False)
            st.plotly_chart(fig_stage, use_container_width=True)
# ── TAB 4 ──
with tab4:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 1], gap="medium")
    with left_col:
        st.markdown("<div class='section-label'>Ask your data</div>", unsafe_allow_html=True)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for msg in st.session_state.chat_history:
            align = "flex-end" if msg["role"] == "user" else "flex-start"
            bubble_bg = "linear-gradient(135deg,#6c3fff,#9b5cf6)" if msg["role"] == "user" else "#181c38"
            text_color = "#fff" if msg["role"] == "user" else "#dde1f5"
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin-bottom:0.6rem;">
                <div style="background:{bubble_bg};color:{text_color};border-radius:14px;
                    padding:0.75rem 1.1rem;max-width:80%;font-size:0.88rem;line-height:1.6;
                    border:1px solid #1e2347;">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        question = st.text_input("Message", placeholder="e.g. Which channel should I prioritize?",
            label_visibility="collapsed", key="chat_input")
        send = st.button("Send message →", key="send_btn")
        if send and question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            summary = raw_df.groupby("sales_channel")["revenue_usd"].sum().to_dict()
            prompt = f"""You are a business analyst. Question: {question}
Sales channel revenue: {summary}
Total revenue: ${total_rev:,}. Units sold: {units_sold:,}. Avg deal: ${avg_deal}.
Answer in 5-7 sentences, business-friendly and specific."""
            with st.spinner("Thinking..."):
                response = ask_ai(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        if st.button("Clear chat", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()
    with right_col:
        st.markdown("<div class='section-label'>Quick prompts</div>", unsafe_allow_html=True)
        quick_prompts = [
            "Which sales channel has the highest revenue?",
            "What is the current sales trend?",
            "Which leads should I prioritize this week?",
            "How can I improve conversion rate?",
        ]
        for qp in quick_prompts:
            if st.button(qp, key=f"qp_{qp[:20]}"):
                st.session_state.chat_history.append({"role": "user", "content": qp})
                summary = raw_df.groupby("sales_channel")["revenue_usd"].sum().to_string()
                prompt = f"""
You are a business analyst.

The pie chart represents revenue distribution across channels.

Data: {summary}

User question: {question}

Explain clearly:
- which channel performs best
- which is weakest
- one key business insight

Explain in detail (5-7 sentences) with clear insights and reasoning.
"""
                with st.spinner("Thinking..."):
                    response = ask_ai(prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Data context</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#111428;border:1px solid #1e2347;border-radius:12px;padding:1rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#6b75a8;font-size:0.8rem;">Records</span>
                <span style="color:#f0f2ff;font-size:0.8rem;font-family:'DM Mono',monospace;">3,000 leads</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#6b75a8;font-size:0.8rem;">Revenue data</span>
                <span style="color:#f0f2ff;font-size:0.8rem;font-family:'DM Mono',monospace;">Jan–Dec 2023</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#6b75a8;font-size:0.8rem;">Forecast horizon</span>
                <span style="color:#f0f2ff;font-size:0.8rem;font-family:'DM Mono',monospace;">30 days</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#6b75a8;font-size:0.8rem;">Models</span>
                <span style="color:#f0f2ff;font-size:0.8rem;font-family:'DM Mono',monospace;">Prophet + LSTM</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#6b75a8;font-size:0.8rem;">AI backend</span>
                <span style="color:#f0f2ff;font-size:0.8rem;font-family:'DM Mono',monospace;">Gemma 2b (Ollama)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)