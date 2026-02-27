"""
Forex Session Range Analyser
=============================
Run with:  streamlit run session_analysis.py

Requirements:
    pip install streamlit yfinance pandas plotly pytz
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, time
import pytz

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Session Range Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stMetric { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 12px; }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .big-num { font-family: 'DM Mono', monospace; font-size: 2.4rem; font-weight: 700; line-height: 1; }
    .label-sm { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; }
    .sub-text { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
    div[data-testid="stSidebar"] { background: #0d1525; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "Asia":    "#38bdf8",
    "London":  "#a78bfa",
    "NY":      "#fb923c",
    "Neutral": "#64748b",
    "Pink":    "#f472b6",
}

# Session hours in UTC
SESSION_UTC = {
    "Asia":   (0,  9),
    "London": (8,  17),
    "NY":     (13, 22),
}

PAIRS = {
    # Forex
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CHF": "USDCHF=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    # Indices / Futures (also useful for session analysis)
    "US500 (S&P)":    "^GSPC",
    "NAS100 (Nasdaq)": "^IXIC",
    "DAX":            "^GDAXI",
    "FTSE 100":       "^FTSE",
    # Commodities
    "Gold (XAU/USD)": "GC=F",
    "Silver (XAG/USD)": "SI=F",
    "WTI Oil":        "CL=F",
    "Bitcoin/USD":    "BTC-USD",
}

LOOKBACK_OPTIONS = {
    "1 Month":   30,
    "3 Months":  90,
    "6 Months": 180,
    "1 Year":   365,
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, days: int) -> pd.DataFrame:
    """Fetch hourly OHLCV data from yfinance."""
    period = "60d" if days <= 60 else "730d"
    df = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index, utc=True)
    # Keep only last `days` worth
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df.index >= cutoff]
    return df


def session_ohlc(hourly: pd.DataFrame, session: str) -> pd.DataFrame:
    """Extract daily OHLC for a given session window (UTC hours)."""
    start_h, end_h = SESSION_UTC[session]
    mask = (hourly.index.hour >= start_h) & (hourly.index.hour < end_h)
    sess = hourly[mask].copy()
    if sess.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns from yfinance (ticker in column level)
    if isinstance(sess.columns, pd.MultiIndex):
        sess.columns = sess.columns.get_level_values(0)

    sess["date"] = sess.index.date
    daily = sess.groupby("date").agg(
        open=("Open", "first"),
        high=("High", "max"),
        low=("Low", "min"),
        close=("Close", "last"),
        volume=("Volume", "sum"),
    )
    daily["range"] = daily["high"] - daily["low"]
    daily["direction"] = np.where(daily["close"] >= daily["open"], "Bullish", "Bearish")
    return daily


def build_session_df(hourly: pd.DataFrame) -> pd.DataFrame:
    """Combine all three sessions into one row-per-day DataFrame."""
    asia   = session_ohlc(hourly, "Asia").add_prefix("asia_")
    london = session_ohlc(hourly, "London").add_prefix("london_")
    ny     = session_ohlc(hourly, "NY").add_prefix("ny_")

    df = asia.join(london, how="inner").join(ny, how="inner")
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute all analytics from the session DataFrame."""
    n = len(df)
    if n == 0:
        return {}

    # --- Range stats ---
    asia_range_median = df["asia_range"].median()
    london_range_median = df["london_range"].median()
    ny_range_median = df["ny_range"].median()

    # --- London breaks Asia ---
    london_breaks_asia = (
        (df["london_high"] > df["asia_high"]) | (df["london_low"] < df["asia_low"])
    )
    # London breaks small Asia (asia range below median)
    small_asia_mask = df["asia_range"] < asia_range_median
    # --- NY breaks London ---
    ny_breaks_london = (
        (df["ny_high"] > df["london_high"]) | (df["ny_low"] < df["london_low"])
    )
    # NY breaks small London
    small_london_mask = df["london_range"] < london_range_median

    # --- NY reverses London ---
    london_bull = df["london_direction"] == "Bullish"
    ny_bull     = df["ny_direction"]     == "Bullish"
    ny_reverses_london = london_bull != ny_bull

    # --- Direction combos ---
    dir_combos = {
        "London ↑ → NY ↑": ((london_bull) & (ny_bull)).sum(),
        "London ↑ → NY ↓": ((london_bull) & (~ny_bull)).sum(),
        "London ↓ → NY ↑": ((~london_bull) & (ny_bull)).sum(),
        "London ↓ → NY ↓": ((~london_bull) & (~ny_bull)).sum(),
    }

    return {
        "n": n,
        # Ranges
        "asia_range_avg":   df["asia_range"].mean(),
        "london_range_avg": df["london_range"].mean(),
        "ny_range_avg":     df["ny_range"].mean(),
        "asia_range_median":   asia_range_median,
        "london_range_median": london_range_median,
        "ny_range_median": ny_range_median,
        "asia_range_max":   df["asia_range"].max(),
        "london_range_max": df["london_range"].max(),
        "ny_range_max":     df["ny_range"].max(),
        "asia_range_min":   df["asia_range"].min(),
        "london_range_min": df["london_range"].min(),
        "ny_range_min":     df["ny_range"].min(),
        "asia_range_std":   df["asia_range"].std(),
        "london_range_std": df["london_range"].std(),
        "ny_range_std":     df["ny_range"].std(),
        # Break probabilities
        "london_breaks_asia_pct":    london_breaks_asia.mean() * 100,
        "ny_breaks_london_pct":      ny_breaks_london.mean() * 100,
        "london_breaks_small_asia_pct": (
            london_breaks_asia[small_asia_mask].mean() * 100 if small_asia_mask.sum() > 0 else 0
        ),
        "ny_breaks_small_london_pct": (
            ny_breaks_london[small_london_mask].mean() * 100 if small_london_mask.sum() > 0 else 0
        ),
        # Reversals
        "ny_reverses_london_pct": ny_reverses_london.mean() * 100,
        "dir_combos": dir_combos,
        # Series for charts
        "df": df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    pair_name = st.selectbox(
        "Currency / Instrument",
        list(PAIRS.keys()),
        index=0,
    )
    ticker = PAIRS[pair_name]

    lookback_label = st.selectbox(
        "Lookback Period",
        list(LOOKBACK_OPTIONS.keys()),
        index=1,
    )
    lookback_days = LOOKBACK_OPTIONS[lookback_label]

    st.markdown("---")
    st.markdown("**Session Times (UTC)**")
    st.markdown("""
    | Session | Window |
    |---------|--------|
    | 🌏 Asia | 00:00 – 09:00 |
    | 🇬🇧 London | 08:00 – 17:00 |
    | 🇺🇸 New York | 13:00 – 22:00 |
    """)
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.75rem;color:#475569'>Data via Yahoo Finance · "
        "1-hour OHLCV · UTC</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"# 📊 Session Range Analyser")
st.markdown(
    f"**{pair_name}** · `{ticker}` · {lookback_label} · "
    f"<span style='color:#64748b;font-size:0.85rem'>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

with st.spinner(f"Fetching {pair_name} hourly data..."):
    hourly = fetch_data(ticker, lookback_days)

if hourly.empty:
    st.error(
        f"⚠️ Could not fetch data for **{pair_name}** (`{ticker}`). "
        "Check your internet connection or try a different instrument."
    )
    st.stop()

df_sessions = build_session_df(hourly)

if len(df_sessions) < 5:
    st.warning("Not enough data to compute statistics. Try a longer lookback or different pair.")
    st.stop()

stats = compute_stats(df_sessions)
df = stats["df"]
n = stats["n"]

# pip value scaling label
pip_label = "Price Units"
if "JPY" in pair_name.upper():
    pip_label = "Pips (0.01)"
elif any(x in pair_name.upper() for x in ["USD", "EUR", "GBP", "AUD", "NZD", "CHF", "CAD"]):
    pip_label = "Pips (0.0001)"

def fmt_range(val: float) -> str:
    """Format range value nicely."""
    if val >= 1000:
        return f"{val:,.0f}"
    elif val >= 10:
        return f"{val:.2f}"
    else:
        return f"{val:.5f}"

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🎯 Probabilities", "📏 Ranges", "🔄 Reversals"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"### Key Metrics — {n} trading days analysed")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🌏 Avg Asia Range", fmt_range(stats["asia_range_avg"]), help="Average high-low range during Asia session")
    with c2:
        st.metric("🇬🇧 Avg London Range", fmt_range(stats["london_range_avg"]))
    with c3:
        st.metric("🇺🇸 Avg NY Range", fmt_range(stats["ny_range_avg"]))

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "London Breaks Asia",
            f"{stats['london_breaks_asia_pct']:.1f}%",
            help="% of days London traded outside Asia's high or low",
        )
    with c2:
        st.metric(
            "NY Breaks London",
            f"{stats['ny_breaks_london_pct']:.1f}%",
            help="% of days NY traded outside London's high or low",
        )
    with c3:
        st.metric(
            "NY Reverses London",
            f"{stats['ny_reverses_london_pct']:.1f}%",
            help="% of days NY closed in opposite direction to London",
        )

    st.markdown("---")
    st.markdown("#### Average Range by Session")

    fig_bar = go.Figure()
    sessions_list = ["Asia", "London", "NY"]
    avg_ranges    = [stats["asia_range_avg"], stats["london_range_avg"], stats["ny_range_avg"]]
    max_ranges    = [stats["asia_range_max"], stats["london_range_max"], stats["ny_range_max"]]
    min_ranges    = [stats["asia_range_min"], stats["london_range_min"], stats["ny_range_min"]]
    colors_list   = [COLORS["Asia"], COLORS["London"], COLORS["NY"]]

    fig_bar.add_trace(go.Bar(
        x=sessions_list,
        y=avg_ranges,
        name="Average",
        marker_color=colors_list,
        text=[fmt_range(v) for v in avg_ranges],
        textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        x=sessions_list,
        y=max_ranges,
        name="Max",
        marker_color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.25)" for c in colors_list],
        text=[fmt_range(v) for v in max_ranges],
        textposition="outside",
    ))
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Range"),
        height=340,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Range over time (rolling 10-day avg)
    st.markdown("#### Session Range — Rolling 10-Day Average")
    fig_roll = go.Figure()
    for col, name, color in [
        ("asia_range", "Asia", COLORS["Asia"]),
        ("london_range", "London", COLORS["London"]),
        ("ny_range", "NY", COLORS["NY"]),
    ]:
        roll = df[col].rolling(10).mean()
        fig_roll.add_trace(go.Scatter(
            x=df.index, y=roll,
            name=name,
            line=dict(color=color, width=2),
            mode="lines",
        ))
    fig_roll.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Range"),
        legend=dict(orientation="h", y=1.08),
        height=320,
    )
    st.plotly_chart(fig_roll, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Session Break Probabilities")
    st.caption(
        "How frequently does each session extend beyond the prior session's range? "
        "Conditional probabilities show whether a compressed prior session increases the likelihood of a breakout."
    )

    # ── London vs Asia ─────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🇬🇧 London vs Asia")
        lb_pct   = stats["london_breaks_asia_pct"]
        lb_sm_pct = stats["london_breaks_small_asia_pct"]
        no_break = 100 - lb_pct

        fig = go.Figure(go.Bar(
            x=["Breaks Asia", "Stays Within", "Breaks (Small Asia)"],
            y=[lb_pct, no_break, lb_sm_pct],
            marker_color=[COLORS["London"], COLORS["Neutral"], COLORS["Pink"]],
            text=[f"{v:.1f}%" for v in [lb_pct, no_break, lb_sm_pct]],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False,
            yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.07)", title="%"),
            xaxis=dict(showgrid=False), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"London breaks Asia's range **{lb_pct:.1f}%** of the time overall.  \n"
            f"When Asia's range is **below median** ({fmt_range(stats['asia_range_median'])}), "
            f"London breaks it **{lb_sm_pct:.1f}%** of the time — "
            f"{'↑ higher' if lb_sm_pct > lb_pct else '↓ lower'} than base rate."
        )

    with col_b:
        st.markdown("#### 🇺🇸 New York vs London")
        nb_pct   = stats["ny_breaks_london_pct"]
        nb_sm_pct = stats["ny_breaks_small_london_pct"]
        no_break_ny = 100 - nb_pct

        fig2 = go.Figure(go.Bar(
            x=["Breaks London", "Stays Within", "Breaks (Small London)"],
            y=[nb_pct, no_break_ny, nb_sm_pct],
            marker_color=[COLORS["NY"], COLORS["Neutral"], COLORS["Pink"]],
            text=[f"{v:.1f}%" for v in [nb_pct, no_break_ny, nb_sm_pct]],
            textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False,
            yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.07)", title="%"),
            xaxis=dict(showgrid=False), height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.info(
            f"NY breaks London's range **{nb_pct:.1f}%** of the time overall.  \n"
            f"When London's range is **below median** ({fmt_range(stats['london_range_median'])}), "
            f"NY breaks it **{nb_sm_pct:.1f}%** of the time — "
            f"{'↑ higher' if nb_sm_pct > nb_pct else '↓ lower'} than base rate."
        )

    # ── Direction matrix ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Direction Outcome Matrix — London vs NY")
    combos = stats["dir_combos"]
    combo_df = pd.DataFrame({
        "Scenario":    list(combos.keys()),
        "Count":       list(combos.values()),
        "Probability": [v / n * 100 for v in combos.values()],
    })

    fig_combo = px.bar(
        combo_df, x="Scenario", y="Probability",
        text=combo_df["Probability"].apply(lambda x: f"{x:.1f}%"),
        color="Scenario",
        color_discrete_map={
            "London ↑ → NY ↑": COLORS["London"],
            "London ↑ → NY ↓": COLORS["Pink"],
            "London ↓ → NY ↑": COLORS["Pink"],
            "London ↓ → NY ↓": COLORS["NY"],
        },
    )
    fig_combo.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8", showlegend=False,
        yaxis=dict(range=[0, 60], gridcolor="rgba(255,255,255,0.07)", title="%"),
        xaxis=dict(showgrid=False), height=320,
    )
    fig_combo.update_traces(textposition="outside")
    st.plotly_chart(fig_combo, use_container_width=True)

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Summary Probability Table")
    summary_data = {
        "Scenario": [
            "London breaks Asia range",
            "London stays within Asia range",
            "London breaks SMALL Asia range",
            "NY breaks London range",
            "NY stays within London range",
            "NY breaks SMALL London range",
            "NY reverses London direction",
            "NY continues London direction",
        ],
        "Probability": [
            f"{stats['london_breaks_asia_pct']:.1f}%",
            f"{100 - stats['london_breaks_asia_pct']:.1f}%",
            f"{stats['london_breaks_small_asia_pct']:.1f}%",
            f"{stats['ny_breaks_london_pct']:.1f}%",
            f"{100 - stats['ny_breaks_london_pct']:.1f}%",
            f"{stats['ny_breaks_small_london_pct']:.1f}%",
            f"{stats['ny_reverses_london_pct']:.1f}%",
            f"{100 - stats['ny_reverses_london_pct']:.1f}%",
        ],
        "Sample Size": [str(n)] * 8,
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RANGES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Range Deep Dive")

    # Stats table
    range_table = pd.DataFrame({
        "Metric":   ["Average", "Median", "Maximum", "Minimum", "Std Dev"],
        "🌏 Asia":   [fmt_range(stats["asia_range_avg"]),   fmt_range(stats["asia_range_median"]),
                      fmt_range(stats["asia_range_max"]),   fmt_range(stats["asia_range_min"]),
                      fmt_range(stats["asia_range_std"])],
        "🇬🇧 London": [fmt_range(stats["london_range_avg"]), fmt_range(stats["london_range_median"]),
                       fmt_range(stats["london_range_max"]), fmt_range(stats["london_range_min"]),
                       fmt_range(stats["london_range_std"])],
        "🇺🇸 NY":    [fmt_range(stats["ny_range_avg"]),     fmt_range(stats["ny_range_median"]),
                      fmt_range(stats["ny_range_max"]),     fmt_range(stats["ny_range_min"]),
                      fmt_range(stats["ny_range_std"])],
    })
    st.dataframe(range_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Range distribution histograms
    st.markdown("#### Range Distribution by Session")
    fig_hist = go.Figure()
    for col, name, color in [
        ("asia_range", "Asia", COLORS["Asia"]),
        ("london_range", "London", COLORS["London"]),
        ("ny_range", "NY", COLORS["NY"]),
    ]:
        fig_hist.add_trace(go.Histogram(
            x=df[col], name=name, marker_color=color, opacity=0.6,
            nbinsx=30, histnorm="probability",
        ))
    fig_hist.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        xaxis=dict(showgrid=False, title="Range"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Probability"),
        legend=dict(orientation="h", y=1.08),
        height=340,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Daily range chart (last 60 days)
    st.markdown("#### Daily Session Ranges")
    show_n = st.slider("Days to show", 10, min(n, 120), min(n, 60))
    df_show = df.tail(show_n)

    fig_daily = go.Figure()
    for col, name, color in [
        ("asia_range", "Asia", COLORS["Asia"]),
        ("london_range", "London", COLORS["London"]),
        ("ny_range", "NY", COLORS["NY"]),
    ]:
        fig_daily.add_trace(go.Scatter(
            x=df_show.index, y=df_show[col],
            name=name, line=dict(color=color, width=1.5), mode="lines",
        ))
    fig_daily.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Range"),
        legend=dict(orientation="h", y=1.08),
        height=320,
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # Correlation between session ranges
    st.markdown("---")
    st.markdown("#### Session Range Correlation")
    corr_df = df[["asia_range", "london_range", "ny_range"]].rename(columns={
        "asia_range": "Asia", "london_range": "London", "ny_range": "NY"
    }).corr()

    fig_corr = px.imshow(
        corr_df, text_auto=".2f", color_continuous_scale="Blues",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8", height=280,
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Values close to 1.0 indicate session ranges tend to expand/contract together on the same day.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REVERSALS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### NY Reversal Analysis")
    st.caption("A 'reversal' is when NY session closes in the opposite direction to London's session close.")

    rev_pct = stats["ny_reverses_london_pct"]
    cont_pct = 100 - rev_pct

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NY Reverses London", f"{rev_pct:.1f}%")
    with col2:
        st.metric("NY Continues London", f"{cont_pct:.1f}%")
    with col3:
        edge = abs(rev_pct - 50)
        direction = "Continuation" if cont_pct > rev_pct else "Reversal"
        st.metric("Edge Towards", direction, f"{edge:.1f}% vs 50/50")

    st.markdown("---")

    # Pie / donut
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = go.Figure(go.Pie(
            labels=["NY Reverses London", "NY Continues London"],
            values=[rev_pct, cont_pct],
            marker_colors=[COLORS["Pink"], COLORS["NY"]],
            hole=0.55,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Bar chart of direction combos
        combos  = stats["dir_combos"]
        cont_sum = combos["London ↑ → NY ↑"] + combos["London ↓ → NY ↓"]
        rev_sum  = combos["London ↑ → NY ↓"] + combos["London ↓ → NY ↑"]

        fig_combo2 = go.Figure(go.Bar(
            x=list(combos.keys()),
            y=[v / n * 100 for v in combos.values()],
            marker_color=[COLORS["London"], COLORS["Pink"], COLORS["Pink"], COLORS["NY"]],
            text=[f"{v/n*100:.1f}%" for v in combos.values()],
            textposition="outside",
        ))
        fig_combo2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False,
            yaxis=dict(range=[0, 50], gridcolor="rgba(255,255,255,0.07)", title="%"),
            xaxis=dict(showgrid=False),
            height=300,
        )
        st.plotly_chart(fig_combo2, use_container_width=True)

    st.markdown("---")

    # Rolling reversal rate
    st.markdown("#### Rolling Reversal Rate (20-day window)")
    df_rev = df.copy()
    df_rev["ny_reverses"] = (df_rev["london_direction"] != df_rev["ny_direction"]).astype(int)
    df_rev["rolling_rev_rate"] = df_rev["ny_reverses"].rolling(20).mean() * 100

    fig_roll_rev = go.Figure()
    fig_roll_rev.add_trace(go.Scatter(
        x=df_rev.index, y=df_rev["rolling_rev_rate"],
        name="Rolling Reversal Rate",
        line=dict(color=COLORS["Pink"], width=2),
        fill="tozeroy",
        fillcolor="rgba(244,114,182,0.1)",
    ))
    fig_roll_rev.add_hline(
        y=50, line_dash="dash",
        line_color="rgba(255,255,255,0.2)",
        annotation_text="50% (random)",
        annotation_font_color="#64748b",
    )
    fig_roll_rev.add_hline(
        y=rev_pct, line_dash="dot",
        line_color=COLORS["Pink"],
        annotation_text=f"Overall avg: {rev_pct:.1f}%",
        annotation_font_color=COLORS["Pink"],
    )
    fig_roll_rev.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8", showlegend=False,
        yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.07)", title="Reversal %"),
        xaxis=dict(showgrid=False),
        height=320,
    )
    st.plotly_chart(fig_roll_rev, use_container_width=True)

    st.markdown("---")
    # Reversal rate by London range size (quintiles)
    st.markdown("#### Reversal Rate by London Range Size")
    st.caption("Does a larger London range make NY more likely to reverse it?")

    df_rev["london_range_quintile"] = pd.qcut(
        df_rev["london_range"], q=5,
        labels=["Q1 (Smallest)", "Q2", "Q3", "Q4", "Q5 (Largest)"]
    )
    rev_by_quintile = df_rev.groupby("london_range_quintile", observed=True)["ny_reverses"].mean() * 100

    fig_q = go.Figure(go.Bar(
        x=rev_by_quintile.index.astype(str),
        y=rev_by_quintile.values,
        marker_color=[COLORS["Pink"]] * 5,
        text=[f"{v:.1f}%" for v in rev_by_quintile.values],
        textposition="outside",
    ))
    fig_q.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig_q.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8", showlegend=False,
        yaxis=dict(range=[0, 80], gridcolor="rgba(255,255,255,0.07)", title="Reversal %"),
        xaxis=dict(showgrid=False, title="London Range Quintile"),
        height=320,
    )
    st.plotly_chart(fig_q, use_container_width=True)
    st.caption(
        "Q1 = smallest 20% of London ranges, Q5 = largest 20%. "
        "If the Q5 bar is higher, extended London sessions tend to attract NY reversals (liquidity grab / mean reversion)."
    )

    st.markdown("---")
    st.markdown("#### Interpretation")
    if rev_pct > 55:
        interp = (
            f"NY reverses London **{rev_pct:.1f}%** of the time — meaningfully above 50%. "
            "This suggests a **mean-reversion tendency**: London often sets a high or low that NY fades. "
            "This is common in pairs with strong Asian/European flows that get corrected during NY."
        )
    elif rev_pct < 45:
        interp = (
            f"NY continues London's direction **{cont_pct:.1f}%** of the time — above 50%. "
            "This suggests **trend-continuation is the dominant behaviour**: when London moves, NY tends to extend it. "
            "This pattern is common in momentum-driven instruments."
        )
    else:
        interp = (
            f"NY reverses London **{rev_pct:.1f}%** of the time — close to the 50% random baseline. "
            "There is **no strong directional edge** between London close and NY direction for this pair over this period. "
            "Context (London range size, time of year) may reveal hidden structure."
        )
    st.info(interp)

# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA EXPANDER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🗃️ View Raw Session Data"):
    show_df = df.copy()
    show_df.index = show_df.index.strftime("%Y-%m-%d")
    for col in show_df.select_dtypes(include="float").columns:
        show_df[col] = show_df[col].round(5)
    st.dataframe(show_df, use_container_width=True)
    csv = show_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=f"{pair_name.replace('/', '_')}_session_data.csv",
        mime="text/csv",
    )
