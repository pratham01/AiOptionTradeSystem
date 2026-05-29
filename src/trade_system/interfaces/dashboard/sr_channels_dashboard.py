import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import logging

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.data.fo_universe import get_fo_universe
from trade_system.application.indicators.support_resistance_channels import (
    SupportResistanceChannelDetector,
)
from trade_system.interfaces.live.helpers import resample_to_timeframe

LOGGER = logging.getLogger(__name__)

# ── PAGE HEADER ──────────────────────────────────────────────────────────────
st.markdown("## 🏗️ Support & Resistance Channels")
st.caption(
    "Automatically detected multi-pivot support & resistance zones ranked by "
    "composite strength. Zones turn green (support), red (resistance), or gray "
    "(price inside channel)."
)
st.markdown("---")

# ── SIDEBAR PARAMETERS ──────────────────────────────────────────────────────
st.sidebar.title("⚙️ SR Channel Settings")

pivot_period = st.sidebar.slider(
    "Pivot Period",
    min_value=4,
    max_value=30,
    value=10,
    help="Bars left & right to confirm a pivot high/low.",
)
channel_width = st.sidebar.slider(
    "Max Channel Width %",
    min_value=1,
    max_value=8,
    value=5,
    help="Maximum allowed width of a channel as % of the 300-bar range.",
)
min_strength = st.sidebar.slider(
    "Minimum Strength",
    min_value=1,
    max_value=10,
    value=1,
    help="Channel must contain at least this many pivot points.",
)
max_num_sr = st.sidebar.slider(
    "Max S/R Channels",
    min_value=1,
    max_value=10,
    value=6,
    help="Maximum number of channels to display.",
)
loopback = st.sidebar.slider(
    "Loopback Period",
    min_value=100,
    max_value=400,
    value=290,
    help="How many bars back to search for pivots.",
)
source_mode = st.sidebar.selectbox(
    "Pivot Source",
    options=["High/Low", "Close/Open"],
    index=0,
    help="Use High/Low or Close/Open for pivot detection.",
)
show_pivots = st.sidebar.checkbox("Show Pivot Points", value=True)
show_breaks = st.sidebar.checkbox("Show Broken S/R Markers", value=True)

# ── SYMBOL & TIMEFRAME SELECTORS ────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    indices = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
    fo_stocks = get_fo_universe()
    all_symbols = indices + sorted(fo_stocks)

    selected_symbol = st.selectbox(
        "Select Asset",
        options=all_symbols,
        index=0,
        help="Select an index or F&O stock to analyze.",
    )

with col2:
    selected_tf = st.selectbox(
        "Select Timeframe",
        options=["15 Minute", "1 Hour", "Daily"],
        index=2,
        help="Timeframe of candles for S/R channel calculation.",
    )


# ── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    engine = get_engine()

    if timeframe == "Daily":
        query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE symbol = :symbol
            ORDER BY timestamp ASC
        """)
        try:
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"symbol": symbol})
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch daily candles for {symbol}: {e}")
            return pd.DataFrame()
    else:
        query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_15m
            WHERE symbol = :symbol AND timestamp >= date('now', '-35 days')
            ORDER BY timestamp ASC
        """)
        try:
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"symbol": symbol})
            if df.empty:
                return pd.DataFrame()
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            if timeframe == "1 Hour":
                df_indexed = df.set_index("timestamp")
                df_resampled = resample_to_timeframe(df_indexed, 60)
                return df_resampled.reset_index()
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch 15m candles for {symbol}: {e}")
            return pd.DataFrame()


# ── MAIN LOGIC ──────────────────────────────────────────────────────────────
df_candles = load_candles(selected_symbol, selected_tf)

if df_candles.empty:
    st.warning(
        f"No candle data found for **{selected_symbol}** on **{selected_tf}**. "
        "Please ensure the data feed or history fetcher is active."
    )
else:
    # Run detector
    detector = SupportResistanceChannelDetector(
        pivot_period=pivot_period,
        source=source_mode,
        channel_width_pct=channel_width,
        min_strength=min_strength,
        max_num_sr=max_num_sr,
        loopback=loopback,
    )

    snapshots = detector.calculate(df_candles)

    # Use the last 200 bars for chart display
    tail_n = min(200, len(df_candles))
    df_chart = df_candles.tail(tail_n).copy()
    chart_start_idx = len(df_candles) - tail_n

    # Latest snapshot
    latest = snapshots[-1] if snapshots else None

    # ── BUILD PLOTLY CHART ───────────────────────────────────────────────
    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_chart["timestamp"],
            open=df_chart["open"],
            high=df_chart["high"],
            low=df_chart["low"],
            close=df_chart["close"],
            name="Price",
            increasing_line_color="#089981",
            decreasing_line_color="#f23645",
        )
    )

    # Draw S/R channel rectangles from the latest snapshot
    if latest and latest.channels:
        color_map = {
            "resistance": "rgba(255, 82, 82, 0.12)",
            "support": "rgba(0, 230, 118, 0.12)",
            "inside": "rgba(158, 158, 158, 0.12)",
        }
        border_map = {
            "resistance": "rgba(255, 82, 82, 0.55)",
            "support": "rgba(0, 230, 118, 0.55)",
            "inside": "rgba(158, 158, 158, 0.55)",
        }
        for ch in latest.channels:
            fill_c = color_map.get(ch.channel_type, color_map["inside"])
            bord_c = border_map.get(ch.channel_type, border_map["inside"])
            fig.add_shape(
                type="rect",
                x0=df_chart["timestamp"].iloc[0],
                x1=df_chart["timestamp"].iloc[-1],
                y0=ch.low,
                y1=ch.high,
                fillcolor=fill_c,
                line=dict(width=1, color=bord_c),
                layer="below",
            )
            # Label on right edge
            label_color = (
                "#ff5252" if ch.channel_type == "resistance"
                else "#00e676" if ch.channel_type == "support"
                else "#9e9e9e"
            )
            mid = (ch.high + ch.low) / 2
            fig.add_annotation(
                x=df_chart["timestamp"].iloc[-1],
                y=mid,
                text=f"{'R' if ch.channel_type == 'resistance' else 'S' if ch.channel_type == 'support' else '~'} "
                     f"{ch.high:.2f} – {ch.low:.2f}",
                showarrow=False,
                font=dict(size=9, color=label_color, family="Courier New"),
                xanchor="left",
                bgcolor="rgba(30,30,30,0.7)",
                borderpad=2,
            )

    # Pivot point markers
    if show_pivots:
        ph_x, ph_y, pl_x, pl_y = [], [], [], []
        for i in range(chart_start_idx, len(df_candles)):
            snap = snapshots[i]
            ts = df_candles["timestamp"].iloc[i]
            if snap.pivot_high is not None:
                # Pivot sits pivot_period bars back
                piv_idx = max(0, i - pivot_period)
                ph_x.append(df_candles["timestamp"].iloc[piv_idx])
                ph_y.append(snap.pivot_high)
            if snap.pivot_low is not None:
                piv_idx = max(0, i - pivot_period)
                pl_x.append(df_candles["timestamp"].iloc[piv_idx])
                pl_y.append(snap.pivot_low)

        if ph_x:
            fig.add_trace(
                go.Scatter(
                    x=ph_x,
                    y=ph_y,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color="#ff5252"),
                    name="Pivot High",
                    hovertemplate="PH ₹%{y:.2f}<extra></extra>",
                )
            )
        if pl_x:
            fig.add_trace(
                go.Scatter(
                    x=pl_x,
                    y=pl_y,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color="#00e676"),
                    name="Pivot Low",
                    hovertemplate="PL ₹%{y:.2f}<extra></extra>",
                )
            )

    # Break markers
    if show_breaks:
        rb_x, rb_y, sb_x, sb_y = [], [], [], []
        for i in range(chart_start_idx, len(df_candles)):
            snap = snapshots[i]
            if snap.break_event:
                ts = df_candles["timestamp"].iloc[i]
                if snap.break_event.break_type == "resistance_broken":
                    rb_x.append(ts)
                    rb_y.append(df_candles["low"].iloc[i] * 0.999)
                else:
                    sb_x.append(ts)
                    sb_y.append(df_candles["high"].iloc[i] * 1.001)

        if rb_x:
            fig.add_trace(
                go.Scatter(
                    x=rb_x,
                    y=rb_y,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#00e676"),
                    name="Resistance Broken",
                    hovertemplate="Resistance Broken<extra></extra>",
                )
            )
        if sb_x:
            fig.add_trace(
                go.Scatter(
                    x=sb_x,
                    y=sb_y,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="#ff5252"),
                    name="Support Broken",
                    hovertemplate="Support Broken<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"Support & Resistance Channels — {selected_symbol} ({selected_tf})",
        xaxis_title="Timeline",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False,
        height=650,
        template="plotly_dark",
        margin=dict(l=30, r=120, t=50, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key="sr_channel_chart")

    # ── CHANNEL SUMMARY TABLE ────────────────────────────────────────────
    st.subheader("🏛️ Active Support & Resistance Zones")
    if latest and latest.channels:
        rows = []
        cur_close = df_candles["close"].iloc[-1]
        for i, ch in enumerate(latest.channels):
            emoji = (
                "🔴" if ch.channel_type == "resistance"
                else "🟢" if ch.channel_type == "support"
                else "⚪"
            )
            dist = ((ch.high + ch.low) / 2 - cur_close) / cur_close * 100
            rows.append(
                {
                    "#": i + 1,
                    "Type": f"{emoji} {ch.channel_type.title()}",
                    "Zone High (₹)": f"{ch.high:.2f}",
                    "Zone Low (₹)": f"{ch.low:.2f}",
                    "Width (₹)": f"{ch.high - ch.low:.2f}",
                    "Distance %": f"{dist:+.2f}%",
                }
            )
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True
        )
    else:
        st.info("No active S/R channels detected with the current parameters.")

    # ── BREAK EVENT LOG ──────────────────────────────────────────────────
    st.subheader("⚡ Recent Break Events")
    break_rows = []
    for i in range(max(0, len(snapshots) - tail_n), len(snapshots)):
        snap = snapshots[i]
        if snap.break_event:
            be = snap.break_event
            emoji = "🟢 ↑" if be.break_type == "resistance_broken" else "🔴 ↓"
            break_rows.append(
                {
                    "Time": be.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Event": f"{emoji} {be.break_type.replace('_', ' ').title()}",
                    "Level": f"{be.level_high:.2f} – {be.level_low:.2f}",
                    "Close (₹)": f"{be.close:.2f}",
                }
            )
    if break_rows:
        st.dataframe(
            pd.DataFrame(break_rows[::-1]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No S/R break events in the visible range.")

# ── EXPLAINER ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
### 🧠 How to read Support & Resistance Channels

- **🟢 Support Zone**: Both the upper and lower boundaries are below the current
  close. Price has historically bounced off this zone.
- **🔴 Resistance Zone**: Both boundaries are above the current close. Price has
  historically been rejected here.
- **⚪ Inside Zone**: The current close sits within the channel. This indicates
  a decision zone — a breakout or breakdown is imminent.
- **Pivot Points**: Triangle markers show confirmed swing highs (▼ red) and swing
  lows (▲ green). Channels are formed by clustering nearby pivots.
- **Break Events**: When the close crosses above a resistance channel (🟢 ↑) or
  below a support channel (🔴 ↓), a break marker is plotted. These are
  high-probability momentum continuation signals.
- **Strength Ranking**: Channels with more clustered pivot points and more
  historical price touches rank higher and are displayed first.
""")
