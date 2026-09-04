import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from sqlalchemy import text
import logging

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.strategy.application.indicators.volumetric_order_flow import VolumetricOrderFlowDetector
from trade_system.interfaces.live.helpers import resample_to_timeframe

LOGGER = logging.getLogger(__name__)

# --- PAGE SETUP ---
st.markdown("## 📊 Volumetric Order Flow Structure [LuxAlgo]")
st.caption("Noise-filtered institutional order block structure, volume points of control (POC), and sweep manipulation trackers.")
st.markdown("---")

# Sidebar Configuration / Inputs
st.sidebar.title("⚙️ Detection Parameters")

pivot_len = st.sidebar.slider("Pivot Length", min_value=1, max_value=10, value=3, help="Bars on left and right for pivot detection.")
vol_lookback = st.sidebar.slider("Volume Lookback", min_value=5, max_value=100, value=20, help="Period used to calculate relative volume changes.")
max_obs = st.sidebar.slider("Max Recent Blocks", min_value=1, max_value=20, value=5, help="Maximum number of active order blocks to display.")
bars_to_show = st.sidebar.slider("Bars to Display", min_value=30, max_value=250, value=100, step=10, help="Number of candles shown on the chart.")
hide_overlapping = st.sidebar.checkbox("Hide Overlapping Blocks", value=False, help="Automatically remove overlapping order blocks of lower volume.")
show_manipulation = st.sidebar.checkbox("Show Manipulation Bubbles", value=True, help="Track and flag institutional liquidity sweeps.")
manip_size = st.sidebar.slider("Bubble Sensitivity", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Adjust bubble sizing sensitivity based on sweep volume.")

# Data Selection controls
col1, col2 = st.columns(2)
with col1:
    # Build list of symbols: Indices + F&O Universe
    indices = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
    fo_stocks = get_fo_universe()
    all_symbols = indices + sorted(fo_stocks)
    
    selected_symbol = st.selectbox(
        "Select Asset",
        options=all_symbols,
        index=0,
        help="Select an index or F&O stock to analyze."
    )
    
with col2:
    selected_tf = st.selectbox(
        "Select Timeframe",
        options=["15 Minute", "1 Hour", "Daily"],
        index=0,
        help="Timeframe of candles to calculate order flow structure on."
    )

# --- DATABASE FETCH & PROCESS ---
@st.cache_data(ttl=60)
def load_historical_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    engine = get_engine()
    
    if timeframe == "Daily":
        # Load daily candles
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
                df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch daily candles for {symbol}: {e}")
            return pd.DataFrame()
            
    else:
        # Load 15m candles
        # Fetch last 30 days of data to have sufficient history for 15m and 1h resample
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
                
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
            
            if timeframe == "1 Hour":
                # Set index temporarily for resampling helper
                df_indexed = df.set_index("timestamp")
                df_resampled = resample_to_timeframe(df_indexed, 60)
                return df_resampled.reset_index()
            else:
                return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch 15m candles for {symbol}: {e}")
            return pd.DataFrame()

# Load candles
df_candles = load_historical_candles(selected_symbol, selected_tf)

if df_candles.empty:
    st.warning(f"No candle data found in the database for {selected_symbol} on {selected_tf}. Please ensure the live data feed or history fetcher is active.")
else:
    # Run Detector
    detector = VolumetricOrderFlowDetector(
        pivot_length=pivot_len,
        vol_lookback=vol_lookback,
        max_obs=max_obs,
        hide_overlapping=hide_overlapping,
        show_manipulation=show_manipulation,
        manip_size=manip_size
    )
    
    # Run calculation
    # returns list of list of VolumetricOrderBlock
    history = detector.calculate(df_candles)
    active_obs = history[-1] if history else []

    # Filter recent N bars selected by user
    df_chart = df_candles.tail(bars_to_show).copy()
    
    # Format time labels for categorical continuous axis (eliminates overnight and weekend blank gaps)
    if selected_tf == "Daily":
        df_chart["time_label"] = df_chart["timestamp"].dt.strftime("%d %b %Y")
    else:
        df_chart["time_label"] = df_chart["timestamp"].dt.strftime("%d %b %H:%M")

    # Map timestamps to categorical strings
    time_to_label = dict(zip(df_chart["timestamp"], df_chart["time_label"]))
    visible_timestamps = df_chart["timestamp"].tolist()
    min_time = visible_timestamps[0]
    max_time = visible_timestamps[-1]
    first_label = df_chart["time_label"].iloc[0]
    last_label = df_chart["time_label"].iloc[-1]
    
    # Build Plotly Chart
    fig = go.Figure()

    # Candlestick Trace with proper TradingView palette
    fig.add_trace(go.Candlestick(
        x=df_chart['time_label'],
        open=df_chart['open'],
        high=df_chart['high'],
        low=df_chart['low'],
        close=df_chart['close'],
        name="Price Action",
        increasing_line_color='#089981',
        decreasing_line_color='#f23645'
    ))

    # Add active order blocks to the chart
    legend_added = {"BOS": False, "CHoCH": False}
    
    for ob in active_obs:
        # We only draw blocks that intersect with the visible chart window
        if ob.breakout_time > max_time:
            continue
            
        # Match breakout time to categorical x coordinate
        # If breakout happened before the visible chart window, start at first visible candle
        if ob.breakout_time <= min_time:
            x_start = first_label
        else:
            matched = [t for t in visible_timestamps if t >= ob.breakout_time]
            x_start = time_to_label[matched[0]] if matched else first_label
            
        x_end = last_label
        color_fill = ob.css_color
        
        # Add background shape for the Order Block Range
        fig.add_shape(
            type="rect",
            x0=x_start,
            x1=x_end,
            y0=ob.low,
            y1=ob.high,
            fillcolor=color_fill,
            opacity=0.15,
            line=dict(width=1.2, color=color_fill),
            layer="below"
        )
        
        # Add Point of Control (POC) level line inside the block
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[ob.poc_level, ob.poc_level],
            mode="lines",
            line=dict(color="#ff9800", width=1.5, dash="dot"),
            name="Volume POC",
            showlegend=False,
            hoverinfo="y+name"
        ))

        # Add BOS/CHoCH structural label text at breakout candle
        if ob.breakout_time in time_to_label:
            lbl_x = time_to_label[ob.breakout_time]
            show_leg = not legend_added.get(ob.label_text, False)
            fig.add_trace(go.Scatter(
                x=[lbl_x],
                y=[ob.high if ob.is_bullish else ob.low],
                mode="markers+text",
                marker=dict(symbol="triangle-up" if ob.is_bullish else "triangle-down", size=9, color=ob.css_color),
                text=[ob.label_text],
                textposition="top center" if ob.is_bullish else "bottom center",
                textfont=dict(color=ob.css_color, size=11, family="Courier New", weight="bold"),
                name=ob.label_text,
                showlegend=show_leg
            ))
            legend_added[ob.label_text] = True

        # Render Manipulation Sweeps (swept high/low wicks)
        if show_manipulation:
            for m in ob.manipulations:
                m_ts = m["timestamp"]
                if m_ts in time_to_label:
                    fig.add_trace(go.Scatter(
                        x=[time_to_label[m_ts]],
                        y=[m["price"]],
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=14 if m["size"] == "huge" else (11 if m["size"] == "large" else (8 if m["size"] == "normal" else 5)),
                            color=ob.css_color,
                            opacity=0.7,
                            line=dict(color="white", width=1)
                        ),
                        name="Liquidity Sweep",
                        showlegend=False,
                        hovertemplate=f"<b>{m['type']}</b><br>Price: ₹{m['price']:.2f}<br>Volume: {m['volume']:.0f}<br>Time: %{{x}}<extra></extra>"
                    ))

    # Layout formatting
    fig.update_layout(
        title=f"Volumetric Order Flow Structure [LuxAlgo] — {selected_symbol} ({selected_tf})",
        xaxis_title="Timeline (Trading Sessions)",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False,
        height=680,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.6)",
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(
        type="category",
        nticks=16,
        tickangle=-30,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)"
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)"
    )

    st.plotly_chart(fig, use_container_width=True, key="volumetric_chart")

    # Render summary analytics
    st.subheader("🏛️ Volumetric Structure Summary")
    if active_obs:
        ob_rows = []
        for ob in active_obs:
            type_str = "🟢 Bullish (Demand)" if ob.is_bullish else "🔴 Bearish (Supply)"
            sweeps_count = len(ob.manipulations)
            latest_sweep = ob.manipulations[-1]["timestamp"].strftime("%H:%M") if sweeps_count > 0 else "None"
            
            ob_rows.append({
                "Origin Time": ob.start_time.strftime("%Y-%m-%d %H:%M"),
                "Breakout Time": ob.breakout_time.strftime("%Y-%m-%d %H:%M"),
                "Type": type_str,
                "Structure Label": ob.label_text,
                "Range High (₹)": f"{ob.high:.2f}",
                "Range Low (₹)": f"{ob.low:.2f}",
                "Volume POC (₹)": f"{ob.poc_level:.2f}",
                "Base Vol": f"{ob.volume:,.0f}",
                "Sweeps Count": sweeps_count,
                "Last Sweep": latest_sweep
            })
        
        st.dataframe(pd.DataFrame(ob_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No active, unmitigated Volumetric Order Blocks detected on the chart. Prices have currently mitigated all historical structural boundaries.")

# --- HELP / EXPLANATIONS SECTION ---
st.divider()
st.markdown("""
### 🧠 How to read Volumetric Order Flow Structure:
- **Order Blocks (BOS / CHoCH):** Identified when price breaks a structural pivot high or low with volume validation.
  - **🟢 Bullish Block (Demand Zone):** Created on crossover above a pivot high. Serves as a support area. Mitigated if price closes below the block range.
  - **🔴 Bearish Block (Supply Zone):** Created on crossunder below a pivot low. Serves as a resistance area. Mitigated if price closes above the block range.
- **Volume POC (Point of Control):** The dotted yellow line inside the block shows the price level with the highest volume profile density inside the breakout bar. It represents the center of institutional commitment.
- **Liquidity Sweeps (Manipulation Bubbles):** Bubbles plotted at wicks sweeping above block highs or below block lows. This indicates institutions taking liquidity (stop hunts) before reversing. Large bubbles denote heavy volume sweeps.
""")
