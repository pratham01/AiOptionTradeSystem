import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from pathlib import Path
import logging

from trade_system.application.analysis.smart_oi_analyzer import SmartOIAnalyzer
from trade_system.infrastructure.database.connection import get_engine
from trade_system.config import Settings
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

LOGGER = logging.getLogger(__name__)

# --- STYLING & CUSTOM CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    div[data-testid="stVerticalBlock"] > div { margin-top: -0.5rem !important; }
    
    /* Smart Cards */
    .status-card {
        background: #1e2130;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .status-title {
        font-size: 0.9rem;
        color: #8b949e;
        margin-bottom: 5px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .status-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .status-desc {
        font-size: 0.8rem;
        color: #8b949e;
    }
    
    /* Transition Timeline */
    .timeline-item {
        background: #161b22;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 4px solid #30363d;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .timeline-time {
        color: #8b949e;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .timeline-signal {
        font-weight: bold;
        margin-left: 5px;
    }
    .timeline-desc {
        color: #c9d1d9;
        margin-top: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_broker():
    """Load and authenticate Fyers broker client."""
    try:
        settings = Settings.load()
        auth_service = FyersAuthService(settings)
        token = auth_service.get_valid_token()
        if not token:
            return None
        broker = FyersBroker(
            client_id=settings.fyers.client_id,
            access_token=token,
            user_id=settings.fyers.user_id,
            authenticator=auth_service.authenticator
        )
        if broker.authenticate():
            return broker
    except Exception as e:
        LOGGER.error(f"Fyers broker creation failed: {e}")
    return None

def load_db_dates(symbol: str) -> list[str]:
    """Get all unique dates in the database for option chain snapshots."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            query = text("""
                SELECT DISTINCT date(timestamp) as snapshot_date 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                ORDER BY snapshot_date DESC
            """)
            df = pd.read_sql(query, conn, params={"symbol": symbol})
            return df["snapshot_date"].tolist()
    except Exception as e:
        LOGGER.error(f"Error loading dates: {e}")
        return []

def load_db_snapshots(symbol: str, target_date: str) -> list[tuple[datetime, pd.DataFrame]]:
    """Load all snapshots for a specific symbol and date."""
    try:
        engine = get_engine()
        snapshots = []
        with engine.connect() as conn:
            from sqlalchemy import text
            query = text("""
                SELECT DISTINCT timestamp 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                  AND date(timestamp) = :date_str 
                ORDER BY timestamp ASC
            """)
            ts_df = pd.read_sql(query, conn, params={"symbol": symbol, "date_str": target_date})
            if ts_df.empty:
                return []
            
            for ts in ts_df["timestamp"]:
                data_query = text("""
                    SELECT * 
                    FROM option_chain_data 
                    WHERE underlying_symbol = :symbol 
                      AND timestamp = :ts
                """)
                df = pd.read_sql(data_query, conn, params={"symbol": symbol, "ts": ts})
                ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                snapshots.append((ts_dt, df))
        return snapshots
    except Exception as e:
        LOGGER.error(f"Error loading snapshots: {e}")
        return []

def load_db_price_data(symbol: str, target_date: str) -> pd.DataFrame:
    """Load 1m OHLCV price data for the selected date."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            query = text("""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_1m 
                WHERE symbol = :symbol 
                  AND date(timestamp) = :date_str 
                ORDER BY timestamp ASC
            """)
            df = pd.read_sql(query, conn, params={"symbol": symbol, "date_str": target_date})
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    except Exception as e:
        LOGGER.error(f"Error loading price data: {e}")
        return pd.DataFrame()

# --- MAIN PAGE SETUP ---
st.title("📊 Nifty 50 Smart OI: Comparing With Price Action")
st.caption("Noise-filtered Institutional derivatives positioning correlated with real-time price action")
st.markdown("---")

# Sidebar Configuration
st.sidebar.title("⚙️ Smart OI Controls")
st.sidebar.markdown("---")

# Select symbol
symbol_map = {
    "NIFTY 50": "NSE:NIFTY50-INDEX",
    "NIFTY BANK": "NSE:NIFTYBANK-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX"
}
selected_symbol_label = st.sidebar.selectbox("Select Underlying", list(symbol_map.keys()), index=0)
db_symbol = symbol_map[selected_symbol_label]

# Load dates
available_dates = load_db_dates(db_symbol)
if not available_dates:
    st.error(f"No option chain data found in database for {selected_symbol_label}.")
    st.stop()

selected_date = st.sidebar.selectbox("Select Analysis Date", available_dates, index=0)

# Load snapshots and price data
snapshots = load_db_snapshots(db_symbol, selected_date)
price_df = load_db_price_data(db_symbol, selected_date)

if not snapshots:
    st.warning(f"No snapshots loaded for {selected_symbol_label} on {selected_date}.")
    st.stop()

# Load latest and previous snapshot
latest_ts, latest_oc = snapshots[-1]
prev_oc = snapshots[-2][1] if len(snapshots) > 1 else None

# Initialize Analyzer
analyzer = SmartOIAnalyzer(db_symbol)

# Get current spot price from price dataframe if available, otherwise estimate
spot_price = None
if not price_df.empty:
    # Filter price data up to the snapshot time
    snap_time = latest_ts
    snap_price_df = price_df[price_df["timestamp"] <= snap_time]
    if not snap_price_df.empty:
        spot_price = snap_price_df.iloc[-1]["close"]
    else:
        spot_price = price_df.iloc[-1]["close"]

# Run analysis
analysis = analyzer.analyze_smart_oi(latest_oc, prev_oc, spot_price=spot_price)
signal = analysis["signal"]
signal_strikes = analysis["signal_strikes"]
summary = analysis["summary"]

# Run Confluence & Divergence detection
confluence_data = {}
if not price_df.empty:
    snap_price_df = price_df[price_df["timestamp"] <= latest_ts]
    confluence_data = analyzer.detect_confluence_divergence(signal, snap_price_df if not snap_price_df.empty else price_df)

# --- 1. CURRENT SMART OI VERDICT ---
st.subheader("🏁 Swarm Verdict & Metrics")
c1, c2, c3, c4 = st.columns(4)

# Set signal badge style
signal_styles = {
    "long buildup": {"color": "#00d084", "desc": "Put Writing Accumulation (Bullish Positioning)"},
    "short buildup": {"color": "#ff4d6d", "desc": "Call Writing (Bearish Positioning)"},
    "unwind": {"color": "#9b5de5", "desc": "Position Exiting (Unwinding)"},
    "neutral": {"color": "#00b4d8", "desc": "No Clear Institutional Commitment"}
}
style = signal_styles.get(signal, {"color": "#8b949e", "desc": "Unknown"})

with c1:
    st.markdown(f"""
    <div class="status-card" style="border-top: 5px solid {style['color']};">
        <div class="status-title">Smart OI Signal</div>
        <div class="status-value" style="color: {style['color']};">{signal.upper()}</div>
        <div class="status-desc">{style['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

# Calculate Put Call Ratio (PCR) from raw data
total_ce_oi = latest_oc[latest_oc["option_type"] == "CE"]["oi"].sum()
total_pe_oi = latest_oc[latest_oc["option_type"] == "PE"]["oi"].sum()
pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

# Calculate Max Pain
# Minimize total pain
strikes = latest_oc["strike"].unique()
total_pain = []
for s in strikes:
    pain = 0
    # Calls pain: buyers profit if strike > s
    # PE pain: buyers profit if strike < s
    for _, row in latest_oc.iterrows():
        stk = row["strike"]
        oi_val = row["oi"]
        opt_type = row["option_type"]
        if opt_type == "CE" and stk < s:
            pain += (s - stk) * oi_val
        elif opt_type == "PE" and stk > s:
            pain += (stk - s) * oi_val
    total_pain.append(pain)
max_pain_strike = strikes[np.argmin(total_pain)] if len(total_pain) > 0 else spot_price

with c2:
    st.markdown(f"""
    <div class="status-card" style="border-top: 5px solid #ffb703;">
        <div class="status-title">Spot Price</div>
        <div class="status-value" style="color: #ffb703;">₹{spot_price:,.2f}</div>
        <div class="status-desc">Snapshot Timestamp: {latest_ts.strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="status-card" style="border-top: 5px solid #00b4d8;">
        <div class="status-title">Put-Call Ratio (PCR)</div>
        <div class="status-value" style="color: #00b4d8;">{pcr:.2f}</div>
        <div class="status-desc">{"🟢 Bullish (>1.0)" if pcr > 1.0 else "🔴 Bearish (<1.0)"} Sentiment</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="status-card" style="border-top: 5px solid #9b5de5;">
        <div class="status-title">Max Pain Strike</div>
        <div class="status-value" style="color: #9b5de5;">₹{int(max_pain_strike):,}</div>
        <div class="status-desc">Expiry-day magnet target</div>
    </div>
    """, unsafe_allow_html=True)

# Narrative Alert box
if confluence_data:
    st.write("")
    alert_type = confluence_data["status"]
    alert_color = "#00d084" if "BULLISH CONFLUENCE" in alert_type else ("#ff4d6d" if "BEARISH CONFLUENCE" in alert_type else ("#ffb703" if "DIVERGENCE" in alert_type else "#00b4d8"))
    
    st.markdown(f"""
    <div style="background:#1e2130; padding:15px; border-radius:8px; border-left:6px solid {alert_color}; margin-bottom:20px;">
        <h4 style="margin:0 0 5px 0; color:{alert_color};">{alert_type}</h4>
        <p style="margin:0; color:#c9d1d9; font-size:0.95rem;">{confluence_data['narrative']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 2. TRANSITIONS & CHARTING ---
tab_chart, tab_transitions, tab_strikes = st.tabs(["📈 Price Action & Smart OI Overlay", "⏱️ Signal Transitions Timeline", "🎯 Filtered Signal Strikes"])

with tab_chart:
    st.subheader("Price vs. Smart OI & technical levels")
    if not price_df.empty:
        from plotly.subplots import make_subplots
        
        # Create dual-axis Plotly Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add Candlesticks on primary y-axis
        fig.add_trace(go.Candlestick(
            x=price_df['timestamp'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name="Price"
        ), secondary_y=False)
        
        # Calculate VWAP
        price_df["typical_price"] = (price_df["high"] + price_df["low"] + price_df["close"]) / 3
        price_df["tp_vol"] = price_df["typical_price"] * price_df["volume"]
        price_df["cum_vol"] = price_df["volume"].cumsum()
        price_df["cum_tp_vol"] = price_df["tp_vol"].cumsum()
        price_df["vwap"] = price_df["cum_tp_vol"] / price_df["cum_vol"].replace(0, 1)
        
        # Add VWAP Line on primary y-axis
        fig.add_trace(go.Scatter(
            x=price_df['timestamp'],
            y=price_df['vwap'],
            line=dict(color="orange", width=2, dash="dash"),
            name="VWAP"
        ), secondary_y=False)
        
        # Add Max Pain Horizontal Line
        fig.add_hline(
            y=max_pain_strike,
            line_dash="dot",
            line_color="#9b5de5",
            annotation_text=f"Max Pain (₹{int(max_pain_strike)})",
            annotation_position="bottom right"
        )
        
        # Calculate signals and PE-CE difference over all database snapshots
        signals_ts = []
        for i in range(len(snapshots)):
            ts_item, oc_item = snapshots[i]
            prev_item = snapshots[i-1][1] if i > 0 else None
            
            # Calculate PE - CE difference (Net Option Writer Positioning)
            ce_oi = oc_item[oc_item["option_type"] == "CE"]["oi"].sum()
            pe_oi = oc_item[oc_item["option_type"] == "PE"]["oi"].sum()
            net_oi_diff = pe_oi - ce_oi
            
            # Find close price
            close_item = price_df[price_df["timestamp"] <= ts_item]
            item_spot = close_item.iloc[-1]["close"] if not close_item.empty else None
            res_item = analyzer.analyze_smart_oi(oc_item, prev_item, spot_price=item_spot)
            
            # Find matching price row
            price_row = price_df[price_df["timestamp"] <= ts_item]
            if not price_row.empty:
                signals_ts.append({
                    "timestamp": price_row.iloc[-1]["timestamp"],
                    "price": price_row.iloc[-1]["close"],
                    "net_oi_diff": net_oi_diff,
                    "signal": res_item["signal"]
                })
        
        # Plot Net PE-CE difference line on secondary y-axis if we have snapshot data
        if signals_ts:
            snap_df = pd.DataFrame(signals_ts).sort_values("timestamp")
            
            # Add Net PE-CE line
            fig.add_trace(go.Scatter(
                x=snap_df['timestamp'],
                y=snap_df['net_oi_diff'],
                line=dict(color="#00e6ff", width=2.5),
                fill='tozeroy',
                fillcolor='rgba(0, 230, 255, 0.1)',
                name="Net PE-CE OI (Support-Resistance)"
            ), secondary_y=True)
            
            # Filter transition points for markers
            last_sig = None
            for item in signals_ts:
                if item["signal"] != last_sig:
                    # Add annotation marker
                    sig_col = signal_styles.get(item["signal"], {"color": "#00b4d8"})["color"]
                    fig.add_annotation(
                        x=item["timestamp"],
                        y=item["price"],
                        text=item["signal"].upper(),
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=sig_col,
                        arrowsize=1,
                        arrowwidth=2,
                        ax=0,
                        ay=-40 if item["signal"] in ["long buildup", "neutral"] else 40,
                        bordercolor=sig_col,
                        borderwidth=1,
                        borderpad=4,
                        bgcolor="#161b22",
                        opacity=0.9
                    )
                    last_sig = item["signal"]
        
        # Style layout
        fig.update_layout(
            title=f"{selected_symbol_label} Price vs Smart OI (PE-CE Net Positioning) Overlay",
            xaxis_title="Time",
            xaxis_rangeslider_visible=False,
            height=550,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Net PE-CE OI (Contracts)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True, key="price_smart_oi_chart")
    else:
        st.info("No intraday price data available to render the candlestick chart.")

with tab_transitions:
    st.subheader("Intraday Signal Transitions Timeline")
    st.caption("Transitions show when institutions are adding or closing positions")
    
    # Calculate transition table
    transition_list = []
    last_sig = None
    for i in range(len(snapshots)):
        ts_item, oc_item = snapshots[i]
        prev_item = snapshots[i-1][1] if i > 0 else None
        
        close_item = price_df[price_df["timestamp"] <= ts_item]
        item_spot = close_item.iloc[-1]["close"] if not close_item.empty else None
        res_item = analyzer.analyze_smart_oi(oc_item, prev_item, spot_price=item_spot)
        
        current_sig = res_item["signal"]
        if current_sig != last_sig:
            transition_list.append({
                "time": ts_item.strftime("%H:%M:%S"),
                "signal": current_sig,
                "spot": item_spot,
                "reasons": f"Bullish flow: {res_item['summary']['bullish_oi_flow']:,} | Bearish flow: {res_item['summary']['bearish_oi_flow']:,} | Unwinding: {res_item['summary']['unwind_oi_flow']:,}"
            })
            last_sig = current_sig

    if transition_list:
        # Display timeline items
        for t in reversed(transition_list):
            sig_col = signal_styles.get(t["signal"], {"color": "#8b949e"})["color"]
            spot_txt = f" @ Spot ₹{t['spot']:,.2f}" if t["spot"] else ""
            st.markdown(f"""
            <div class="timeline-item" style="border-left-color: {sig_col};">
                <span class="timeline-time">⏱️ {t['time']}</span>
                <span class="timeline-signal" style="color: {sig_col};">{t['signal'].upper()}</span>
                <span style="color: #8b949e;">{spot_txt}</span>
                <div class="timeline-desc">{t['reasons']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No signal transitions detected on this day.")

with tab_strikes:
    st.subheader("🎯 Active Institutional Signal Strikes")
    st.caption("Noise-filtered strikes where serious institutional positions are being committed.")
    
    if signal_strikes:
        # Render a table of signal strikes
        df_strikes = pd.DataFrame(signal_strikes)
        
        # Beautify column names and formats
        df_strikes = df_strikes.rename(columns={
            "strike": "Strike Price",
            "option_type": "Option Type",
            "ltp": "Option Price (LTP)",
            "oi": "Open Interest (OI)",
            "oi_change": "OI Change (Contracts)",
            "oi_change_pct": "OI Change %",
            "volume": "Interval Volume",
            "buildup": "Buildup Category",
            "action": "Institutional Action"
        })
        
        # Reorder
        df_strikes = df_strikes[[
            "Strike Price", "Option Type", "Option Price (LTP)", 
            "Open Interest (OI)", "OI Change (Contracts)", "OI Change %",
            "Interval Volume", "Buildup Category", "Institutional Action"
        ]]
        
        # Color coding buildup
        def color_buildup(val):
            if "Short Buildup" in val:
                return "color: #ff4d6d; font-weight: bold;" # Sellers active
            elif "Long Buildup" in val:
                return "color: #00d084; font-weight: bold;" # Buyers active
            elif "Covering" in val:
                return "color: #ffb703;"
            return ""
            
        st.dataframe(
            df_strikes,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike Price": st.column_config.NumberColumn("Strike Price", format="₹%d"),
                "Option Price (LTP)": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                "Open Interest (OI)": st.column_config.NumberColumn("OI", format="%d"),
                "OI Change (Contracts)": st.column_config.NumberColumn("OI Change", format="%+d"),
                "OI Change %": st.column_config.NumberColumn("OI Change %", format="%.2f%%"),
                "Interval Volume": st.column_config.NumberColumn("Volume", format="%d"),
            }
        )
    else:
        st.info("No active signal strikes passed the filters at this snapshot.")

st.markdown("---")

# --- 3. PCR & OPEN INTEREST WALLS ---
st.subheader("🧱 Institutional Walls: CE vs. PE Open Interest")
st.caption("Visualizing filtered CE vs PE positioning side-by-side to pinpoint key support & resistance zones.")

if not latest_oc.empty:
    # Filter latest_oc to near-ATM strikes to keep visualization clean
    atm_strike = summary["atm_strike"]
    visual_df = latest_oc[latest_oc["strike"].apply(lambda s: abs(s - atm_strike) <= 5 * analyzer.strike_step)].copy()
    
    # Sort
    visual_df = visual_df.sort_values("strike")
    
    # Create side-by-side bar chart
    fig_walls = go.Figure()
    
    ce_data = visual_df[visual_df["option_type"] == "CE"]
    pe_data = visual_df[visual_df["option_type"] == "PE"]
    
    fig_walls.add_trace(go.Bar(
        x=ce_data["strike"],
        y=ce_data["oi"],
        name="Call OI (Resistance)",
        marker_color="#ff4d6d"
    ))
    
    fig_walls.add_trace(go.Bar(
        x=pe_data["strike"],
        y=pe_data["oi"],
        name="Put OI (Support)",
        marker_color="#00d084"
    ))
    
    fig_walls.update_layout(
        title="Open Interest Concentration at Key Strikes",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest (Contracts)",
        barmode="group",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig_walls, use_container_width=True, key="oi_walls_chart")
else:
    st.info("Waiting for option chain data to render visual walls...")

# Add help section at bottom
st.divider()
st.markdown("""
### 🧠 How to read Nifty Smart OI:
- **Smart OI shows positioning:** Putting writing accumulation (Put Short Buildup) means institutional traders are selling puts, expecting Nifty to stay above that strike. This is a **Bullish positioning** signal. Call writing accumulation (Call Short Buildup) means institutions are writing calls, creating a **Bearish resistance** wall.
- **Price Action shows direction:** We compare this positioning with the underlying price relative to **VWAP** and recent highs/lows.
- **Bullish Confluence:** Put writing accumulating + Price breaking above VWAP/resistance + Volume expanding. Highly reliable long entries.
- **Bearish Confluence:** Call writing accumulating + Price breaking below VWAP/support + Volume expanding. Highly reliable short entries.
- **Divergence Handling:**
  - *Rising Price + Bearish Smart OI:* Indicates a false rally or short squeeze with weak institutional buying follow-through. Expect reversals.
  - *Falling Price + Bullish Smart OI:* Indicates a false decline or bull trap. Puts are being aggressively written at lows, expecting a reversal.
""")
