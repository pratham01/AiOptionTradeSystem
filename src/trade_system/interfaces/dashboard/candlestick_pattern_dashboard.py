import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime
import os

from trade_system.application.agent.candlestick_pattern_agent import CandlestickPatternAgent
from trade_system.infrastructure.database.connection import get_engine
from sqlalchemy import text

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .research-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .stat-box {
        background-color: #161b22;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #9b5de5;
        text-align: center;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #9b5de5;
        margin-bottom: 5px;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Agent
agent = CandlestickPatternAgent()

st.title("🕯️ Candlestick Pattern & TradingView Lab")
st.caption("Scan daily F&O stock candles for standard pattern formations, evaluate their next-day directional probability, and capture live TradingView charts.")
st.markdown("---")

tab_scan, tab_backtest, tab_viewer = st.tabs([
    "🔍 Active Candlestick Scan",
    "🧪 Pattern Probability Backtester",
    "🖼️ TradingView Chart Viewer"
])

# Helper to load symbols
@st.cache_data(ttl=600)
def load_ohlcv_symbols():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            query = text("SELECT DISTINCT symbol FROM ohlcv_daily ORDER BY symbol ASC")
            df = pd.read_sql(query, conn)
            return df["symbol"].tolist()
    except Exception:
        return ["NSE:NIFTY50-INDEX"]

# Helper to load unique dates from database
@st.cache_data(ttl=600)
def load_ohlcv_dates():
    try:
        df = agent._load_combined_daily_candles()
        if not df.empty:
            dates = sorted(df["timestamp"].dt.date.unique(), reverse=True)
            return [d.strftime("%Y-%m-%d") for d in dates]
    except Exception:
        pass
    return [datetime.now().strftime("%Y-%m-%d")]

symbols = load_ohlcv_symbols()
available_dates = load_ohlcv_dates()

# --- TAB 1: ACTIVE SCAN ---
with tab_scan:
    st.subheader("F&O Candlestick Pattern Signals Scan")
    st.caption("Scans the daily market data to detect completed patterns and evaluates their historical accuracy.")
    
    col_date, col_btn = st.columns([2.0, 1.0])
    with col_date:
        if available_dates:
            selected_scan_date = st.selectbox("📅 Select Scan Date", available_dates, index=0)
        else:
            selected_scan_date = datetime.now().strftime("%Y-%m-%d")
            st.warning("No dates found in database. Defaulting to today.")
    with col_btn:
        st.write("") # spacer
        run_scan = st.button("⚡ Run Candlestick Scan", type="primary", use_container_width=True)
        
    if run_scan:
        with st.spinner(f"Scanning database daily candles for active patterns as of {selected_scan_date}..."):
            try:
                active_setups = agent.scan_active_setups(target_date=selected_scan_date)
                st.session_state.active_setups = active_setups
                st.session_state.last_scan_date = selected_scan_date
            except Exception as e:
                st.error(f"Failed to scan setups: {e}")
                
    if "active_setups" in st.session_state:
        setups = st.session_state.active_setups
        scan_date_used = st.session_state.get("last_scan_date", selected_scan_date)
        
        st.markdown(f"#### Active Signals Detected as of **{scan_date_used}**")
        
        if not setups:
            st.info(f"No active candlestick pattern signals detected on {scan_date_used}. (Since these are daily patterns, signals are highly selective).")
        else:
            df_setups = pd.DataFrame(setups)
            st.dataframe(
                df_setups,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Hist Count": st.column_config.NumberColumn("Hist Occurrences"),
                    "t-stat": st.column_config.NumberColumn("t-statistic", format="%.3f")
                }
            )
            
            # TradingView Capture Section
            st.markdown("### 📸 Capture Live TradingView Chart")
            clean_symbols = [s["Symbol"] for s in setups]
            col_sel, col_cap = st.columns([2.0, 1.0])
            with col_sel:
                selected_sym = st.selectbox("Select symbol to capture:", clean_symbols)
            with col_cap:
                st.write("") # Spacer
                if st.button("Capture TradingView Chart", use_container_width=True):
                    with st.spinner(f"Launching Playwright to capture NSE:{selected_sym} chart..."):
                        out_dir = Path("data/screenshots")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{selected_sym}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        out_path = out_dir / filename
                        
                        success = agent.capture_tradingview_screenshot(f"NSE:{selected_sym}-EQ", str(out_path))
                        if success:
                            st.success(f"Successfully captured chart for NSE:{selected_sym}!")
                            st.image(str(out_path), caption=f"TradingView Chart: NSE:{selected_sym}")
                        else:
                            st.error(f"Failed to capture chart. Check Playwright installations or logs.")
    else:
        st.info("Select a date and click the button above to run the scan on the F&O universe.")

# --- TAB 2: BACKTESTER ---
with tab_backtest:
    st.subheader("Candlestick Pattern Probability Backtester")
    st.caption("Test the historical win rate and profit expectancy of a specific candlestick pattern on any stock.")
    
    col_sym, col_pat = st.columns(2)
    with col_sym:
        sel_symbol = st.selectbox("Underlying Stock", symbols, index=symbols.index("NSE:NIFTY50-INDEX") if "NSE:NIFTY50-INDEX" in symbols else 0)
    with col_pat:
        sel_pattern = st.selectbox(
            "Candlestick Pattern",
            options=["Bullish Engulfing", "Bearish Engulfing", "Hammer", "Shooting Star", "Doji", "Morning Star", "Evening Star"]
        )
        
    if st.button("🔬 Calculate Pattern Probabilities", type="primary", use_container_width=True):
        with st.spinner(f"Evaluating {sel_pattern} pattern over history for {sel_symbol}..."):
            try:
                stats = agent.calculate_probabilities(sel_symbol)
                st.session_state.pattern_stats = {
                    "symbol": sel_symbol,
                    "pattern": sel_pattern,
                    "stats": stats.get(sel_pattern, {})
                }
            except Exception as e:
                st.error(f"Failed to run pattern backtest: {e}")
                
    if "pattern_stats" in st.session_state:
        p_stats = st.session_state.pattern_stats
        symbol_name = p_stats["symbol"].split(":")[-1].replace("-EQ", "")
        pattern_name = p_stats["pattern"]
        stats_val = p_stats["stats"]
        
        if not stats_val or stats_val.get("count", 0) == 0:
            st.warning(f"No occurrences of {pattern_name} pattern were detected in the history of {symbol_name}.")
        else:
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #00d084;">
                    <div class="stat-val">{stats_val['count']}</div>
                    <div class="stat-label">Total Signals</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #ffb703;">
                    <div class="stat-val">{stats_val['win_rate']:.1f}%</div>
                    <div class="stat-label">Next-Day Win Rate</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #9b5de5;">
                    <div class="stat-val">{stats_val['avg_pnl']:+.2f}%</div>
                    <div class="stat-label">Avg Next-Day Return</div>
                </div>
                """, unsafe_allow_html=True)
            with sc4:
                st.markdown(f"""
                <div class="stat-box" style="border-left-color: #ff4d6d;">
                    <div class="stat-val">{stats_val['t_stat']:.3f}</div>
                    <div class="stat-label">t-statistic</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            is_sig = abs(stats_val["t_stat"]) >= 2.0
            verdict_color = "#00d084" if is_sig else "#ff4d6d"
            verdict_text = "Statistically Significant Edge!" if is_sig else "Not Statistically Significant (Noise)"
            
            st.markdown(f"""
            <div style="background:#1e2130; padding:15px; border-radius:8px; border-left:6px solid {verdict_color};">
                <h4 style="margin:0 0 5px 0; color:{verdict_color};">{verdict_text}</h4>
                <p style="margin:0; color:#c9d1d9; font-size:0.95rem;">
                    In the history of <b>{symbol_name}</b>, the <b>{pattern_name}</b> pattern has triggered <b>{stats_val['count']}</b> times. 
                    The average return on the next trading session is <b>{stats_val['avg_pnl']:+.2f}%</b> with a win rate of <b>{stats_val['win_rate']:.1f}%</b>. 
                    t-statistic: <b>{stats_val['t_stat']:.3f}</b>. 
                    {f"This pattern has a mathematically verified predictive edge on {symbol_name}." if is_sig else f"The low t-statistic indicates that price changes following this pattern could be due to random noise."}
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 3: VIEWER ---
with tab_viewer:
    st.subheader("Saved TradingView Chart Screenshots")
    st.caption("Browse and view live TradingView chart screenshots captured by the agent.")
    
    screenshot_dir = Path("data/screenshots")
    if not screenshot_dir.exists():
        st.info("No TradingView charts have been captured yet. Use the scanner tab to capture some!")
    else:
        files = sorted(list(screenshot_dir.glob("*.png")), key=lambda x: os.path.getmtime(x), reverse=True)
        if not files:
            st.info("No screenshot files found in the directory.")
        else:
            file_names = [f.name for f in files]
            sel_file = st.selectbox("Select saved chart to view:", file_names)
            
            if sel_file:
                img_path = screenshot_dir / sel_file
                file_time = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime('%Y-%m-%d %H:%M:%S')
                st.write(f"📅 Captured on: **{file_time}**")
                st.image(str(img_path), use_container_width=True)
                
                # Delete option
                if st.button("🗑️ Delete Saved Screenshot"):
                    try:
                        os.remove(img_path)
                        st.success(f"Deleted {sel_file}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete file: {e}")
