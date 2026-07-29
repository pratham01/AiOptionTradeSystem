"""
Gamma Blast Dashboard — Streamlit page for the 0DTE Zero-Hero strategy.

Displays:
  1. Live compression & squeeze monitors for indices (Nifty, BankNifty, Sensex)
  2. Dynamic ATM/OTM options suggestions with real-time premium pricing
  3. Historical backtest performance of the 0DTE strategy
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime
from pathlib import Path
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.analysis.application.analysis.gamma_blast_strategy import GammaBlastDetector

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; }

    .gb-hero {
        background: linear-gradient(135deg, #110c11 0%, #2e0828 50%, #0d0614 100%);
        padding: 0.7rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border-left: 5px solid #d946ef;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .gb-hero h3 { margin: 0; color: #f472b6; font-size: 1.15rem; }
    .gb-hero .hero-sub { color: #cbd5e1; font-size: 0.78rem; }

    .index-card {
        background: #180f2b;
        border: 1px solid rgba(217, 70, 239, 0.2);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .status-squeezed {
        color: #f43f5e;
        font-weight: 700;
        animation: pulse 2s infinite;
    }
    .status-normal {
        color: #64748b;
    }

    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(217, 70, 239, 0.4) 50%, transparent 100%);
        margin: 0.6rem 0;
        border: none;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0.5rem 0 0.4rem;
    }
    .section-header h3 { margin: 0; font-size: 1rem; color: #f1f5f9; }
    .section-header .badge {
        background: rgba(217, 70, 239, 0.2);
        color: #f472b6;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gb-hero">
    <h3>⚡ 0DTE Gamma Blast Lab</h3>
    <span class="hero-sub">Expiry Day Breakouts • Bollinger Band Squeezes • Exponential Options Convexity</span>
</div>
""", unsafe_allow_html=True)


# ── Date & Index Selector ──────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_available_dates():
    engine = get_engine()
    query = text("SELECT DISTINCT date(timestamp) as d FROM ohlcv_1m ORDER BY d DESC")
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return [row[0] for row in result if row[0] is not None]
    except Exception:
        return []

dates = fetch_available_dates()
col_date, col_index = st.columns(2)
with col_date:
    selected = st.selectbox("📅 Trading Date", dates[:30] if dates else ["No data"], index=0)
with col_index:
    underlying = st.selectbox("🎯 Index Symbol", ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX", "NSE:NIFTYBANK-INDEX"], index=0)

if not dates or selected == "No data":
    st.warning("No 1-minute historical index data available in the database.")
    st.stop()

target_date = date.fromisoformat(selected) if isinstance(selected, str) else selected


# ── Expiry Banner & Dynamic Options ────────────────────────────────────────────
detector = GammaBlastDetector()
is_exp = detector.is_expiry_day(underlying, target_date)
expiry_msg = "🔥 Today is weekly expiry day!" if is_exp else "⚖️ Today is NOT expiry day."

st.markdown(f"**Status:** {expiry_msg}")


# ── Squeeze Monitor ────────────────────────────────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <h3>🔍 Volatility Squeeze Monitor</h3>
    <span class="badge">Realtime / Historical</span>
</div>
""", unsafe_allow_html=True)

# Fetch 1m candles for target day up to 14:30
engine = get_engine()
query_spot = text("""
    SELECT timestamp, close, high, low, open
    FROM ohlcv_1m
    WHERE symbol = :symbol AND date(timestamp) = :td
    ORDER BY timestamp ASC
""")
with engine.connect() as conn:
    df_1m = pd.read_sql(query_spot, conn, params={"symbol": underlying, "td": target_date.isoformat()})

if not df_1m.empty:
    df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], format="mixed")
    
    # Calculate Bollinger squeeze on historical day
    df_before_3 = df_1m[df_1m["timestamp"].dt.time <= datetime.strptime("14:30", "%H:%M").time()]
    is_sq, current_w = detector.check_volatility_squeeze(df_before_3)
    
    obs_high, obs_low = detector.get_consolidation_range(df_1m, target_date)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="index-card">
            <h4>Bollinger Squeeze</h4>
            <p class="{'status-squeezed' if is_sq else 'status-normal'}">
                {'🚨 SQUEEZED' if is_sq else 'NORMAL'}
            </p>
            <p style='font-size:0.75rem; color:#94a3b8;'>BB Width: {current_w:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.metric("Consolidation Range High", f"₹{obs_high:,.2f}" if obs_high > 0 else "N/A")
    with col3:
        st.metric("Consolidation Range Low", f"₹{obs_low:,.2f}" if obs_low > 0 else "N/A")
        
    # Chart the range & spot price path
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_1m["timestamp"], y=df_1m["close"], mode="lines", name="Spot Price", line=dict(color="#d946ef")))
    
    if obs_high > 0 and obs_low > 0:
        fig.add_hline(y=obs_high, line_dash="dash", line_color="#ef4444", annotation_text="Obs High")
        fig.add_hline(y=obs_low, line_dash="dash", line_color="#22c55e", annotation_text="Obs Low")
        
    fig.update_layout(title="Spot Price Path (1-Minute Candles)", template="plotly_dark", height=280, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No spot candles available for the selected date.")


# ── Option Chain Monitor ───────────────────────────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <h3>💵 Expiring Option Chain Snapshots</h3>
    <span class="badge">Cheap Premium Selection (₹5 - ₹25)</span>
</div>
""", unsafe_allow_html=True)

query_oc = text("""
    SELECT timestamp, symbol, strike, option_type, ltp, oi, iv, delta, gamma
    FROM option_chain_data
    WHERE underlying_symbol = :symbol AND date(timestamp) = :td
    ORDER BY timestamp, symbol
""")
with engine.connect() as conn:
    df_oc = pd.read_sql(query_oc, conn, params={"symbol": underlying, "td": target_date.isoformat()})

if not df_oc.empty:
    df_oc["timestamp"] = pd.to_datetime(df_oc["timestamp"], format="mixed")
    timestamps = sorted(df_oc["timestamp"].unique())
    selected_ts = st.select_slider("Select Option Chain Timestamp", options=timestamps, format_func=lambda x: x.strftime("%H:%M:%S"))
    
    df_ts = df_oc[df_oc["timestamp"] == selected_ts]
    
    # Filter to low-premium options
    df_cheap = df_ts[(df_ts["ltp"] >= 5.0) & (df_ts["ltp"] <= 25.0)].copy()
    if not df_cheap.empty:
        df_display = df_cheap[["symbol", "strike", "option_type", "ltp", "oi", "delta", "gamma"]].sort_values("strike")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No options found within the ₹5 - ₹25 premium range for this snapshot.")
else:
    st.info("No option chain snapshots stored for this date.")


# ── Backtest Performance Summary ───────────────────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

backtest_path = Path("reports/gamma_blast/gamma_blast_summary.csv")
trades_path = Path("reports/gamma_blast/gamma_blast_trades.csv")

if backtest_path.exists():
    st.markdown("""
    <div class="section-header">
        <h3>📊 0DTE Backtest Summary</h3>
        <span class="badge">Verified Results</span>
    </div>
    """, unsafe_allow_html=True)

    summary = pd.read_csv(backtest_path)
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

    if not summary.empty:
        total_trades = int(summary["total_trades"].sum())
        win_rate = summary["win_rate"].mean()
        avg_pnl = summary["avg_pnl_pct"].mean()
        total_pnl = summary["total_pnl_pct"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", total_trades)
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Avg P&L / Trade", f"{avg_pnl:+.1f}%")
        m4.metric("Cumulative P&L", f"{total_pnl:+.1f}%")

        st.dataframe(summary, use_container_width=True, hide_index=True)

    if not trades.empty:
        st.markdown("**All Generated Trades Log**")
        st.dataframe(trades, use_container_width=True, hide_index=True)
else:
    st.info("Run the 0DTE backtester to see historical performance: `python -m trade_system.domains.analysis.application.backtesting.gamma_blast_backtest`")
