import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import logging
from sqlalchemy import text

# Import our new analyzer
from trade_system.domains.analysis.application.analysis.greeks_exposure_analyzer import compute_greeks_exposure
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)

# --- CACHED DATA LOADING ---
@st.cache_data(ttl=60)
def load_latest_option_chain_with_greeks(symbol: str, target_date: str) -> pd.DataFrame:
    """Fetch the absolute latest option chain snapshot for the given date that has greeks."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Find the latest timestamp for this date
            start_time = f"{target_date} 00:00:00"
            end_time = f"{target_date} 23:59:59"
            
            ts_query = text("""
                SELECT MAX(timestamp) as latest_ts 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                  AND timestamp >= :start_time
                  AND timestamp <= :end_time
            """)
            ts_res = pd.read_sql(ts_query, conn, params={"symbol": symbol, "start_time": start_time, "end_time": end_time})
            
            if ts_res.empty or pd.isna(ts_res.iloc[0]["latest_ts"]):
                return pd.DataFrame()
                
            latest_ts = ts_res.iloc[0]["latest_ts"]
            
            # Fetch the actual data
            data_query = text("""
                SELECT * 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                  AND timestamp = :ts
            """)
            df = pd.read_sql(data_query, conn, params={"symbol": symbol, "ts": latest_ts})
            return df
    except Exception as e:
        LOGGER.error(f"Error loading option chain for GEX: {e}")
        return pd.DataFrame()

def render_gex_chart(exposure_by_strike: pd.DataFrame, spot_price: float, zero_gamma: float):
    """Render the classic DEXT/SpotGamma style GEX bar chart."""
    # Filter to strikes around spot price for better visibility (± 20 strikes)
    # Estimate strike step
    strikes = sorted(exposure_by_strike["strike"].unique())
    strike_step = strikes[1] - strikes[0] if len(strikes) > 1 else 50
    
    min_strike = spot_price - (20 * strike_step)
    max_strike = spot_price + (20 * strike_step)
    
    plot_df = exposure_by_strike[(exposure_by_strike["strike"] >= min_strike) & (exposure_by_strike["strike"] <= max_strike)]
    
    # We need to split into CE GEX and PE GEX for the grouped bar chart.
    # But exposure_by_strike has aggregated 'gex'. Let's just plot the Net GEX per strike, colored by sign.
    
    fig = go.Figure()
    
    # Positive GEX (Call dominated)
    pos_df = plot_df[plot_df["gex"] > 0]
    fig.add_trace(go.Bar(
        x=pos_df["strike"],
        y=pos_df["gex"],
        name="Positive GEX (Calls)",
        marker_color="#00d084",
        opacity=0.8
    ))
    
    # Negative GEX (Put dominated)
    neg_df = plot_df[plot_df["gex"] < 0]
    fig.add_trace(go.Bar(
        x=neg_df["strike"],
        y=neg_df["gex"],
        name="Negative GEX (Puts)",
        marker_color="#ff4d6d",
        opacity=0.8
    ))
    
    # Add Spot Price Line
    fig.add_vline(x=spot_price, line_dash="dash", line_color="#ffffff", annotation_text="Spot Price", annotation_position="top right")
    
    # Add Zero Gamma Line
    if zero_gamma > 0:
        fig.add_vline(x=zero_gamma, line_dash="dot", line_color="#ffb703", annotation_text="Zero Gamma", annotation_position="bottom right")

    fig.update_layout(
        title="Absolute Gamma Exposure (GEX) Profile",
        xaxis_title="Strike Price",
        yaxis_title="Total GEX",
        barmode="relative",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_dex_chart(exposure_by_strike: pd.DataFrame, spot_price: float):
    """Render the Delta Exposure chart."""
    strikes = sorted(exposure_by_strike["strike"].unique())
    strike_step = strikes[1] - strikes[0] if len(strikes) > 1 else 50
    min_strike = spot_price - (20 * strike_step)
    max_strike = spot_price + (20 * strike_step)
    plot_df = exposure_by_strike[(exposure_by_strike["strike"] >= min_strike) & (exposure_by_strike["strike"] <= max_strike)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=plot_df["strike"],
        y=plot_df["dex"],
        name="Net Delta Exposure",
        marker_color="#3b82f6",
        opacity=0.8
    ))
    
    fig.add_vline(x=spot_price, line_dash="dash", line_color="#ffffff", annotation_text="Spot Price")

    fig.update_layout(
        title="Net Delta Exposure (DEX) Profile",
        xaxis_title="Strike Price",
        yaxis_title="Total DEX",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🧲 Greeks Exposure Engine (GEX & DEX)")
    st.caption("Institutional dealer hedging models. Identifies Gamma Walls, Zero Gamma Flips, and Market Regimes.")
    
    # --- Sidebar Controls ---
    st.sidebar.header("Filter Settings")
    symbol_map = {
        "Nifty 50": "NSE:NIFTY50-INDEX",
        "Bank Nifty": "NSE:NIFTYBANK-INDEX",
        "Sensex": "BSE:SENSEX-INDEX"
    }
    selected_symbol_label = st.sidebar.selectbox("Select Index", list(symbol_map.keys()))
    db_symbol = symbol_map[selected_symbol_label]
    selected_date = st.sidebar.date_input("Select Date", date.today()).strftime("%Y-%m-%d")
    
    # Load Data
    df = load_latest_option_chain_with_greeks(db_symbol, selected_date)
    
    if df.empty:
        st.warning(f"No options data found with Greeks for {selected_symbol_label} on {selected_date}.")
        return

    # Derive Spot Price
    # Use average of ATM strikes, or simply assume we can derive it from the deepest ITM or just take median
    pe_df = df[df["option_type"] == "PE"]
    ce_df = df[df["option_type"] == "CE"]
    if not pe_df.empty and not ce_df.empty:
        # Simple approximation of spot price based on ATM straddle crossover
        spot_price = df["strike"].mean() # Fallback
        # Actually better: where CE and PE LTP are closest
        merged = pd.merge(ce_df, pe_df, on="strike", suffixes=("_ce", "_pe"))
        merged["diff"] = (merged["ltp_ce"] - merged["ltp_pe"]).abs()
        spot_price = float(merged.sort_values("diff").iloc[0]["strike"])
    else:
        spot_price = df["strike"].mean()
        
    st.sidebar.metric("Estimated Spot Price", f"₹{spot_price:,.2f}")

    # Compute Exposure
    analysis = compute_greeks_exposure(df, spot_price)
    if not analysis:
        st.error("Failed to compute Greeks exposure. Ensure delta and gamma columns exist and are valid.")
        return

    # --- Dashboard Layout ---
    
    # 1. Top Level Metrics & Regime
    st.markdown(f"### Current Market Regime: **{analysis['regime']}**")
    st.info(analysis["regime_action"])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total GEX (Gamma)", f"{analysis['total_gex']:,.0f}")
    m2.metric("Total DEX (Delta)", f"{analysis['total_dex']:,.0f}")
    m3.metric("Zero Gamma Flip", f"₹{analysis['zero_gamma_strike']:,.0f}")
    m4.metric("Spot to Zero Gamma", f"₹{spot_price - analysis['zero_gamma_strike']:,.0f}")
    
    # 2. Gamma Walls
    w1, w2 = st.columns(2)
    with w1:
        st.markdown(f"""
        <div class="card" style="border-top: 4px solid #00d084; padding: 15px; background: #1e2130; border-radius: 10px;">
            <h4 style="margin:0; color:#00d084;">Call Gamma Wall (Resistance)</h4>
            <h2 style="margin:0;">₹{analysis['call_gamma_wall']:,.0f}</h2>
            <p style="margin:0; color:#8b949e; font-size:14px;">If Spot > this level, expect extreme volatility.</p>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown(f"""
        <div class="card" style="border-top: 4px solid #ff4d6d; padding: 15px; background: #1e2130; border-radius: 10px;">
            <h4 style="margin:0; color:#ff4d6d;">Put Gamma Wall (Support)</h4>
            <h2 style="margin:0;">₹{analysis['put_gamma_wall']:,.0f}</h2>
            <p style="margin:0; color:#8b949e; font-size:14px;">If Spot < this level, dealers will aggressively short.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Strategy Signals
    if analysis["signals"]:
        st.subheader("🔥 Strategy Engine Signals")
        for sig in analysis["signals"]:
            st.success(sig)
            
    st.markdown("---")
    
    # 4. Charts
    tab_gex, tab_dex = st.tabs(["📊 Gamma Exposure (GEX)", "🌊 Delta Exposure (DEX)"])
    
    with tab_gex:
        render_gex_chart(analysis["exposure_by_strike"], spot_price, analysis["zero_gamma_strike"])
        
    with tab_dex:
        render_dex_chart(analysis["exposure_by_strike"], spot_price)

if __name__ == "__main__":
    # Standard Streamlit execution wrapper
    main()
