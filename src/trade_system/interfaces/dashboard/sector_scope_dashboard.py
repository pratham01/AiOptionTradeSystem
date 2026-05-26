import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from pathlib import Path
import json

from trade_system.infrastructure.database.connection import get_engine
from sqlalchemy import text
from trade_system.infrastructure.data.fo_universe import get_sector_mapping, get_stocks_by_sector
from trade_system.application.analysis.breakout_screener import BreakoutScreener

# Set page title and styling
st.markdown("## 🧭 Sector Scope & Intraday Breakout Monitor")
st.caption("TradeFinder-style Sector Leadership, Intraday Volume Surge, and Opening Range Breakout (ORB) Tracker")

@st.cache_data(ttl=30)
def fetch_target_date_and_data():
    engine = get_engine()
    # Find latest date in stock database
    query_date = text("SELECT MAX(date(timestamp)) FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX'")
    with engine.connect() as conn:
        latest_date_str = conn.execute(query_date).scalar()
        
    if not latest_date_str:
        return None, pd.DataFrame()
        
    # Fetch last 10 days of data around latest date to compute volume SMAs and previous closes
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date(:latest_date, '-10 days') AND timestamp <= date(:latest_date, '+1 day')
        ORDER BY symbol, timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"latest_date": latest_date_str})
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return latest_date_str, df

latest_date_str, df = fetch_target_date_and_data()

if latest_date_str is None or df.empty:
    st.warning("No data found in the 15-minute database. Please start the live bot or run a history backfill first.")
else:
    target_date = pd.to_datetime(latest_date_str).date()
    st.info(f"📅 Displaying data for latest available session: **{latest_date_str}**")
    
    # -------------------------------------------------------------
    # 1. Sector Performance Calculations
    # -------------------------------------------------------------
    fo_metadata = get_sector_mapping()
    df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))
    
    today_df = df[df['timestamp'].dt.date == target_date]
    prev_df = df[df['timestamp'].dt.date < target_date]
    
    # Latest stock closes today
    latest_closes = today_df[~today_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()
    # Closes for stocks on previous day
    prev_closes = prev_df[~prev_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()
    
    merged_closes = pd.merge(latest_closes, prev_closes, on=['symbol', 'sector'], suffixes=('_last', '_prev'))
    merged_closes['pChange'] = ((merged_closes['close_last'] - merged_closes['close_prev']) / merged_closes['close_prev']) * 100
    
    # Sector performance
    sector_perf = merged_closes.groupby('sector')['pChange'].mean().reset_index()
    sector_perf = sector_perf[sector_perf['sector'] != 'UNKNOWN']
    sector_perf = sector_perf.sort_values(by='pChange', ascending=False)
    
    # Split into leading and lagging
    leading_sectors = sector_perf.head(3)['sector'].tolist()
    lagging_sectors = sector_perf.tail(3)['sector'].tolist()
    
    # -------------------------------------------------------------
    # 2. Sector Dashboard Layout
    # -------------------------------------------------------------
    col_lead, col_lag = st.columns(2)
    
    with col_lead:
        st.markdown("### 🔥 Top Performing Sectors (LONG Bias)")
        for idx, row in sector_perf.head(3).iterrows():
            st.success(f"**{row['sector']}** | Avg Gain: **{row['pChange']:+.2f}%**")
            
    with col_lag:
        st.markdown("### ❄️ Worst Performing Sectors (SHORT Bias)")
        for idx, row in sector_perf.tail(3).iterrows():
            st.error(f"**{row['sector']}** | Avg Loss: **{row['pChange']:+.2f}%**")
            
    st.markdown("---")
    
    # Sector Leaderboard Plotly Chart
    fig = px.bar(
        sector_perf,
        x="sector",
        y="pChange",
        color="pChange",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="F&O Sector Scope Leaderboard (%)",
        labels={"sector": "Sector", "pChange": "Average Return (%)"}
    )
    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # -------------------------------------------------------------
    # 3. Active Screener Alerts Board
    # -------------------------------------------------------------
    st.markdown("### 🚨 Active Breakout / Breakdown Screener Alerts")
    
    # User Control Panel for High-Probability Filters
    st.caption("Configure High-Probability Strategy Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_vwap = st.checkbox("VWAP Filter (Price > VWAP for Long, < VWAP for Short)", value=True)
    with col2:
        use_wick = st.checkbox("Candle Wick Filter (Solid wicks <= 35%)", value=True)
    with col3:
        use_index = st.checkbox("Index Trend Alignment (Nifty50/NiftyBank correlation)", value=True)
        
    # Scan using BreakoutScreener
    screener = BreakoutScreener()
    alerts = screener.scan_for_breakouts(
        top_sectors_count=3,
        target_date=target_date,
        use_vwap_filter=use_vwap,
        use_wick_filter=use_wick,
        use_index_filter=use_index
    )
    
    if alerts:
        alert_rows = []
        for a in alerts:
            clean_sym = a['symbol'].replace("NSE:", "").replace("-EQ", "")
            vol_ratio = a['volume'] / a['vol_sma'] if a['vol_sma'] > 0 else 0
            
            # Format trigger description
            direction_badge = "🟢 LONG Breakout" if a['direction'] == "LONG" else "🔴 SHORT Breakdown"
            ref_level = a['orb_high'] if a['direction'] == "LONG" else a['orb_low']
            
            alert_rows.append({
                "Time": today_df[today_df['symbol'] == a['symbol']].iloc[-1]['timestamp'].strftime("%H:%M") if not today_df[today_df['symbol'] == a['symbol']].empty else "15:15",
                "Symbol": clean_sym,
                "Sector": a['sector'],
                "Direction": direction_badge,
                "LTP (₹)": f"{a['close']:.2f}",
                "ORB Level (₹)": f"{ref_level:.2f}",
                "Volume Surge": f"{vol_ratio:.1f}x",
                "VWAP (₹)": f"{a['vwap']:.2f}"
            })
            
        st.dataframe(
            pd.DataFrame(alert_rows).set_index("Time"),
            use_container_width=True
        )
    else:
        st.info("No active breakouts or breakdowns detected under the current filter criteria.")
        
    st.markdown("---")
    
    # -------------------------------------------------------------
    # 4. Sector Drill-Down Monitor
    # -------------------------------------------------------------
    st.markdown("### 🔎 Sector Drill-Down (Stock Scope)")
    
    selected_sector = st.selectbox("Select a Sector to monitor members", sorted(sector_perf['sector'].unique()))
    
    if selected_sector:
        sector_symbols = get_stocks_by_sector(selected_sector)
        st.markdown(f"Displaying F&O members for sector: **{selected_sector}** (Total: {len(sector_symbols)} stocks)")
        
        stock_rows = []
        for symbol in sector_symbols:
            stock_all = df[df['symbol'] == symbol].sort_values('timestamp')
            if len(stock_all) < 20:
                continue
                
            # 20 SMA Volume
            stock_all['vol_sma_20'] = stock_all['volume'].rolling(window=20).mean()
            
            # Slice today's candles from stock_all
            stock_today = stock_all[stock_all['timestamp'].dt.date == target_date]
            if stock_today.empty:
                continue
                
            # Compute indicators for this stock today
            latest_row = stock_today.iloc[-1]
            orb_candles = stock_today.head(4)
            orb_high = orb_candles['high'].max()
            orb_low = orb_candles['low'].min()
            
            prev_vol_sma = stock_today.iloc[-2]['vol_sma_20'] if len(stock_today) > 1 else stock_all.iloc[-2]['vol_sma_20']
            vol_ratio = latest_row['volume'] / prev_vol_sma if prev_vol_sma > 0 else 0
            
            # Dynamic VWAP
            cum_pv = (stock_today['close'] * stock_today['volume']).cumsum()
            cum_vol = stock_today['volume'].cumsum()
            vwap = cum_pv.iloc[-1] / cum_vol.iloc[-1]
            
            # Check Breakout Status
            status = "Rangebound"
            if latest_row['close'] > orb_high:
                status = "🟢 Breakout"
            elif latest_row['close'] < orb_low:
                status = "🔴 Breakdown"
                
            # Find percentage change vs prev close
            sym_change = merged_closes[merged_closes['symbol'] == symbol]
            p_change = sym_change.iloc[0]['pChange'] if not sym_change.empty else 0.0
            
            stock_rows.append({
                "Symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                "LTP (₹)": latest_row['close'],
                "Change %": p_change,
                "Volume Surge": vol_ratio,
                "VWAP (₹)": vwap,
                "ORB High (₹)": orb_high,
                "ORB Low (₹)": orb_low,
                "Status": status
            })
            
        if stock_rows:
            df_stocks = pd.DataFrame(stock_rows).sort_values(by="Change %", ascending=False)
            
            # Format and display
            st.dataframe(
                df_stocks.style.format({
                    "LTP (₹)": "{:.2f}",
                    "Change %": "{:+.2f}%",
                    "Volume Surge": "{:.1f}x",
                    "VWAP (₹)": "{:.2f}",
                    "ORB High (₹)": "{:.2f}",
                    "ORB Low (₹)": "{:.2f}"
                }),
                use_container_width=True
            )
        else:
            st.caption("No stock data available for the members of this sector today.")
