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

@st.cache_data(ttl=300)
def fetch_available_dates():
    engine = get_engine()
    query = text("SELECT DISTINCT date(timestamp) as d FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX' ORDER BY d DESC")
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return [row[0] for row in result if row[0] is not None]
    except Exception as e:
        return []

@st.cache_data(ttl=30)
def fetch_target_date_and_data(selected_date_str=None):
    engine = get_engine()
    if selected_date_str is None:
        # Find latest date in stock database
        query_date = text("SELECT MAX(date(timestamp)) FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX'")
        with engine.connect() as conn:
            selected_date_str = conn.execute(query_date).scalar()
            
    if not selected_date_str:
        return None, pd.DataFrame(), None
        
    query_max_ts = text("SELECT MAX(timestamp) FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX' AND date(timestamp) = :sel_date")
    with engine.connect() as conn:
        last_candle_ts = conn.execute(query_max_ts, {"sel_date": selected_date_str}).scalar()
        
    # Fetch last 10 days of data around latest date to compute volume SMAs and previous closes
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date(:latest_date, '-10 days') AND timestamp <= date(:latest_date, '+1 day')
        ORDER BY symbol, timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"latest_date": selected_date_str})
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    return selected_date_str, df, last_candle_ts

@st.cache_data(ttl=30)
def fetch_live_quotes(symbols):
    try:
        from trade_system.infrastructure.brokers.factory import get_broker_manager
        from trade_system.config import Settings
        settings = Settings.load()
        manager = get_broker_manager(settings)
        return manager.get_quotes(symbols)
    except Exception as e:
        # Re-raise so the caller's try-except block can handle it
        raise e

available_dates = fetch_available_dates()

# Inject today's date if it is a weekday and current local time is after 9:00 AM (premarket start)
today_str = date.today().strftime("%Y-%m-%d")
is_weekday = datetime.today().weekday() < 5
is_after_nine = datetime.now().time() >= datetime.strptime("09:00:00", "%H:%M:%S").time()

if is_weekday and is_after_nine:
    if today_str not in available_dates:
        available_dates.insert(0, today_str)

selected_date_str = None
if available_dates:
    selected_date_str = st.sidebar.selectbox(
        "📅 Analysis Date",
        options=available_dates,
        index=0,
        help="Select a trading date to analyze post-market / historical data"
    )

selected_lookback = st.sidebar.radio(
    "⏱️ Performance Lookback",
    options=["Daily (Today)", "Last 15 Mins", "Last 1 Hour", "Last 2 Hours"],
    index=0,
    help="Select the lookback interval for sector return calculations"
)

latest_date_str, df, last_candle_ts = fetch_target_date_and_data(selected_date_str)

if latest_date_str is None or df.empty:
    st.warning("No data found in the 15-minute database. Please start the live bot or run a history backfill first.")
else:
    target_date = pd.to_datetime(latest_date_str).date()
    
    # Resolve sector info and check database dates for fallback
    fo_metadata = get_sector_mapping()
    df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))
    
    unique_db_dates = sorted(list(df['timestamp'].dt.date.unique()))
    db_target_date = target_date
    if target_date not in unique_db_dates and unique_db_dates:
        db_target_date = unique_db_dates[-1]

    # Fetch live quotes if selected date is today
    is_today = (target_date == date.today())
    quotes = {}
    if is_today:
        try:
            # Fetch quotes for all symbols in the F&O universe using cached helper
            symbols = list(fo_metadata.keys())
            quotes = fetch_live_quotes(symbols)
        except Exception as e:
            st.sidebar.warning(f"Could not fetch live quotes: {e}")

    # Fallback logic if we are on "today" but have no quotes (due to API failure/429)
    # and today's date has no candles in the database yet
    if is_today and not quotes and target_date not in unique_db_dates:
        st.sidebar.warning(f"⚠️ Live quotes unavailable. Showing data as of database date: {db_target_date}")
        target_date = db_target_date
        latest_date_str = db_target_date.strftime("%Y-%m-%d")
        is_today = False
        
    db_update_time = pd.to_datetime(last_candle_ts).strftime("%Y-%m-%d %H:%M:%S") if last_candle_ts else "N/A"
    refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.info(
        f"📅 **Session Date:** {latest_date_str} | "
        f"🕒 **Latest DB Candle:** {db_update_time} | "
        f"🔄 **Refreshed:** {refresh_time}"
    )
    
    # -------------------------------------------------------------
    # 1. Sector Performance Calculations
    # -------------------------------------------------------------
    # Determine lookback parameters
    if selected_lookback == "Last 15 Mins":
        step = 1
    elif selected_lookback == "Last 1 Hour":
        step = 4
    elif selected_lookback == "Last 2 Hours":
        step = 8
    else:
        step = None  # Daily return (Today)

    # Filter data up to target_date
    df_filtered = df[df['timestamp'].dt.date <= target_date]
    rows = []
    
    # Calculate performance for each symbol based on lookback
    grouped = df_filtered[~df_filtered['symbol'].str.contains("INDEX")].groupby('symbol')
    
    for symbol, grp in grouped:
        grp_sorted = grp.sort_values('timestamp')
        if grp_sorted.empty:
            continue
            
        sector = fo_metadata.get(symbol, "UNKNOWN")
        
        # Latest price
        latest_row = grp_sorted.iloc[-1]
        close_last = float(latest_row['close'])
        
        # Override with live quote LTP if applicable
        if is_today and quotes and symbol in quotes:
            quote = quotes[symbol]
            close_last = quote.last_price or quote.close or quote.open
            
        # Reference price
        if step is not None:
            if len(grp_sorted) > step:
                # Compare to the candle step periods ago
                ref_idx = -step
                close_prev = float(grp_sorted.iloc[ref_idx]['close'])
            else:
                close_prev = float(grp_sorted.iloc[0]['close'])
        else:
            # Daily return: compare to previous day's close
            prev_candles = grp_sorted[grp_sorted['timestamp'].dt.date < target_date]
            if not prev_candles.empty:
                close_prev = float(prev_candles.iloc[-1]['close'])
            elif is_today and quotes and symbol in quotes:
                close_prev = quotes[symbol].previous_close
            else:
                close_prev = float(grp_sorted.iloc[0]['close'])
                
        if close_prev > 0 and close_last > 0:
            pchange = ((close_last - close_prev) / close_prev) * 100
            rows.append({
                "symbol": symbol,
                "sector": sector,
                "close_last": close_last,
                "close_prev": close_prev,
                "pChange": pchange
            })
            
    if rows:
        merged_closes = pd.DataFrame(rows)
        # Ensure any new quotes symbols not in database are added for Today view
        if is_today and quotes:
            live_symbols_to_add = []
            for symbol, quote in quotes.items():
                if symbol not in merged_closes['symbol'].values:
                    sector = fo_metadata.get(symbol, "UNKNOWN")
                    ltp = quote.last_price or quote.close or quote.open
                    prev_close = quote.previous_close
                    if prev_close > 0 and ltp > 0:
                        pchange = ((ltp - prev_close) / prev_close) * 100
                        live_symbols_to_add.append({
                            "symbol": symbol,
                            "sector": sector,
                            "close_last": ltp,
                            "close_prev": prev_close,
                            "pChange": pchange
                        })
            if live_symbols_to_add:
                merged_closes = pd.concat([merged_closes, pd.DataFrame(live_symbols_to_add)], ignore_index=True)
            st.sidebar.success(f"⚡ Loaded {len(merged_closes)} live quotes (inc. premarket)")
    else:
        merged_closes = pd.DataFrame(columns=["symbol", "sector", "close_last", "close_prev", "pChange"])
    
    # Sector performance
    if not merged_closes.empty:
        sector_perf = merged_closes.groupby('sector')['pChange'].mean().reset_index()
    else:
        sector_perf = pd.DataFrame(columns=['sector', 'pChange'])
        
    sector_perf = sector_perf[sector_perf['sector'] != 'UNKNOWN']
    sector_perf = sector_perf.sort_values(by='pChange', ascending=False)
    
    options = sorted(sector_perf['sector'].unique())
    
    # Split into leading and lagging
    leading_sectors = sector_perf.head(3)['sector'].tolist()
    lagging_sectors = sector_perf.tail(3)['sector'].tolist()

    # Pre-calculate global leaderboards
    top_gainers = merged_closes.sort_values(by='pChange', ascending=False).head(5)
    top_losers = merged_closes.sort_values(by='pChange', ascending=True).head(5)
    
    gainers_data = []
    for idx, row in top_gainers.iterrows():
        sym_clean = row['symbol'].replace("NSE:", "").replace("-EQ", "")
        gainers_data.append({
            "Symbol": sym_clean,
            "Sector": row['sector'],
            "LTP (₹)": f"{row['close_last']:.2f}",
            "Change %": f"+{row['pChange']:.2f}%"
        })
        
    losers_data = []
    for idx, row in top_losers.iterrows():
        sym_clean = row['symbol'].replace("NSE:", "").replace("-EQ", "")
        losers_data.append({
            "Symbol": sym_clean,
            "Sector": row['sector'],
            "LTP (₹)": f"{row['close_last']:.2f}",
            "Change %": f"{row['pChange']:.2f}%"
        })

    # -------------------------------------------------------------
    # 2. Session State Initialization & Interactivity Event Handlers
    # -------------------------------------------------------------
    if 'selected_sector_drill' not in st.session_state:
        st.session_state['selected_sector_drill'] = options[0] if options else None

    # Track last processed selections to prevent selection feedback loops when user manually changes the selectbox
    if 'last_plotly_selection' not in st.session_state:
        st.session_state['last_plotly_selection'] = None
    if 'last_gainer_selection' not in st.session_state:
        st.session_state['last_gainer_selection'] = None
    if 'last_loser_selection' not in st.session_state:
        st.session_state['last_loser_selection'] = None

    selected_sector_plotly = None
    if "plotly_sector_chart" in st.session_state and st.session_state["plotly_sector_chart"]:
        select_event = st.session_state["plotly_sector_chart"]
        if isinstance(select_event, dict) and "selection" in select_event:
            selection = select_event["selection"]
            if "points" in selection and selection["points"]:
                point = selection["points"][0]
                selected_sector_plotly = point.get("x")

    selected_sector_gainer = None
    if "gainer_leaderboard" in st.session_state and st.session_state["gainer_leaderboard"]:
        g_selection = st.session_state["gainer_leaderboard"].get("selection", {})
        g_rows = g_selection.get("rows", [])
        if g_rows:
            g_idx = g_rows[0]
            if 0 <= g_idx < len(gainers_data):
                selected_sector_gainer = gainers_data[g_idx]["Sector"]

    selected_sector_loser = None
    if "loser_leaderboard" in st.session_state and st.session_state["loser_leaderboard"]:
        l_selection = st.session_state["loser_leaderboard"].get("selection", {})
        l_rows = l_selection.get("rows", [])
        if l_rows:
            l_idx = l_rows[0]
            if 0 <= l_idx < len(losers_data):
                selected_sector_loser = losers_data[l_idx]["Sector"]

    # Check for new selection events and update selected_sector_drill
    target_sector = None
    if selected_sector_plotly != st.session_state['last_plotly_selection']:
        st.session_state['last_plotly_selection'] = selected_sector_plotly
        if selected_sector_plotly in options:
            target_sector = selected_sector_plotly

    elif selected_sector_gainer != st.session_state['last_gainer_selection']:
        st.session_state['last_gainer_selection'] = selected_sector_gainer
        if selected_sector_gainer in options:
            target_sector = selected_sector_gainer

    elif selected_sector_loser != st.session_state['last_loser_selection']:
        st.session_state['last_loser_selection'] = selected_sector_loser
        if selected_sector_loser in options:
            target_sector = selected_sector_loser

    if target_sector:
        st.session_state['selected_sector_drill'] = target_sector

    # -------------------------------------------------------------
    # 3. Market Overview Dashboard Leaderboard (Sectors & Stocks)
    # -------------------------------------------------------------
    st.markdown("### 🏆 Market Overview & Leaderboards")
    st.caption("💡 Click any sector button or stock row to instantly drill down into members")
    
    col_lead, col_lag, col_gainers, col_losers = st.columns(4)
    
    with col_lead:
        st.markdown("#### 🔥 Top 3 Sectors")
        for idx, row in sector_perf.head(3).iterrows():
            if st.button(f"🟢 {row['sector']} ({row['pChange']:+.2f}%)", key=f"btn_lead_{row['sector']}", use_container_width=True):
                st.session_state['selected_sector_drill'] = row['sector']
                st.rerun()
            
    with col_lag:
        st.markdown("#### ❄️ Worst 3 Sectors")
        for idx, row in sector_perf.tail(3).iloc[::-1].iterrows():
            if st.button(f"🔴 {row['sector']} ({row['pChange']:+.2f}%)", key=f"btn_lag_{row['sector']}", use_container_width=True):
                st.session_state['selected_sector_drill'] = row['sector']
                st.rerun()

    with col_gainers:
        st.markdown("#### 🟢 Top 5 Gainers")
        st.dataframe(
            pd.DataFrame(gainers_data),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="gainer_leaderboard"
        )
        
    with col_losers:
        st.markdown("#### 🔴 Top 5 Losers")
        st.dataframe(
            pd.DataFrame(losers_data),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="loser_leaderboard"
        )
            
    st.markdown("---")
    
    # Sector Leaderboard Plotly Chart
    fig = px.bar(
        sector_perf,
        x="sector",
        y="pChange",
        color="pChange",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="F&O Sector Scope Leaderboard (%) (Click bar to drill down)",
        labels={"sector": "Sector", "pChange": "Average Return (%)"}
    )
    fig.update_layout(height=400, template="plotly_dark")
    
    # Render Plotly chart interactively if on_select is supported
    import inspect
    sig = inspect.signature(st.plotly_chart)
    if "on_select" in sig.parameters:
        st.plotly_chart(fig, use_container_width=True, key="plotly_sector_chart", on_select="rerun")
    else:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # -------------------------------------------------------------
    # 3. Active Screener Alerts Board
    # -------------------------------------------------------------
    st.markdown("### 🚨 Active Breakout / Breakdown Screener Alerts")
    
    # User Control Panel for High-Probability Filters
    st.caption("Configure High-Probability Strategy Filters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        use_sector = st.checkbox("Sector Leadership Filter", value=False)
    with col2:
        use_vwap = st.checkbox("VWAP Filter", value=True, help="Price > VWAP for Long, < VWAP for Short")
    with col3:
        use_wick = st.checkbox("Candle Wick Filter", value=True, help="Solid wicks <= 35%")
    with col4:
        use_index = st.checkbox("Index Alignment", value=True, help="Nifty50/NiftyBank correlation")
    with col5:
        vol_surge = st.selectbox(
            "Min Vol Surge",
            options=[1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
            index=2, # defaults to 1.5
            format_func=lambda x: f"{x}x",
            help="Minimum volume relative to the 20-period 15-minute Volume SMA"
        )

    # Second row of control panel for daily indicators & filters
    st.caption("Daily and Volume Filters")
    col1b, col2b, col3b, col4b = st.columns(4)
    with col1b:
        use_52w = st.checkbox("52-Week High Filter", value=False, help="Filter for breakout signals trading at/near 52W high")
    with col2b:
        use_weekly = st.checkbox("Weekly High Filter", value=False, help="Filter for breakout signals trading at/near weekly high")
    with col3b:
        use_daily_st_touch = st.checkbox("Daily ST Touch Filter", value=False, help="Filter for breakout signals overlapping their daily Supertrend")
    with col4b:
        use_abnormal_vol = st.checkbox("Abnormal Vol Filter", value=False, help="Enforce the Min Vol Surge ratio on all breakout signals")
        
    # Scan using BreakoutScreener with error handling
    screener = BreakoutScreener()
    try:
        alerts = screener.scan_for_breakouts(
            top_sectors_count=3,
            target_date=target_date,
            use_vwap_filter=use_vwap,
            use_wick_filter=use_wick,
            use_index_filter=use_index,
            use_sector_filter=use_sector,
            vol_surge_threshold=float(vol_surge),
            use_52w_filter=use_52w,
            use_weekly_high_filter=use_weekly,
            use_daily_st_touch_filter=use_daily_st_touch,
            use_abnormal_vol_filter=use_abnormal_vol
        )
    except Exception as e:
        st.error(f"Error running BreakoutScreener: {e}")
        alerts = []
    
    if alerts:
        alert_rows = []
        for a in alerts:
            clean_sym = a['symbol'].replace("NSE:", "").replace("-EQ", "")
            vol_ratio = a['volume'] / a['vol_sma'] if a['vol_sma'] > 0 else 0
            
            # Format trigger description
            direction_badge = "🟢 LONG Breakout" if a['direction'] == "LONG" else "🔴 SHORT Breakdown"
            ref_level = a['orb_high'] if a['direction'] == "LONG" else a['orb_low']
            
            # Alert type badge
            alert_type = a.get('alert_type', 'ORB_BREAKOUT')
            type_badges = {
                "ORB_BREAKOUT": "📊 ORB",
                "MOMENTUM_MOVER": "🚀 Momentum",
                "PREV_DAY_HIGH": "⬆️ Prev Day High",
                "PREV_DAY_LOW": "⬇️ Prev Day Low"
            }
            type_badge = type_badges.get(alert_type, alert_type)
            
            # Confluence tags
            tags = []
            if a.get('is_52w_high'): tags.append("52W High 🔥")
            if a.get('is_weekly_high'): tags.append("Wk High 📈")
            if a.get('touches_daily_st'): tags.append("ST Touch 🎯")
            tag_str = ", ".join(tags) if tags else "Normal"
            
            trigger_time_str = a['trigger_time'].strftime("%H:%M") if 'trigger_time' in a and pd.notna(a['trigger_time']) else "15:15"
            alert_rows.append({
                "Time": trigger_time_str,
                "Symbol": clean_sym,
                "Sector": a['sector'],
                "Type": type_badge,
                "Direction": direction_badge,
                "Change %": f"{a.get('pchange', 0):+.2f}%",
                "LTP (₹)": f"{a['close']:.2f}",
                "ORB Level (₹)": f"{ref_level:.2f}",
                "Volume Surge": f"{vol_ratio:.1f}x",
                "VWAP (₹)": f"{a['vwap']:.2f}",
                "Daily Filters": tag_str
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
    
    selected_sector = st.selectbox(
        "Select a Sector to monitor members",
        options,
        key="selected_sector_drill"
    )
    
    if selected_sector:
        sector_symbols = get_stocks_by_sector(selected_sector)
        st.markdown(f"Displaying F&O members for sector: **{selected_sector}** (Total: {len(sector_symbols)} stocks)")
        
        # Sector drill-down filtering checkboxes
        st.caption("Apply Drill-Down Filters:")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            f_52w = st.checkbox("52W High Breakout Only", value=False, key=f"f_52w_{selected_sector}")
        with col_f2:
            f_weekly = st.checkbox("Weekly High Breakout Only", value=False, key=f"f_weekly_{selected_sector}")
        with col_f3:
            f_st = st.checkbox("Daily ST Touch Only", value=False, key=f"f_st_{selected_sector}")
        with col_f4:
            f_abnormal = st.checkbox("Abnormal Vol Only", value=False, key=f"f_abnormal_{selected_sector}", help=f"Volume Surge >= {vol_surge}x")
            
        # Pre-fetch daily data for all sector symbols to keep it fast
        daily_groups = {}
        if sector_symbols:
            engine = get_engine()
            # Format symbols safely in SQL list to bypass sqlite3 list parameter limitation
            sym_list_str = ", ".join(f"'{s}'" for s in sector_symbols)
            query_daily = text(f"""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM ohlcv_daily 
                WHERE symbol IN ({sym_list_str}) AND timestamp >= date(:target_date, '-365 days') AND timestamp <= :target_date
                ORDER BY symbol, timestamp ASC
            """)
            try:
                with engine.connect() as conn:
                    df_daily = pd.read_sql(query_daily, conn, params={"target_date": target_date.isoformat()})
                df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], format='mixed')
                daily_groups = {sym: grp.sort_values('timestamp') for sym, grp in df_daily.groupby('symbol')}
            except Exception as e:
                st.warning(f"Failed to fetch daily candles for sector: {e}")

        stock_rows = []
        for symbol in sector_symbols:
            stock_all = df[df['symbol'] == symbol].sort_values('timestamp')
            if len(stock_all) < 20:
                continue
                
            # 20 SMA Volume
            stock_all['vol_sma_20'] = stock_all['volume'].rolling(window=20).mean()
            
            # Slice today's candles from stock_all
            stock_today = stock_all[stock_all['timestamp'].dt.date == target_date]
            if stock_today.empty and not is_today:
                continue
                
            # Compute indicators for this stock today
            latest_row = stock_today.iloc[-1] if not stock_today.empty else None
            
            # If we don't have stock_today but it is today, mock a row using the quote
            if latest_row is None and is_today and symbol in quotes:
                quote = quotes[symbol]
                ltp = quote.last_price or quote.close or quote.open
                vol = quote.volume
                latest_row = pd.Series({"close": ltp, "volume": vol})
                
            if latest_row is None:
                continue
                
            orb_candles = stock_today.head(4)
            orb_high = orb_candles['high'].max() if not orb_candles.empty else latest_row['close']
            orb_low = orb_candles['low'].min() if not orb_candles.empty else latest_row['close']
            
            prev_vol_sma = stock_today.iloc[-2]['vol_sma_20'] if len(stock_today) > 1 else stock_all.iloc[-2]['vol_sma_20']
            vol_ratio = latest_row['volume'] / prev_vol_sma if prev_vol_sma > 0 else 0
            
            # Dynamic VWAP
            if not stock_today.empty:
                cum_pv = (stock_today['close'] * stock_today['volume']).cumsum()
                cum_vol = stock_today['volume'].cumsum()
                vwap = cum_pv.iloc[-1] / cum_vol.iloc[-1]
            else:
                vwap = latest_row['close']
            
            # Check Breakout Status
            status = "Rangebound"
            if latest_row['close'] > orb_high:
                status = "🟢 Breakout"
            elif latest_row['close'] < orb_low:
                status = "🔴 Breakdown"
                
            # Find percentage change vs prev close
            sym_change = merged_closes[merged_closes['symbol'] == symbol]
            p_change = sym_change.iloc[0]['pChange'] if not sym_change.empty else 0.0
            
            # Daily Technical Calculations
            grp_d = daily_groups.get(symbol)
            is_52w_high = False
            is_weekly_high = False
            touches_daily_st = False
            
            if grp_d is not None and not grp_d.empty:
                latest_close = float(latest_row['close'])
                
                # 52-Week High (max high of last 250 daily bars excluding the last one)
                daily_highs = grp_d['high'].iloc[:-1] if len(grp_d) > 1 else grp_d['high']
                if not daily_highs.empty:
                    fifty_two_week_high = daily_highs.max()
                    is_52w_high = latest_close >= (fifty_two_week_high * 0.995)
                
                # Weekly High (max high of last week's daily bars, i.e. past 5 days excluding today)
                weekly_highs = grp_d['high'].iloc[-6:-1] if len(grp_d) > 5 else grp_d['high']
                if not weekly_highs.empty:
                    weekly_high = weekly_highs.max()
                    is_weekly_high = latest_close >= (weekly_high * 0.995)
                
                # Daily Supertrend Touch
                try:
                    from trade_system.application.indicators import calculate_supertrend
                    st_d = calculate_supertrend(grp_d)
                    if not st_d.empty:
                        last_st_d = st_d.iloc[-1]
                        st_val = float(last_st_d['supertrend'])
                        touches_daily_st = float(last_st_d['low']) <= st_val <= float(last_st_d['high'])
                except Exception as ex:
                    pass
            
            # Apply Filter Checkbox logic to Sector Drill Down
            if f_52w and not is_52w_high:
                continue
            if f_weekly and not is_weekly_high:
                continue
            if f_st and not touches_daily_st:
                continue
            if f_abnormal and vol_ratio < float(vol_surge):
                continue

            stock_rows.append({
                "Symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                "LTP (₹)": latest_row['close'],
                "Change %": p_change,
                "Volume Surge": vol_ratio,
                "VWAP (₹)": vwap,
                "ORB High (₹)": orb_high,
                "ORB Low (₹)": orb_low,
                "Status": status,
                "52W High": "🔥 52W" if is_52w_high else "No",
                "Weekly High": "📈 Weekly" if is_weekly_high else "No",
                "ST Touch": "🎯 ST Touch" if touches_daily_st else "No"
            })
            
        if stock_rows:
            df_stocks = pd.DataFrame(stock_rows).sort_values(by="Change %", ascending=False)
            
            # Format and display all members
            st.dataframe(
                df_stocks.style.format({
                    "LTP (₹)": "{:.2f}",
                    "Change %": "{:+.2f}%",
                    "Volume Surge": "{:.1f}x",
                    "VWAP (₹)": "{:.2f}",
                    "ORB High (₹)": "{:.2f}",
                    "ORB Low (₹)": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("No stock data available for the members of this sector today matching the filters.")
