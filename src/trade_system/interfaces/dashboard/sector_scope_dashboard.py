import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, time as dt_time
from pathlib import Path
import json
from trade_system.interfaces.dashboard.shared_broker import fetch_live_quotes

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from sqlalchemy import text
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping, get_stocks_by_sector
from trade_system.domains.analysis.application.analysis.breakout_screener import BreakoutScreener
import importlib
import trade_system.domains.analysis.application.analysis.intraday_edge_scorer
import trade_system.domains.analysis.application.analysis.smart_entry_trigger
importlib.reload(trade_system.domains.analysis.application.analysis.intraday_edge_scorer)
importlib.reload(trade_system.domains.analysis.application.analysis.smart_entry_trigger)

from trade_system.domains.analysis.application.analysis.intraday_edge_scorer import IntradayEdgeScorer
from trade_system.domains.analysis.application.analysis.smart_entry_trigger import SmartEntryTrigger
from trade_system.shared.notifications.telegram import TelegramNotifier
from trade_system.shared.config import Settings

# --- COMPACT CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; }

    /* Compact Hero */
    .sector-hero-compact {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
        padding: 0.7rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border-left: 5px solid #7c3aed;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .sector-hero-compact h3 { margin: 0; color: #e2e8f0; font-size: 1.15rem; }
    .sector-hero-compact .hero-sub { color: #94a3b8; font-size: 0.78rem; }

    /* Status Pill */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .pill-live { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .pill-hist { background: rgba(234, 179, 8, 0.15); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.3); }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0.5rem 0 0.4rem;
    }
    .section-header h3 { margin: 0; font-size: 1rem; color: #e2e8f0; }
    .section-header .badge {
        background: rgba(124, 58, 237, 0.2);
        color: #a78bfa;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 600;
    }

    /* Drill Down Panel */
    .drill-panel {
        background: linear-gradient(180deg, #13132b 0%, #1a1a2e 100%);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin-top: 0.3rem;
    }
    .drill-panel-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .drill-panel-header h3 { margin: 0; color: #a78bfa; font-size: 0.95rem; }
    .drill-stock-count {
        background: rgba(124, 58, 237, 0.2);
        color: #c4b5fd;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* Divider */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(124, 58, 237, 0.4) 50%, transparent 100%);
        margin: 0.6rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- COMPACT HERO ---
st.markdown("""
<div class="sector-hero-compact">
    <h3>🧭 Sector Scope & Breakout Monitor</h3>
    <span class="hero-sub">Sector Leadership • Volume Surge • ORB Scanner</span>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=120)
def fetch_available_dates():
    engine = get_engine()
    query = text("SELECT DISTINCT date(timestamp) as d FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX' ORDER BY d DESC")
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return [row[0] for row in result if row[0] is not None]
    except Exception as e:
        return []

@st.cache_data(ttl=60)
def fetch_target_date_and_data(selected_date_str=None):
    engine = get_engine()
    if selected_date_str is None:
        # Find latest timestamp in database using index
        query_date = text("SELECT MAX(timestamp) FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX'")
        with engine.connect() as conn:
            max_ts = conn.execute(query_date).scalar()
            if max_ts:
                if isinstance(max_ts, str):
                    selected_date_str = max_ts.split()[0]
                else:
                    selected_date_str = max_ts.strftime("%Y-%m-%d")
            
    if not selected_date_str:
        return None, pd.DataFrame(), None
        
    start_time = f"{selected_date_str} 00:00:00"
    end_time = f"{selected_date_str} 23:59:59"
    query_max_ts = text("""
        SELECT MAX(timestamp) 
        FROM ohlcv_15m 
        WHERE symbol != 'NSE:NIFTY50-INDEX' 
          AND timestamp >= :start_time 
          AND timestamp <= :end_time
    """)
    with engine.connect() as conn:
        last_candle_ts = conn.execute(query_max_ts, {"start_time": start_time, "end_time": end_time}).scalar()
        
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
    
    # Compact status bar
    pill_cls = "pill-live" if is_today else "pill-hist"
    pill_text = "● LIVE" if is_today else "● HIST"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.4rem; flex-wrap:wrap;">
        <span class="status-pill {pill_cls}">{pill_text}</span>
        <span style="color:#94a3b8; font-size:0.78rem;">📅 {latest_date_str}</span>
        <span style="color:#64748b; font-size:0.75rem;">🕒 {db_update_time}</span>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # --- Time-Adjusted RVOL (Relative Volume) ---
        # Compute today's volume and time-adjusted historical average
        volume_today = 0
        avg_vol_time_adj = 0
        vol_surge = 0.0
        
        today_candles = grp_sorted[grp_sorted['timestamp'].dt.date == target_date]
        prev_candles_all = grp_sorted[grp_sorted['timestamp'].dt.date < target_date]
        
        # Today's volume: sum of intraday candle volumes (or live quote)
        if is_today and quotes and symbol in quotes:
            volume_today = quotes[symbol].volume or 0
        elif not today_candles.empty:
            volume_today = int(today_candles['volume'].sum())
        
        # Determine the current time cutoff based on today's available candles or current time
        current_time_limit = None
        if not today_candles.empty:
            current_time_limit = today_candles['timestamp'].dt.time.max()
        elif is_today:
            current_time_limit = datetime.now().time()
            
        # Historical average volume up to the same time of day over the last 10 trading days
        if not prev_candles_all.empty and current_time_limit is not None:
            prev_candles_all = prev_candles_all.copy()
            # Filter historical candles to only include those up to the current time limit
            prev_candles_filtered = prev_candles_all[prev_candles_all['timestamp'].dt.time <= current_time_limit].copy()
            
            if not prev_candles_filtered.empty:
                prev_candles_filtered['trade_date'] = prev_candles_filtered['timestamp'].dt.date
                daily_vols = prev_candles_filtered.groupby('trade_date')['volume'].sum()
                # Use a larger sample (10 days) for time-adjusted average to smooth out noise
                last_10_days = daily_vols.sort_index().tail(10)
                if not last_10_days.empty:
                    avg_vol_time_adj = int(last_10_days.mean())
        
        if avg_vol_time_adj > 0 and volume_today > 0:
            vol_surge = round(volume_today / avg_vol_time_adj, 2)
        
        # Override with live quote LTP and compute pChange if applicable
        pchange = None
        if is_today and quotes and symbol in quotes:
            quote = quotes[symbol]
            close_last = quote.last_price or quote.close or quote.open
            if step is None:
                # Daily return: use the exchange-reported change percentage directly
                pchange = quote.change_percent
                close_prev = quote.previous_close
            else:
                # Intraday return lookbacks: compare today's live LTP with today's database candles
                if not today_candles.empty and len(today_candles) > 1:
                    if len(today_candles) > step:
                        ref_idx = -step
                        close_prev = float(today_candles.iloc[ref_idx]['close'])
                    else:
                        close_prev = float(today_candles.iloc[0]['open'])
                    if close_prev > 0 and close_last > 0:
                        pchange = ((close_last - close_prev) / close_prev) * 100
                else:
                    # Fallback: no intraday candles in DB yet — use daily change from live quote
                    pchange = quote.change_percent
                    close_prev = quote.previous_close
        else:
            # Historical date or fallback
            if step is not None:
                if len(grp_sorted) > step:
                    ref_idx = -step - 1
                    close_prev = float(grp_sorted.iloc[ref_idx]['close'])
                else:
                    close_prev = float(grp_sorted.iloc[0]['close'])
            else:
                # Daily return: compare to previous day's close
                prev_candles = grp_sorted[grp_sorted['timestamp'].dt.date < target_date]
                if not prev_candles.empty:
                    close_prev = float(prev_candles.iloc[-1]['close'])
                else:
                    close_prev = float(grp_sorted.iloc[0]['close'])
            
            if close_prev > 0 and close_last > 0:
                pchange = ((close_last - close_prev) / close_prev) * 100

        if pchange is not None:
            rows.append({
                "symbol": symbol,
                "sector": sector,
                "close_last": close_last,
                "close_prev": close_prev,
                "pChange": pchange,
                "volume_today": volume_today,
                "avg_vol_time_adj": avg_vol_time_adj,
                "vol_surge": vol_surge
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
                    # Always use daily change for symbols with no DB history
                    pchange = quote.change_percent
                    if prev_close > 0 and ltp > 0:
                        live_symbols_to_add.append({
                            "symbol": symbol,
                            "sector": sector,
                            "close_last": ltp,
                            "close_prev": prev_close,
                            "pChange": pchange,
                            "volume_today": quote.volume or 0,
                            "avg_vol_time_adj": 0,
                            "vol_surge": 0.0
                        })
            if live_symbols_to_add:
                merged_closes = pd.concat([merged_closes, pd.DataFrame(live_symbols_to_add)], ignore_index=True)
            st.sidebar.success(f"⚡ Loaded {len(merged_closes)} live quotes")
            
            # Warn if intraday lookback but no DB candles for today
            if step is not None:
                today_candle_count = len(df[df['timestamp'].dt.date == target_date])
                if today_candle_count < 2:
                    st.sidebar.warning(f"⚠️ No intraday candles in DB yet. Showing **daily change** as fallback. Start the live collector for {selected_lookback} precision.")
    else:
        merged_closes = pd.DataFrame(columns=["symbol", "sector", "close_last", "close_prev", "pChange", "volume_today", "avg_vol_time_adj", "vol_surge"])
    
    # Sector performance
    if not merged_closes.empty:
        sector_perf = merged_closes.groupby('sector')['pChange'].mean().reset_index()
    else:
        sector_perf = pd.DataFrame(columns=['sector', 'pChange'])
        
    sector_perf = sector_perf[sector_perf['sector'] != 'UNKNOWN']
    sector_perf = sector_perf.sort_values(by='pChange', ascending=False)
    
    options = sorted(sector_perf['sector'].unique())

    # -----------------------------------------------------------------
    # Pre-compute Intraday VWAP (15m) & Daily RSI (14-Day) for symbols
    # -----------------------------------------------------------------
    def _compute_vwap_rsi(df_all: pd.DataFrame, symbols: list, target_date) -> dict:
        """Returns {symbol: {vwap, vs_vwap, daily_rsi}} for all symbols in list."""
        result = {}
        tgt_df = df_all[df_all['timestamp'].dt.date == target_date]
        if tgt_df.empty and not df_all.empty:
            latest_available = df_all['timestamp'].dt.date.max()
            tgt_df = df_all[df_all['timestamp'].dt.date == latest_available]

        # Fetch daily data in bulk to properly compute 14-day RSI (Wilder's EMA)
        rsi_dict = {}
        if symbols:
            try:
                from trade_system.domains.market_data.infrastructure.database.connection import get_engine
                from sqlalchemy import text
                placeholders = ', '.join([f"'{s}'" for s in symbols])
                # Fetch recent 50 daily candles per symbol to ensure sufficient data for EMA
                query_daily = text(f"""
                    SELECT symbol, timestamp, close 
                    FROM (
                        SELECT symbol, timestamp, close,
                               row_number() over (partition by symbol order by timestamp desc) as rn
                        FROM ohlcv_daily 
                        WHERE symbol IN ({placeholders})
                    )
                    WHERE rn <= 50
                    ORDER BY symbol, timestamp ASC
                """)
                with get_engine().connect() as conn:
                    daily_df = pd.read_sql(query_daily, conn)
                
                if not daily_df.empty:
                    for sym, group in daily_df.groupby('symbol'):
                        closes = group['close']
                        if len(closes) >= 15:
                            delta = closes.diff()
                            gain = delta.where(delta > 0, 0.0)
                            loss = -delta.where(delta < 0, 0.0)
                            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                            val = rsi.iloc[-1]
                            if pd.notna(val):
                                rsi_dict[sym] = round(val, 1)
            except Exception as e:
                pass

        for sym in symbols:
            sym_df_full = df_all[df_all['symbol'] == sym].sort_values('timestamp').copy()
            sym_df = tgt_df[tgt_df['symbol'] == sym].sort_values('timestamp').copy()

            if sym_df_full.empty:
                result[sym] = {"vwap": None, "vs_vwap": None, "daily_rsi": None}
                continue

            # 1. Intraday VWAP on target date 15m candles
            vwap_val = None
            vs_vwap = None
            if not sym_df.empty and len(sym_df) >= 2:
                sym_df['tp'] = (sym_df['high'] + sym_df['low'] + sym_df['close']) / 3.0
                cum_vol = sym_df['volume'].cumsum()
                safe_cum_vol = cum_vol.replace(0, float('nan'))
                vwap_series = (sym_df['tp'] * sym_df['volume']).cumsum() / safe_cum_vol
                vwap_val = float(vwap_series.iloc[-1]) if not vwap_series.dropna().empty else None
                close_last = float(sym_df['close'].iloc[-1])
                vs_vwap = ((close_last - vwap_val) / vwap_val * 100) if vwap_val and vwap_val > 0 else None

            # 2. Daily RSI (14-Day) from pre-calculated dictionary
            daily_rsi_val = rsi_dict.get(sym)

            result[sym] = {"vwap": vwap_val, "vs_vwap": vs_vwap, "daily_rsi": daily_rsi_val}
        return result

    # Pre-calculate global leaderboards
    top_gainers = merged_closes.sort_values(by='pChange', ascending=False).head(10)
    top_losers = merged_closes.sort_values(by='pChange', ascending=True).head(10)

    def _compute_entry_times(prev_closes: dict, target_date) -> dict:
        try:
            from trade_system.domains.market_data.infrastructure.database.connection import get_engine
            from sqlalchemy import text
            query = text("""
                SELECT symbol, timestamp, close
                FROM ohlcv_5m
                WHERE date(timestamp) = :tgt_date
            """)
            with get_engine().connect() as conn:
                tgt_df = pd.read_sql(query, conn, params={"tgt_date": target_date.strftime("%Y-%m-%d")})
        except Exception:
            tgt_df = pd.DataFrame()
            
        if tgt_df.empty:
            return {}
            
        tgt_df['timestamp'] = pd.to_datetime(tgt_df['timestamp'], format='mixed')
        
        # Fast way to compute % change for all symbols at all timestamps
        pivot_closes = tgt_df.pivot_table(index='timestamp', columns='symbol', values='close')
        pivot_closes = pivot_closes.ffill()
        
        entry_times = {}
        prev_series = pd.Series(prev_closes)
        
        symbols_to_keep = pivot_closes.columns.intersection(prev_series.index)
        pivot_closes = pivot_closes[symbols_to_keep]
        prev_series = prev_series[symbols_to_keep]
        
        if pivot_closes.empty:
            return {}
            
        # Compute percentage change
        pchange_df = ((pivot_closes - prev_series) / prev_series) * 100
        
        # Rank gainers/losers
        gainer_ranks = pchange_df.rank(axis=1, ascending=False, method='min')
        loser_ranks = pchange_df.rank(axis=1, ascending=True, method='min')
        
        for sym in symbols_to_keep:
            g_times = gainer_ranks.index[gainer_ranks[sym] <= 10]
            l_times = loser_ranks.index[loser_ranks[sym] <= 10]
            
            g_entry = g_times[0].strftime("%H:%M") if len(g_times) > 0 else "—"
            l_entry = l_times[0].strftime("%H:%M") if len(l_times) > 0 else "—"
            
            entry_times[sym] = {"gainer_entry": g_entry, "loser_entry": l_entry}
            
        return entry_times

    if not merged_closes.empty:
        prev_closes_dict = merged_closes.set_index('symbol')['close_prev'].to_dict()
        entry_times_dict = _compute_entry_times(prev_closes_dict, target_date)
    else:
        entry_times_dict = {}

    # -----------------------------------------------------------------
    # Compute Near Breakout / Near Breakdown stocks (within 0.75% of Key Resistance/Support or High/Low)
    # -----------------------------------------------------------------
    def _compute_near_breakout_breakdown(df_all: pd.DataFrame, merged_df: pd.DataFrame, target_date) -> tuple[list, list]:
        tgt_df = df_all[df_all['timestamp'].dt.date == target_date]
        if tgt_df.empty and not df_all.empty:
            latest_available = df_all['timestamp'].dt.date.max()
            tgt_df = df_all[df_all['timestamp'].dt.date == latest_available]

        near_bo = []
        near_bd = []

        if tgt_df.empty:
            return near_bo, near_bd

        for symbol, grp in tgt_df.groupby('symbol'):
            if symbol.endswith('-INDEX'):
                continue
            grp_sorted = grp.sort_values('timestamp')
            if len(grp_sorted) < 2:
                continue

            day_high = float(grp_sorted['high'].max())
            day_low = float(grp_sorted['low'].min())
            ltp = float(grp_sorted['close'].iloc[-1])
            if ltp <= 0:
                continue

            # Distance to day high / low
            dist_high_pct = ((day_high - ltp) / ltp) * 100
            dist_low_pct = ((ltp - day_low) / ltp) * 100

            # Match sector and volume info from merged_df
            match_row = merged_df[merged_df['symbol'] == symbol]
            sector = match_row['sector'].values[0] if not match_row.empty else "UNKNOWN"
            pchange = match_row['pChange'].values[0] if not match_row.empty else 0.0
            vs = match_row['vol_surge'].values[0] if not match_row.empty and 'vol_surge' in match_row else 0.0

            # Near Breakout: Price within 0.75% below Day High, with positive movement
            if 0 <= dist_high_pct <= 0.75 and pchange > 0:
                near_bo.append({
                    "symbol": symbol,
                    "sector": sector,
                    "ltp": ltp,
                    "level": day_high,
                    "dist_pct": dist_high_pct,
                    "pChange": pchange,
                    "vol_surge": vs,
                    "type": "Near Day High"
                })

            # Near Breakdown: Price within 0.75% above Day Low, with negative movement
            if 0 <= dist_low_pct <= 0.75 and pchange < 0:
                near_bd.append({
                    "symbol": symbol,
                    "sector": sector,
                    "ltp": ltp,
                    "level": day_low,
                    "dist_pct": dist_low_pct,
                    "pChange": pchange,
                    "vol_surge": vs,
                    "type": "Near Day Low"
                })

        near_bo_sorted = sorted(near_bo, key=lambda x: x['dist_pct'])[:5]
        near_bd_sorted = sorted(near_bd, key=lambda x: x['dist_pct'])[:5]

        return near_bo_sorted, near_bd_sorted

    raw_near_bo, raw_near_bd = _compute_near_breakout_breakdown(df, merged_closes, target_date)

    _gl_symbols = list(top_gainers['symbol'].values) + list(top_losers['symbol'].values) + [x['symbol'] for x in raw_near_bo] + [x['symbol'] for x in raw_near_bd]
    _indicators = _compute_vwap_rsi(df, list(set(_gl_symbols)), target_date)

    gainers_data = []
    for idx, row in top_gainers.iterrows():
        sym_clean = row['symbol'].replace("NSE:", "").replace("-EQ", "")
        vs = row.get('vol_surge', 0)
        # Conviction emoji: 🟢 strong (≥1.5x), 🟡 normal (1.0–1.5x), 🔴 weak (<1.0x), ⚪ no data
        if vs >= 1.5:
            vs_label = f"🟢 {vs:.1f}x"
        elif vs >= 1.0:
            vs_label = f"🟡 {vs:.1f}x"
        elif vs > 0:
            vs_label = f"🔴 {vs:.1f}x"
        else:
            vs_label = "⚪ N/A"
        ind = _indicators.get(row['symbol'], {})
        rsi_v = ind.get('daily_rsi')
        vwap_v = ind.get('vwap')
        vs_vwap = ind.get('vs_vwap')
        # RSI label
        if rsi_v is None:
            rsi_label = "—"
        elif rsi_v >= 70:
            rsi_label = f"🔥 {rsi_v:.0f}"
        elif rsi_v <= 30:
            rsi_label = f"🧊 {rsi_v:.0f}"
        else:
            rsi_label = f"{rsi_v:.0f}"
        vwap_v = ind.get('vwap')
        if vs_vwap is None:
            vwap_label = "—"
        elif vs_vwap > 0:
            vwap_label = f"↑ {vs_vwap:+.1f}%"
        else:
            vwap_label = f"↓ {vs_vwap:+.1f}%"
            
        vwap_val_str = f"₹{vwap_v:.2f}" if vwap_v is not None else "—"
        
        entry_info = entry_times_dict.get(row['symbol'], {})
        entry_time = entry_info.get("gainer_entry", "—")
        gainers_data.append({
            "Symbol": sym_clean,
            "Sector": row['sector'],
            "LTP": row['close_last'],
            "Change": row['pChange'],
            "Vol Surge": vs_label,
            "Daily RSI": rsi_label,
            "VWAP": vwap_val_str,
            "Entry Time": entry_time,
        })

    losers_data = []
    for idx, row in top_losers.iterrows():
        sym_clean = row['symbol'].replace("NSE:", "").replace("-EQ", "")
        vs = row.get('vol_surge', 0)
        if vs >= 1.5:
            vs_label = f"🟢 {vs:.1f}x"
        elif vs >= 1.0:
            vs_label = f"🟡 {vs:.1f}x"
        elif vs > 0:
            vs_label = f"🔴 {vs:.1f}x"
        else:
            vs_label = "⚪ N/A"
        ind = _indicators.get(row['symbol'], {})
        rsi_v = ind.get('daily_rsi')
        vs_vwap = ind.get('vs_vwap')
        if rsi_v is None:
            rsi_label = "—"
        elif rsi_v >= 70:
            rsi_label = f"🔥 {rsi_v:.0f}"
        elif rsi_v <= 30:
            rsi_label = f"🧊 {rsi_v:.0f}"
        else:
            rsi_label = f"{rsi_v:.0f}"
        vwap_v = ind.get('vwap')
        if vs_vwap is None:
            vwap_label = "—"
        elif vs_vwap > 0:
            vwap_label = f"↑ {vs_vwap:+.1f}%"
        else:
            vwap_label = f"↓ {vs_vwap:+.1f}%"
            
        vwap_val_str = f"₹{vwap_v:.2f}" if vwap_v is not None else "—"
        
        entry_info = entry_times_dict.get(row['symbol'], {})
        entry_time = entry_info.get("loser_entry", "—")
        losers_data.append({
            "Symbol": sym_clean,
            "Sector": row['sector'],
            "LTP": row['close_last'],
            "Change": row['pChange'],
            "Vol Surge": vs_label,
            "Daily RSI": rsi_label,
            "VWAP": vwap_val_str,
            "Entry Time": entry_time,
        })

    near_bo_data = []
    for item in raw_near_bo:
        sym_clean = item['symbol'].replace("NSE:", "").replace("-EQ", "")
        vs = item.get('vol_surge', 0)
        vs_label = f"🟢 {vs:.1f}x" if vs >= 1.5 else (f"🟡 {vs:.1f}x" if vs >= 1.0 else (f"🔴 {vs:.1f}x" if vs > 0 else "⚪ N/A"))
        ind = _indicators.get(item['symbol'], {})
        rsi_v = ind.get('daily_rsi')
        vs_vwap = ind.get('vs_vwap')
        rsi_label = f"🔥 {rsi_v:.0f}" if rsi_v and rsi_v >= 70 else (f"🧊 {rsi_v:.0f}" if rsi_v and rsi_v <= 30 else (f"{rsi_v:.0f}" if rsi_v else "—"))
        vwap_label = f"↑ {vs_vwap:+.1f}%" if vs_vwap and vs_vwap > 0 else (f"↓ {vs_vwap:+.1f}%" if vs_vwap else "—")
        near_bo_data.append({
            "Symbol": sym_clean,
            "Sector": item['sector'],
            "LTP": item['ltp'],
            "High/Res": item['level'],
            "Dist %": f"{item['dist_pct']:.2f}%",
            "Change": item['pChange'],
            "Vol Surge": vs_label,
            "Daily RSI": rsi_label,
            "vs VWAP": vwap_label
        })

    near_bd_data = []
    for item in raw_near_bd:
        sym_clean = item['symbol'].replace("NSE:", "").replace("-EQ", "")
        vs = item.get('vol_surge', 0)
        vs_label = f"🟢 {vs:.1f}x" if vs >= 1.5 else (f"🟡 {vs:.1f}x" if vs >= 1.0 else (f"🔴 {vs:.1f}x" if vs > 0 else "⚪ N/A"))
        ind = _indicators.get(item['symbol'], {})
        rsi_v = ind.get('daily_rsi')
        vs_vwap = ind.get('vs_vwap')
        rsi_label = f"🔥 {rsi_v:.0f}" if rsi_v and rsi_v >= 70 else (f"🧊 {rsi_v:.0f}" if rsi_v and rsi_v <= 30 else (f"{rsi_v:.0f}" if rsi_v else "—"))
        vwap_label = f"↑ {vs_vwap:+.1f}%" if vs_vwap and vs_vwap > 0 else (f"↓ {vs_vwap:+.1f}%" if vs_vwap else "—")
        near_bd_data.append({
            "Symbol": sym_clean,
            "Sector": item['sector'],
            "LTP": item['ltp'],
            "Low/Sup": item['level'],
            "Dist %": f"{item['dist_pct']:.2f}%",
            "Change": item['pChange'],
            "Vol Surge": vs_label,
            "Daily RSI": rsi_label,
            "vs VWAP": vwap_label
        })

    # Max absolute change for bar widths
    max_change = max(abs(sector_perf['pChange'].max()), abs(sector_perf['pChange'].min()), 0.01)

    # -------------------------------------------------------------
    # 2. Session State Initialization
    # -------------------------------------------------------------
    if 'selected_sector_drill' not in st.session_state:
        st.session_state['selected_sector_drill'] = options[0] if options else None
    if 'clicked_stock_sector' not in st.session_state:
        st.session_state['clicked_stock_sector'] = None

    # Track last processed selections to prevent selection feedback loops
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

    # Check for gainer/loser table clicks
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
            st.session_state['clicked_stock_sector'] = selected_sector_gainer

    elif selected_sector_loser != st.session_state['last_loser_selection']:
        st.session_state['last_loser_selection'] = selected_sector_loser
        if selected_sector_loser in options:
            target_sector = selected_sector_loser
            st.session_state['clicked_stock_sector'] = selected_sector_loser

    if target_sector:
        st.session_state['selected_sector_drill'] = target_sector

    # -------------------------------------------------------------
    # 3. SECTOR CARDS — Compact 6-column Grid
    # -------------------------------------------------------------
    st.markdown("""
    <div class="section-header">
        <h3>🏆 Sector Board</h3>
        <span class="badge">CLICK TO DRILL DOWN</span>
    </div>
    """, unsafe_allow_html=True)

    n_sectors = len(sector_perf)
    cols_per_row = 6
    sector_rows_list = list(sector_perf.iterrows())
    
    for row_start in range(0, n_sectors, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            item_idx = row_start + col_idx
            if item_idx >= n_sectors:
                break
            _, row = sector_rows_list[item_idx]
            with cols[col_idx]:
                is_positive = row['pChange'] >= 0
                bar_color = "#22c55e" if is_positive else "#ef4444"
                bar_width = min(abs(row['pChange']) / max_change * 100, 100)
                arrow = "▲" if is_positive else "▼"
                
                if st.button(
                    f"{row['sector']} {arrow}{row['pChange']:+.1f}%",
                    key=f"sector_card_{row['sector']}",
                    use_container_width=True,
                    type="secondary" if row['sector'] != st.session_state.get('selected_sector_drill') else "primary"
                ):
                    st.session_state['selected_sector_drill'] = row['sector']
                    st.session_state['clicked_stock_sector'] = row['sector']
                    st.rerun()
                
                st.markdown(f'<div style="height:2px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;"><div style="height:100%;width:{bar_width}%;background:{bar_color};"></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. TABBED PANELS — Gainers/Losers | Chart | Scanner | Drill-Down
    # -------------------------------------------------------------
    tab_leaderboard, tab_chart, tab_scanner, tab_drilldown, tab_compression, tab_edge, tab_cycle = st.tabs([
        "📊 Gainers & Losers",
        "📈 Sector Chart",
        "🚨 Breakout Scanner",
        "🔎 Sector Drill-Down",
        "📦 Volatility Squeeze",
        "⚡ Intraday Edge Finder",
        "📅 Off-Market Cycle Analyst"
    ])

    # ---- TAB 1: Gainers & Losers ----
    with tab_leaderboard:
        col_gainers, col_losers = st.columns(2)

        with col_gainers:
            st.markdown("##### 🟢 Top 10 Gainers")
            if gainers_data:
                gainer_df = pd.DataFrame(gainers_data)
                st.dataframe(
                    gainer_df.style.format({
                        "LTP": "₹{:.2f}",
                        "Change": "{:+.2f}%"
                    }).map(
                        lambda v: "color: #22c55e; font-weight:700" if isinstance(v, (int, float)) and v > 0 else "",
                        subset=["Change"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="gainer_leaderboard"
                )

        with col_losers:
            st.markdown("##### 🔴 Top 10 Losers")
            if losers_data:
                loser_df = pd.DataFrame(losers_data)
                st.dataframe(
                    loser_df.style.format({
                        "LTP": "₹{:.2f}",
                        "Change": "{:+.2f}%"
                    }).map(
                        lambda v: "color: #ef4444; font-weight:700" if isinstance(v, (int, float)) and v < 0 else "",
                        subset=["Change"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="loser_leaderboard"
                )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col_near_bo, col_near_bd = st.columns(2)

        with col_near_bo:
            st.markdown("##### 🚀 Stocks Near Breakout (< 0.75% to Day High)")
            if near_bo_data:
                bo_df = pd.DataFrame(near_bo_data)
                st.dataframe(
                    bo_df.style.format({
                        "LTP": "₹{:.2f}",
                        "High/Res": "₹{:.2f}",
                        "Change": "{:+.2f}%"
                    }).map(
                        lambda v: "color: #22c55e; font-weight:700" if isinstance(v, (int, float)) and v > 0 else "",
                        subset=["Change"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    key="near_bo_table"
                )
            else:
                st.info("No stocks currently within 0.75% of intraday breakout level.")

        with col_near_bd:
            st.markdown("##### ⚠️ Stocks Near Breakdown (< 0.75% to Day Low)")
            if near_bd_data:
                bd_df = pd.DataFrame(near_bd_data)
                st.dataframe(
                    bd_df.style.format({
                        "LTP": "₹{:.2f}",
                        "Low/Sup": "₹{:.2f}",
                        "Change": "{:+.2f}%"
                    }).map(
                        lambda v: "color: #ef4444; font-weight:700" if isinstance(v, (int, float)) and v < 0 else "",
                        subset=["Change"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    key="near_bd_table"
                )
            else:
                st.info("No stocks currently within 0.75% of intraday breakdown level.")

        if st.session_state.get('clicked_stock_sector'):
            clicked_sector = st.session_state['clicked_stock_sector']
            sector_stocks_df = merged_closes[merged_closes['sector'] == clicked_sector].copy()
            if not sector_stocks_df.empty:
                sector_stocks_df['Symbol'] = sector_stocks_df['symbol'].str.replace("NSE:", "").str.replace("-EQ", "")
                sector_stocks_df = sector_stocks_df.sort_values('pChange', ascending=False)
                sector_avg = sector_stocks_df['pChange'].mean()
                
                st.markdown(f"""
                <div class="drill-panel">
                    <div class="drill-panel-header">
                        <h3>📋 {clicked_sector} — All Stocks</h3>
                        <span class="drill-stock-count">{len(sector_stocks_df)} stocks • Avg: {sector_avg:+.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Build vol surge & indicator labels for drill-down
                def _vol_surge_label(vs):
                    if vs >= 1.5:
                        return f"🟢 {vs:.1f}x"
                    elif vs >= 1.0:
                        return f"🟡 {vs:.1f}x"
                    elif vs > 0:
                        return f"🔴 {vs:.1f}x"
                    return "⚪ N/A"

                sector_stocks_df['Vol Surge'] = sector_stocks_df['vol_surge'].apply(_vol_surge_label)

                # Map Daily RSI & vs VWAP indicators
                rsi_labels = []
                vwap_labels = []
                for sym in sector_stocks_df['symbol']:
                    ind = _indicators.get(sym, {})
                    rsi_v = ind.get('daily_rsi')
                    vs_vwap = ind.get('vs_vwap')
                    if rsi_v is None:
                        rsi_labels.append("—")
                    elif rsi_v >= 70:
                        rsi_labels.append(f"🔥 {rsi_v:.0f}")
                    elif rsi_v <= 30:
                        rsi_labels.append(f"🧊 {rsi_v:.0f}")
                    else:
                        rsi_labels.append(f"{rsi_v:.0f}")

                    if vs_vwap is None:
                        vwap_labels.append("—")
                    elif vs_vwap > 0:
                        vwap_labels.append(f"↑ {vs_vwap:+.1f}%")
                    else:
                        vwap_labels.append(f"↓ {vs_vwap:+.1f}%")

                sector_stocks_df['Daily RSI'] = rsi_labels
                sector_stocks_df['vs VWAP'] = vwap_labels

                display_df = sector_stocks_df[['Symbol', 'close_last', 'pChange', 'Vol Surge', 'Daily RSI', 'vs VWAP']].copy()
                display_df.columns = ['Symbol', 'LTP (₹)', 'Change %', 'Vol Surge', 'Daily RSI', 'vs VWAP']
                st.dataframe(
                    display_df.style.format({
                        "LTP (₹)": "₹{:.2f}",
                        "Change %": "{:+.2f}%"
                    }).map(
                        lambda v: "color: #22c55e; font-weight:700" if isinstance(v, (int, float)) and v > 0 else ("color: #ef4444; font-weight:700" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=["Change %"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=min(300, len(sector_stocks_df) * 35 + 38)
                )

    # ---- TAB 2: Sector Chart ----
    with tab_chart:
        fig = go.Figure()
        colors = ['#22c55e' if v >= 0 else '#ef4444' for v in sector_perf['pChange']]
        fig.add_trace(go.Bar(
            x=sector_perf['sector'],
            y=sector_perf['pChange'],
            marker_color=colors,
            marker_line_color='rgba(255,255,255,0.1)',
            marker_line_width=1,
            text=[f"{v:+.2f}%" for v in sector_perf['pChange']],
            textposition='outside',
            textfont=dict(size=10, color='#e2e8f0'),
            hovertemplate='<b>%{x}</b><br>Return: %{y:+.2f}%<extra></extra>'
        ))
        fig.update_layout(
            height=320,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(title="", tickangle=-35, gridcolor='rgba(255,255,255,0.03)'),
            yaxis=dict(title="Avg Return (%)", gridcolor='rgba(255,255,255,0.05)', zeroline=True, zerolinecolor='rgba(255,255,255,0.15)', zerolinewidth=1),
            margin=dict(t=15, b=70, l=50, r=15),
            bargap=0.2,
        )
        import inspect
        sig = inspect.signature(st.plotly_chart)
        if "on_select" in sig.parameters:
            st.plotly_chart(fig, use_container_width=True, key="plotly_sector_chart", on_select="rerun")
        else:
            st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 3: Breakout Scanner ----
    with tab_scanner:
        with st.expander("⚙️ Scanner Filters & Strategies", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                use_sector = st.checkbox("Sector Leadership", value=False)
            with col2:
                use_vwap = st.checkbox("VWAP Filter", value=True)
            with col3:
                use_wick = st.checkbox("Wick Filter", value=True)
            with col4:
                use_index = st.checkbox("Index Alignment", value=True)
            with col5:
                vol_surge = st.selectbox("Min Vol", options=[1.0, 1.2, 1.5, 1.8, 2.0, 2.5], index=2, format_func=lambda x: f"{x}x")
            
            col1b, col2b, col3b, col4b = st.columns(4)
            with col1b:
                use_52w = st.checkbox("52W High", value=False)
            with col2b:
                use_weekly = st.checkbox("Weekly High", value=False)
            with col3b:
                use_daily_st_touch = st.checkbox("Daily ST Touch", value=False)
            with col4b:
                use_abnormal_vol = st.checkbox("Abnormal Vol", value=False)
                
            selected_strategies = st.multiselect(
                "🎯 Strategy Selection",
                options=["ORB Breakout", "Consolidation Breakout", "Gap Fill", "Mean Reversion", "VWAP Pullback", "Momentum Mover", "Previous Day High/Low"],
                default=["ORB Breakout", "Consolidation Breakout", "Gap Fill", "Mean Reversion", "VWAP Pullback"],
                help="Filter the scanner results by one or more trading strategies"
            )
        
        strategy_map = {
            "ORB Breakout": ["ORB_BREAKOUT"],
            "Consolidation Breakout": ["CONSOLIDATION_BREAKOUT"],
            "Gap Fill": ["GAP_FILL"],
            "Mean Reversion": ["MEAN_REVERSION"],
            "VWAP Pullback": ["VWAP_PULLBACK"],
            "Momentum Mover": ["MOMENTUM_MOVER"],
            "Previous Day High/Low": ["PREV_DAY_HIGH", "PREV_DAY_LOW"]
        }
        allowed_types = []
        for s in selected_strategies:
            allowed_types.extend(strategy_map[s])

        screener = BreakoutScreener()
        try:
            alerts = screener.scan_for_breakouts(
                top_sectors_count=3, target_date=target_date,
                use_vwap_filter=use_vwap, use_wick_filter=use_wick,
                use_index_filter=use_index, use_sector_filter=use_sector,
                vol_surge_threshold=float(vol_surge),
                use_52w_filter=use_52w, use_weekly_high_filter=use_weekly,
                use_daily_st_touch_filter=use_daily_st_touch,
                use_abnormal_vol_filter=use_abnormal_vol
            )
        except Exception as e:
            st.error(f"Error: {e}")
            alerts = []
        
        if alerts:
            alerts = [a for a in alerts if a.get('alert_type') in allowed_types]
            
        if alerts:
            alert_rows = []
            for a in alerts:
                clean_sym = a['symbol'].replace("NSE:", "").replace("-EQ", "")
                vol_ratio = a['volume'] / a['vol_sma'] if a['vol_sma'] > 0 else 0
                direction_badge = "🟢 LONG" if a['direction'] in ["LONG", "CALL"] else "🔴 SHORT"
                
                type_badges = {
                    "ORB_BREAKOUT": "📊 ORB",
                    "CONSOLIDATION_BREAKOUT": "📦 Cons",
                    "GAP_FILL": "🔄 GapFill",
                    "MEAN_REVERSION": "🎯 Rev",
                    "VWAP_PULLBACK": "⚓ Pullback",
                    "MOMENTUM_MOVER": "🚀 Mom",
                    "PREV_DAY_HIGH": "⬆️ PDH",
                    "PREV_DAY_LOW": "⬇️ PDL"
                }
                type_badge = type_badges.get(a.get('alert_type', 'ORB_BREAKOUT'), a.get('alert_type', ''))
                
                # Format Trigger/Ref info based on strategy
                ref_level = a.get('orb_high') if a.get('direction') == "LONG" else a.get('orb_low')
                if a.get('alert_type') == 'ORB_BREAKOUT':
                    ref_str = f"ORB: {ref_level:.2f}" if ref_level else "—"
                elif a.get('alert_type') == 'CONSOLIDATION_BREAKOUT':
                    ref_str = f"R/S: {a.get('resistance', 0.0):.1f}/{a.get('support', 0.0):.1f}"
                elif a.get('alert_type') == 'GAP_FILL':
                    ref_str = f"Gap: {a.get('gap_pct', 0.0):+.1f}%"
                elif a.get('alert_type') == 'MEAN_REVERSION':
                    ref_str = f"RSI: {a.get('rsi', 0.0):.0f}"
                elif a.get('alert_type') == 'VWAP_PULLBACK':
                    ref_str = f"VWAP: {a.get('vwap', 0.0):.1f}"
                else:
                    ref_str = f"{ref_level:.2f}" if ref_level else "—"
                    
                tags = []
                if a.get('is_52w_high'): tags.append("52W🔥")
                if a.get('is_weekly_high'): tags.append("Wk📈")
                if a.get('touches_daily_st'): tags.append("ST🎯")
                trigger_time_str = a['trigger_time'].strftime("%H:%M") if 'trigger_time' in a and pd.notna(a['trigger_time']) else "15:15"
                alert_rows.append({
                    "Time": trigger_time_str, "Symbol": clean_sym, "Sector": a['sector'],
                    "Type": type_badge, "Dir": direction_badge,
                    "Chg%": f"{a.get('pchange', 0):+.2f}%",
                    "LTP": f"{a['close']:.2f}", "Trigger/Ref": ref_str,
                    "VolX": f"{vol_ratio:.1f}x", "Tags": ", ".join(tags) or "—"
                })
            st.dataframe(pd.DataFrame(alert_rows).set_index("Time"), use_container_width=True)
        else:
            st.info("No active alerts found for the selected strategies and filters.")

    # ---- TAB 4: Sector Drill-Down ----
    with tab_drilldown:
        selected_sector = st.selectbox("Sector", options, key="selected_sector_drill")
        
        if selected_sector:
            sector_symbols = get_stocks_by_sector(selected_sector)
            sector_data = merged_closes[merged_closes['sector'] == selected_sector]
            s_avg = sector_data['pChange'].mean() if not sector_data.empty else 0
            s_max = sector_data['pChange'].max() if not sector_data.empty else 0
            s_min = sector_data['pChange'].min() if not sector_data.empty else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sector", selected_sector)
            m2.metric("Avg", f"{s_avg:+.2f}%")
            m3.metric("Best", f"{s_max:+.2f}%")
            m4.metric("Worst", f"{s_min:+.2f}%")
            
            with st.expander("🔧 Filters", expanded=False):
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                with col_f1:
                    f_52w = st.checkbox("52W High Only", value=False, key=f"f_52w_{selected_sector}")
                with col_f2:
                    f_weekly = st.checkbox("Weekly High Only", value=False, key=f"f_weekly_{selected_sector}")
                with col_f3:
                    f_st = st.checkbox("Daily ST Touch", value=False, key=f"f_st_{selected_sector}")
                with col_f4:
                    f_abnormal = st.checkbox("Abnormal Vol", value=False, key=f"f_abnormal_{selected_sector}")
                
            daily_groups = {}
            if sector_symbols:
                engine = get_engine()
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
                    st.warning(f"Failed to fetch daily candles: {e}")

            stock_rows = []
            for symbol in sector_symbols:
                stock_all = df[df['symbol'] == symbol].sort_values('timestamp')
                if len(stock_all) < 20:
                    continue
                stock_all['vol_sma_20'] = stock_all['volume'].rolling(window=20).mean()
                stock_today = stock_all[stock_all['timestamp'].dt.date == target_date]
                if stock_today.empty and not is_today:
                    continue
                latest_row = stock_today.iloc[-1] if not stock_today.empty else None
                if latest_row is None and is_today and symbol in quotes:
                    quote = quotes[symbol]
                    ltp = quote.last_price or quote.close or quote.open
                    vol = quote.volume
                    latest_row = pd.Series({"close": ltp, "volume": vol})
                if latest_row is None:
                    continue
                grp_d = daily_groups.get(symbol)
                orb_candles = stock_today.head(4)
                orb_high = orb_candles['high'].max() if not orb_candles.empty else latest_row['close']
                orb_low = orb_candles['low'].min() if not orb_candles.empty else latest_row['close']
                if not stock_today.empty:
                    prev_vol_sma = stock_today.iloc[-2]['vol_sma_20'] if len(stock_today) > 1 else stock_all.iloc[-2]['vol_sma_20']
                    vol_ratio = latest_row['volume'] / prev_vol_sma if prev_vol_sma > 0 else 0
                else:
                    if grp_d is not None and not grp_d.empty:
                        daily_vols = grp_d['volume'].iloc[:-1] if len(grp_d) > 1 else grp_d['volume']
                        avg_daily_vol = daily_vols.tail(20).mean() if not daily_vols.empty else 0
                        now_ist = datetime.now()
                        market_start_dt = datetime.combine(now_ist.date(), dt_time(9, 15))
                        market_end_dt = datetime.combine(now_ist.date(), dt_time(15, 30))
                        if market_start_dt <= now_ist <= market_end_dt:
                            elapsed_minutes = (now_ist - market_start_dt).total_seconds() / 60.0
                        elif now_ist > market_end_dt:
                            elapsed_minutes = 375.0
                        else:
                            elapsed_minutes = 1.0
                        fraction_of_day = max(min(elapsed_minutes / 375.0, 1.0), 0.01)
                        expected_vol = avg_daily_vol * fraction_of_day
                        vol_ratio = latest_row['volume'] / expected_vol if expected_vol > 0 else 0
                    else:
                        vol_ratio = 1.0
                if not stock_today.empty:
                    cum_pv = (stock_today['close'] * stock_today['volume']).cumsum()
                    cum_vol = stock_today['volume'].cumsum()
                    vwap = cum_pv.iloc[-1] / cum_vol.iloc[-1]
                else:
                    vwap = latest_row['close']
                status = "Range"
                if latest_row['close'] > orb_high: status = "🟢 Breakout"
                elif latest_row['close'] < orb_low: status = "🔴 Breakdown"
                sym_change = merged_closes[merged_closes['symbol'] == symbol]
                p_change = sym_change.iloc[0]['pChange'] if not sym_change.empty else 0.0
                grp_d = daily_groups.get(symbol)
                is_52w_high = is_weekly_high = touches_daily_st = False
                if grp_d is not None and not grp_d.empty:
                    latest_close = float(latest_row['close'])
                    daily_highs = grp_d['high'].iloc[:-1] if len(grp_d) > 1 else grp_d['high']
                    if not daily_highs.empty:
                        is_52w_high = latest_close >= (daily_highs.max() * 0.995)
                    weekly_highs = grp_d['high'].iloc[-6:-1] if len(grp_d) > 5 else grp_d['high']
                    if not weekly_highs.empty:
                        is_weekly_high = latest_close >= (weekly_highs.max() * 0.995)
                    try:
                        from trade_system.domains.strategy.application.indicators import calculate_supertrend
                        st_d = calculate_supertrend(grp_d)
                        if not st_d.empty:
                            last_st_d = st_d.iloc[-1]
                            st_val = float(last_st_d['supertrend'])
                            touches_daily_st = float(last_st_d['low']) <= st_val <= float(last_st_d['high'])
                    except Exception:
                        pass
                if f_52w and not is_52w_high: continue
                if f_weekly and not is_weekly_high: continue
                if f_st and not touches_daily_st: continue
                if f_abnormal and vol_ratio < float(vol_surge): continue

                ind = _indicators.get(symbol, {})
                rsi_v = ind.get('daily_rsi')
                if rsi_v is None:
                    rsi_label = "—"
                elif rsi_v >= 70:
                    rsi_label = f"🔥 {rsi_v:.0f}"
                elif rsi_v <= 30:
                    rsi_label = f"🧊 {rsi_v:.0f}"
                else:
                    rsi_label = f"{rsi_v:.0f}"

                stock_rows.append({
                    "Symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                    "LTP (₹)": latest_row['close'], "Change %": p_change,
                    "Vol Surge": vol_ratio, "VWAP": vwap,
                    "Daily RSI": rsi_label,
                    "ORB Hi": orb_high, "ORB Lo": orb_low,
                    "Status": status,
                    "52W": "🔥" if is_52w_high else "—",
                    "Wk Hi": "📈" if is_weekly_high else "—",
                    "ST": "🎯" if touches_daily_st else "—"
                })
            if stock_rows:
                df_stocks = pd.DataFrame(stock_rows).sort_values(by="Change %", ascending=False)
                st.dataframe(
                    df_stocks.style.format({
                        "LTP (₹)": "{:.2f}", "Change %": "{:+.2f}%",
                        "Vol Surge": "{:.1f}x", "VWAP": "{:.2f}",
                        "ORB Hi": "{:.2f}", "ORB Lo": "{:.2f}"
                    }),
                    use_container_width=True, hide_index=True
                )
            else:
                st.caption("No stock data matching filters.")

    # ---- TAB 5: Volatility Squeeze ----
    with tab_compression:
        st.subheader("📦 Volatility Compression & Squeeze Radar")
        st.caption("Scans F&O and liquid stocks for low-volatility coiling patterns (NR7, Inside Bars) that typically precede explosive moves.")
        
        from trade_system.domains.strategy.application.indicators.compression import CompressionIndicator
        
        screener = BreakoutScreener()
        fo_symbols = list(screener.fo_metadata.keys())
        
        if not fo_symbols:
            st.info("No F&O symbols available in configuration.")
        else:
            with st.spinner("Scanning database for volatility compression setups..."):
                engine = get_engine()
                # Fetch recent 30 bars of 15m candles for all stocks up to target_date
                query_comp = text("""
                    SELECT symbol, timestamp, open, high, low, close, volume 
                    FROM ohlcv_15m 
                    WHERE timestamp >= date(:target_date, '-10 days') AND timestamp <= :target_date
                    ORDER BY symbol, timestamp ASC
                """)
                try:
                    with engine.connect() as conn:
                        df_comp = pd.read_sql(query_comp, conn, params={"target_date": target_date.isoformat()})
                    df_comp['timestamp'] = pd.to_datetime(df_comp['timestamp'], format="mixed")
                except Exception as ex:
                    st.error(f"Failed to query database for compression scan: {ex}")
                    df_comp = pd.DataFrame()
            
            if df_comp.empty:
                st.info("No 15-minute candle data available to scan for compression.")
            else:
                # Group and apply CompressionIndicator
                compression_rows = []
                indicator = CompressionIndicator(atr_period=14, lookback=4)
                
                for symbol, group in df_comp.groupby('symbol'):
                    group = group.sort_values('timestamp').reset_index(drop=True)
                    if len(group) < 20:
                        continue
                    
                    # Calculate compression
                    calc_df = indicator.calculate(group)
                    if calc_df.empty:
                        continue
                        
                    latest_bar = calc_df.iloc[-1]
                    is_comp = latest_bar.get('is_compressed', False)
                    is_nr7 = latest_bar.get('nr7', False)
                    is_inside = latest_bar.get('inside_bar', False)
                    comp_score = latest_bar.get('range_compression', 0.0)
                    atr = latest_bar.get('atr', 0.0)
                    close = latest_bar.get('close', 0.0)
                    
                    status = []
                    if is_nr7: status.append("NR7 🎯")
                    if is_inside: status.append("Inside Bar 📥")
                    if comp_score > 20: status.append(f"Range Squeeze ({comp_score:.0f}%)")
                    
                    status_str = ", ".join(status) if status else "No Squeeze"
                    
                    clean_sym = symbol.replace("NSE:", "").replace("-EQ", "")
                    sector = screener.fo_metadata.get(symbol, "UNKNOWN")
                    
                    compression_rows.append({
                        "Symbol": clean_sym,
                        "Sector": sector,
                        "Close (₹)": close,
                        "ATR (₹)": atr,
                        "Squeeze Score": comp_score,
                        "Setup Signals": status_str,
                        "Coiled": "🔥 YES" if is_comp else "—",
                        "_is_comp": is_comp
                    })
                
                if compression_rows:
                    comp_df = pd.DataFrame(compression_rows)
                    
                    col_show_all, = st.columns([1])
                    with col_show_all:
                        show_all = st.checkbox("Show all stocks (including uncoiled)", value=False, key="comp_show_all_cb")
                    
                    if not show_all:
                        display_df = comp_df[comp_df["_is_comp"] == True]
                    else:
                        display_df = comp_df
                        
                    display_df = display_df.drop(columns=["_is_comp"]).sort_values(by="Squeeze Score", ascending=False)
                    
                    if display_df.empty:
                        st.info("No coiled or compressed setups found for this date. Check 'Show all' to browse raw values.")
                    else:
                        st.dataframe(
                            display_df.style.format({
                                "Close (₹)": "{:.2f}",
                                "ATR (₹)": "{:.2f}",
                                "Squeeze Score": "{:.1f}%"
                            }).map(
                                lambda v: "color: #ff9f1c; font-weight:700" if v == "🔥 YES" else "",
                                subset=["Coiled"]
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.info("No stock data returned from indicator scanning.")

    # ---- TAB 6: Intraday Edge Finder ----
    with tab_edge:
        st.subheader("⚡ Multi-Layer Edge Finder")
        st.caption("Confluence-based scoring system analyzing 7 layers (Sector, Relative Strength, VWAP, Volume/CVD, Timing, Supertrend, and Compression) to find high-probability setups.")
        
        def get_approx_strike(ltp):
            if ltp <= 0:
                return 0
            if ltp > 10000:
                step = 100
            elif ltp > 5000:
                step = 50
            elif ltp > 2000:
                step = 20
            elif ltp > 1000:
                step = 10
            elif ltp > 500:
                step = 5
            elif ltp > 200:
                step = 2.5
            else:
                step = 1.0
            return round(ltp / step) * step

        # Horizon Selector
        selected_horizon = st.radio(
            "⏳ Trade Horizon",
            options=["⚡ Intraday", "📅 Weekly Swing", "🏛️ Monthly Positional"],
            horizontal=True,
            key="edge_horizon_selector"
        )
        
        horizon_map = {
            "⚡ Intraday": "INTRADAY",
            "📅 Weekly Swing": "WEEKLY",
            "🏛️ Monthly Positional": "MONTHLY"
        }
        horizon_code = horizon_map[selected_horizon]

        # 1. UI Controls
        control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        with control_col1:
            min_score = st.slider("Minimum Edge Score", min_value=30, max_value=100, value=65, step=5, key="edge_min_score_slider")
        with control_col2:
            direction_filter = st.selectbox("Direction Filter", options=["All", "CALL only", "PUT only"], key="edge_dir_filter_sb")
        with control_col3:
            show_triggered_only = st.checkbox("Show Triggered Entries Only", value=False, key="edge_triggered_only_cb")

        # 2. Scanning and Scoring
        with st.spinner(f"Analyzing 7 confluence layers for {selected_horizon} setups..."):
            try:
                scorer = IntradayEdgeScorer(horizon=horizon_code)
                if horizon_code == "INTRADAY":
                    edges = scorer.scan(
                        target_date=target_date,
                        sector_perf=sector_perf,
                        stock_perf=merged_closes
                    )
                    df_base_trig = df_filtered
                else:
                    edges = scorer.scan(
                        target_date=target_date
                    )
                    # Fetch base candles for triggers
                    lookback_days = 365 if horizon_code == "WEEKLY" else 1000
                    df_daily_all = scorer._fetch_daily_data(target_date, lookback_days=lookback_days)
                    if horizon_code == "WEEKLY":
                        df_base_trig = df_daily_all
                    else:
                        df_base_trig = scorer._resample_candles(df_daily_all, "W")
                
                # Apply entry triggers to the candidates
                trigger_eval = SmartEntryTrigger(horizon=horizon_code)
                enriched_edges = []
                for edge in edges:
                    # Filter by score
                    if edge.final_score < min_score:
                        continue
                    # Filter by direction
                    if direction_filter == "CALL only" and edge.direction != "CALL":
                        continue
                    if direction_filter == "PUT only" and edge.direction != "PUT":
                        continue
                        
                    # Find matching base candles
                    sym_df = df_base_trig[df_base_trig["symbol"] == edge.symbol]
                    if not sym_df.empty:
                        trig = trigger_eval.evaluate(
                            direction=edge.direction,
                            ltp=edge.ltp,
                            atr=edge.atr,
                            df_base=sym_df,
                            target_date=target_date
                        )
                        if trig:
                            edge.entry_price = trig.entry_price
                            edge.stop_loss = trig.stop_loss
                            edge.target_1 = trig.target_1
                            edge.target_2 = trig.target_2
                            edge.entry_status = trig.status
                            
                    # Filter by entry status if checked
                    if show_triggered_only and edge.entry_status != "TRIGGERED":
                        continue
                        
                    enriched_edges.append(edge)
            except Exception as e:
                st.error(f"Error running Edge Scorer: {e}")
                enriched_edges = []

        if not enriched_edges:
            st.info("No stocks match the selected edge score and trigger filters. Try lowering the Minimum Edge Score slider.")
        else:
            # 3. Display High-Conviction Dashboard Cards for the Top 3
            top_3 = sorted([e for e in enriched_edges if e.final_score >= 70], key=lambda x: x.final_score, reverse=True)[:3]
            
            if top_3:
                st.markdown("### 🏆 Top High-Conviction Setups")
                card_cols = st.columns(len(top_3))
                for idx, col in enumerate(card_cols):
                    edge = top_3[idx]
                    with col:
                        # Styling based on direction
                        dir_color = "#22c55e" if edge.direction == "CALL" else "#ef4444"
                        dir_bg = "rgba(34, 197, 94, 0.1)" if edge.direction == "CALL" else "rgba(239, 68, 68, 0.1)"
                        dir_border = "rgba(34, 197, 94, 0.3)" if edge.direction == "CALL" else "rgba(239, 68, 68, 0.3)"
                        
                        clean_sym = edge.symbol.replace("NSE:", "").replace("-EQ", "")
                        approx_strike = get_approx_strike(edge.ltp)
                        
                        status_badge = "⚪ WAITING"
                        if edge.entry_status == "TRIGGERED":
                            status_badge = "🟢 TRIGGERED"
                        elif edge.entry_status == "APPROACHING":
                            status_badge = "🟡 APPROACHING"
                            
                        # Recommendation details based on horizon
                        if horizon_code == "INTRADAY":
                            inst_label = "Option Strike"
                            inst_rec = f"₹{approx_strike:.0f} {edge.direction == 'CALL' and 'CE' or 'PE'}"
                            rec_footer = f"ATM Option: ₹{approx_strike:.0f} {edge.direction == 'CALL' and 'CE' or 'PE'}"
                            title_label = "Intraday Edge Alert"
                        elif horizon_code == "WEEKLY":
                            inst_label = "Option Strike"
                            inst_rec = f"₹{approx_strike:.0f} {edge.direction == 'CALL' and 'CE' or 'PE'} (Next Month)"
                            rec_footer = f"Option Strike: ₹{approx_strike:.0f} {edge.direction == 'CALL' and 'CE' or 'PE'} (Next Month)"
                            title_label = "Weekly Swing Alert"
                        else:  # MONTHLY
                            inst_label = "Trade Tool"
                            inst_rec = "Equity Delivery / LEAPS"
                            rec_footer = "Equity Delivery / LEAPS option suggested"
                            title_label = "Monthly Positional Alert"

                        st.markdown(f"""
                        <div style="background:{dir_bg}; border: 1px solid {dir_border}; padding:15px; border-radius:10px; margin-bottom:15px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:1.3rem; font-weight:700; color:#ffffff;">{clean_sym}</span>
                                <span style="background:{dir_color}; color:#111827; font-weight:800; font-size:0.75rem; padding:2px 8px; border-radius:4px;">{edge.direction}</span>
                            </div>
                            <div style="color:#94a3b8; font-size:0.8rem; margin-top:2px;">Sector: {edge.sector}</div>
                            <hr style="margin:10px 0; border-color:rgba(255,255,255,0.08);"/>
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="color:#94a3b8; font-size:0.85rem;">Edge Score:</span>
                                <span style="color:#ffffff; font-weight:700; font-size:1.1rem;">{edge.final_score:.0f}/100</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="color:#94a3b8; font-size:0.85rem;">LTP:</span>
                                <span style="color:#ffffff; font-weight:700;">₹{edge.ltp:,.2f} ({edge.change_pct:+.2f}%)</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="color:#94a3b8; font-size:0.85rem;">Entry Status:</span>
                                <span style="color:{dir_color if edge.entry_status == 'TRIGGERED' else '#eab308' if edge.entry_status == 'APPROACHING' else '#94a3b8'}; font-weight:700;">{status_badge}</span>
                            </div>
                            <hr style="margin:10px 0; border-color:rgba(255,255,255,0.08);"/>
                            <div style="font-size:0.85rem; margin-bottom:5px; color:#a78bfa; font-weight:600;">📐 Recommended Levels:</div>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span style="color:#94a3b8;">Entry:</span>
                                <span style="color:#ffffff; font-weight:700;">₹{edge.entry_price:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span style="color:#94a3b8;">Stop Loss:</span>
                                <span style="color:#ef4444; font-weight:700;">₹{edge.stop_loss:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span style="color:#94a3b8;">Target 1:</span>
                                <span style="color:#22c55e; font-weight:700;">₹{edge.target_1:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span style="color:#94a3b8;">Target 2:</span>
                                <span style="color:#22c55e; font-weight:700;">₹{edge.target_2:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span style="color:#94a3b8;">Risk-Reward:</span>
                                <span style="color:#eab308; font-weight:700;">{edge.risk_reward:.2f}R</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span style="color:#94a3b8;">{inst_label}:</span>
                                <span style="color:#38bdf8; font-weight:700;">{inst_rec}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Alert Button
                        msg = (
                            f"⚡ <b>{title_label}</b> ⚡\n\n"
                            f"Symbol: <b>#{clean_sym}</b> ({edge.sector})\n"
                            f"Direction: <b>{'🟢 BUY CALL / LONG' if edge.direction == 'CALL' else '🔴 BUY PUT / SHORT'}</b>\n"
                            f"Edge Score: <b>{edge.final_score:.0f}/100</b>\n"
                            f"LTP: ₹{edge.ltp:,.2f} ({edge.change_pct:+.2f}%)\n"
                            f"Status: <b>{edge.entry_status}</b>\n\n"
                            f"📐 <b>Levels:</b>\n"
                            f"• Entry: ₹{edge.entry_price:,.2f}\n"
                            f"• Stop Loss: ₹{edge.stop_loss:,.2f}\n"
                            f"• Target 1: ₹{edge.target_1:,.2f}\n"
                            f"• Target 2: ₹{edge.target_2:,.2f}\n"
                            f"• Risk-Reward: <b>{edge.risk_reward:.2f}R</b>\n\n"
                            f"💡 <i>{rec_footer}</i>"
                        )
                        
                        if st.button(f"📢 Alert {clean_sym}", key=f"alert_btn_{edge.symbol}"):
                            try:
                                settings = Settings.load()
                                notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
                                notifier.send(msg)
                                st.toast(f"✅ Alert for {clean_sym} sent successfully!")
                            except Exception as alert_ex:
                                st.error(f"Failed to send alert: {alert_ex}")

            # 4. Ranked Table of all matching setups
            st.markdown("### 📋 Full Scored Watchlist")
            
            table_rows = []
            for edge in enriched_edges:
                clean_sym = edge.symbol.replace("NSE:", "").replace("-EQ", "")
                
                # Extract layer score status icons
                sec_ok = "🟢" if edge.layers["sector_momentum"].direction == edge.direction else "⚪"
                rs_ok = "🟢" if edge.layers["relative_strength"].direction == edge.direction else "⚪"
                vwap_ok = "🟢" if edge.layers["vwap_location"].direction == edge.direction else "⚪"
                vol_ok = "🟢" if edge.layers["volume_confirmation"].direction == edge.direction else "⚪"
                timing_ok = "🟢" if edge.layers["momentum_timing"].direction == edge.direction else "⚪"
                st_ok = "🟢" if edge.layers["supertrend_alignment"].direction == edge.direction else "⚪"
                comp_ok = "🟢" if edge.layers["compression_release"].direction == edge.direction else "⚪"
                
                table_rows.append({
                    "Symbol": clean_sym,
                    "Sector": edge.sector,
                    "Edge Score": edge.final_score,
                    "Dir": "🟢 CALL" if edge.direction == "CALL" else "🔴 PUT",
                    "Status": edge.entry_status,
                    "LTP (₹)": edge.ltp,
                    "Chg %": edge.change_pct,
                    "Sector Mtm": sec_ok,
                    "Rel Strength": rs_ok,
                    "VWAP Loc": vwap_ok,
                    "Vol Conf": vol_ok,
                    "Timing": timing_ok,
                    "ST Align": st_ok,
                    "Comp Release": comp_ok,
                    "_symbol": edge.symbol
                })
                
            if table_rows:
                df_table = pd.DataFrame(table_rows)
                st.dataframe(
                    df_table.style.format({
                        "LTP (₹)": "{:.2f}",
                        "Chg %": "{:+.2f}%",
                        "Edge Score": "{:.0f}/100"
                    }).map(
                        lambda v: "color: #22c55e; font-weight:700" if v == "🟢 CALL" else "color: #ef4444; font-weight:700" if v == "🔴 PUT" else "",
                        subset=["Dir"]
                    ).map(
                        lambda v: "color: #22c55e; font-weight:700" if v == "TRIGGERED" else "color: #eab308; font-weight:700" if v == "APPROACHING" else "",
                        subset=["Status"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="edge_finder_table"
                )
                
                # Check for table selection drill down
                selected_row = st.session_state.get("edge_finder_table", {}).get("selection", {}).get("rows", [])
                if selected_row:
                    row_idx = selected_row[0]
                    if 0 <= row_idx < len(enriched_edges):
                        selected_edge = enriched_edges[row_idx]
                        
                        st.markdown(f"### 🔍 Detailed Analysis: **{selected_edge.symbol.replace('NSE:', '').replace('-EQ', '')}**")
                        
                        # 7 Layers breakdown
                        st.markdown("#### 🛡️ Confluence Layer Breakdown")
                        for layer_key, result in selected_edge.layers.items():
                            layer_label = layer_key.replace("_", " ").title()
                            
                            # Subscore bar
                            st.markdown(f"**{layer_label}** — {result.detail}")
                            bar_color = "#22c55e" if result.direction == "CALL" else "#ef4444" if result.direction == "PUT" else "#94a3b8"
                            bar_width = int(result.score * 100)
                            st.markdown(f'<div style="height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;margin-bottom:10px;"><div style="height:100%;width:{bar_width}%;background:{bar_color};"></div></div>', unsafe_allow_html=True)

    # ---- TAB 7: Off-Market Cycle Analyst ----
    with tab_cycle:
        st.markdown("### 📅 Time Cycles & Off-Market Analyst")
        st.caption("Quantitative swing cycle phase estimation and daily pivot calculation for off-market preparation.")
        
        # Load F&O symbols list
        all_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"] + list(get_sector_mapping().keys())
        selected_cycle_symbol = st.selectbox(
            "Select Asset for Analysis",
            options=all_symbols,
            key="selected_cycle_symbol"
        )
        
        if selected_cycle_symbol:
            engine = get_engine()
            # Fetch daily candles (180 days lookback to find cycle history)
            query_cycle = text("""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_daily 
                WHERE symbol = :sym AND timestamp >= date(:target_date, '-180 days') AND timestamp <= :target_date
                ORDER BY timestamp ASC
            """)
            try:
                with engine.connect() as conn:
                    df_cycle = pd.read_sql(query_cycle, conn, params={
                        "sym": selected_cycle_symbol,
                        "target_date": target_date.isoformat()
                    })
                
                if df_cycle.empty or len(df_cycle) < 30:
                    st.warning("Insufficient daily historical candles in database to run cycle analysis.")
                else:
                    df_cycle['timestamp'] = pd.to_datetime(df_cycle['timestamp'], format='mixed')
                    
                    # 1. Hurst Swing Cycle Calculations
                    df_c = df_cycle.copy().sort_values("timestamp").reset_index(drop=True)
                    window = 10
                    troughs = []
                    peaks = []
                    n = len(df_c)
                    
                    for idx in range(window, n - window):
                        val = df_c.loc[idx, "close"]
                        sub = df_c.loc[idx - window : idx + window, "close"]
                        if val == sub.min():
                            troughs.append((df_c.loc[idx, "timestamp"], val, idx))
                        elif val == sub.max():
                            peaks.append((df_c.loc[idx, "timestamp"], val, idx))
                            
                    if not troughs:
                        min_idx = df_c["close"].idxmin()
                        troughs.append((df_c.loc[min_idx, "timestamp"], df_c.loc[min_idx, "close"], min_idx))
                        
                    cycle_lengths = []
                    for j in range(1, len(troughs)):
                        diff = (troughs[j][0] - troughs[j-1][0]).days
                        cycle_lengths.append(diff)
                        
                    avg_cycle = np.mean(cycle_lengths) if cycle_lengths else 20.0
                    if pd.isna(avg_cycle) or avg_cycle <= 0:
                        avg_cycle = 20.0
                        
                    last_trough_date, last_trough_val, last_trough_idx = troughs[-1]
                    latest_row = df_c.iloc[-1]
                    days_elapsed = (latest_row["timestamp"].date() - last_trough_date.date()).days
                    trading_days_elapsed = max(0, int(days_elapsed * 5 / 7))
                    
                    t_trading = avg_cycle * 5 / 7
                    if t_trading <= 0:
                        t_trading = 15.0
                    progress = trading_days_elapsed / t_trading
                    
                    if progress < 0.35:
                        phase = "Fresh Impulse (Rising)"
                        phase_icon = "📈"
                        bias = "BULLISH (High Probability Long)"
                        color = "#22c55e"
                    elif 0.35 <= progress <= 0.65:
                        phase = "Exhaustion Zone (Peaking)"
                        phase_icon = "⚠️"
                        bias = "NEUTRAL (Risk of Pullback, Avoid Longs)"
                        color = "#eab308"
                    else:
                        phase = "Reversion Phase (Falling)"
                        phase_icon = "📉"
                        bias = "BEARISH (Look for Short Entries)"
                        color = "#ef4444"
                        
                    # Metrics Grid
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Est. Cycle Period", f"{avg_cycle:.1f} Calendar Days")
                    col_m2.metric("Cycle Progress", f"{progress:.0%} ({trading_days_elapsed:.0f}/{t_trading:.0f} Trading Days)")
                    col_m3.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 2px;">Current Phase</div>
                        <div style="color: {color}; font-weight: 700; font-size: 1.1rem;">{phase_icon} {phase}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    col_m4.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 2px;">Trading Bias</div>
                        <div style="color: {color}; font-weight: 700; font-size: 1.1rem;">{bias}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 2. Charts: Price History and Cycle Wave
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        st.markdown("##### 📈 Swing Peaks & Troughs")
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(
                            x=df_c["timestamp"], y=df_c["close"],
                            name="Close Price", line=dict(color="#38bdf8", width=2)
                        ))
                        # Add Troughs
                        tr_dates = [t[0] for t in troughs]
                        tr_vals = [t[1] for t in troughs]
                        fig_price.add_trace(go.Scatter(
                            x=tr_dates, y=tr_vals,
                            mode="markers", name="Troughs (Bottoms)",
                            marker=dict(symbol="triangle-up", size=12, color="#22c55e")
                        ))
                        # Add Peaks
                        pk_dates = [p[0] for p in peaks]
                        pk_vals = [p[1] for p in peaks]
                        fig_price.add_trace(go.Scatter(
                            x=pk_dates, y=pk_vals,
                            mode="markers", name="Peaks (Tops)",
                            marker=dict(symbol="triangle-down", size=12, color="#ef4444")
                        ))
                        fig_price.update_layout(
                            height=350, margin=dict(l=0, r=0, t=10, b=0),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_price, use_container_width=True)
                        
                    with col_chart2:
                        st.markdown("##### 🌊 Cycle Phase Wave")
                        # Draw a sine wave representing the cycle phase progression
                        wave_x = np.linspace(0, 1, 100)
                        wave_y = -np.cos(2 * np.pi * wave_x)
                        
                        fig_wave = go.Figure()
                        # Base Sine Wave
                        fig_wave.add_trace(go.Scatter(
                            x=wave_x, y=wave_y,
                            mode="lines", name="Cycle Phase",
                            line=dict(color="#a78bfa", width=2, dash="dash")
                        ))
                        # Current Position dot
                        curr_progress = min(1.0, max(0.0, progress))
                        curr_y = -np.cos(2 * np.pi * curr_progress)
                        fig_wave.add_trace(go.Scatter(
                            x=[curr_progress], y=[curr_y],
                            mode="markers", name="Current Phase",
                            marker=dict(size=14, color=color, symbol="circle")
                        ))
                        fig_wave.update_layout(
                            height=350, margin=dict(l=0, r=0, t=10, b=0),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title="Cycle Progress", tickformat=".0%"),
                            yaxis=dict(title="Amplitude (Bullish vs Bearish)", tickvals=[-1, 0, 1], ticktext=["Trough (Buy)", "Neutral", "Peak (Sell)"]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_wave, use_container_width=True)
                        
                    # 3. Off-Market Calculations (DeMark, Camarilla, Volume)
                    st.markdown("---")
                    st.markdown("##### 🛡️ Off-Market Prep Levels (Next Session)")
                    
                    last_c = float(latest_row["close"])
                    last_o = float(latest_row["open"])
                    last_h = float(latest_row["high"])
                    last_l = float(latest_row["low"])
                    last_v = float(latest_row["volume"])
                    
                    # Demark Projector
                    if last_c > last_o:
                        demark_x = last_h + last_l + last_c + last_h
                    elif last_c < last_o:
                        demark_x = last_h + last_l + last_c + last_l
                    else:
                        demark_x = last_h + last_l + last_c + last_c
                    proj_h = demark_x / 2 - last_l
                    proj_l = demark_x / 2 - last_h
                    
                    # Camarilla
                    rng = last_h - last_l
                    r4 = last_c + rng * 1.1 / 2
                    r3 = last_c + rng * 1.1 / 4
                    s3 = last_c - rng * 1.1 / 4
                    s4 = last_c - rng * 1.1 / 2
                    
                    # Volume Z-score
                    avg_v_20 = df_c["volume"].tail(20).mean()
                    std_v_20 = df_c["volume"].tail(20).std()
                    vol_z = (last_v - avg_v_20) / std_v_20 if std_v_20 > 0 else 0.0
                    vol_status = "High Institutional (Accumulation/Distribution)" if vol_z > 1.5 else "Neutral/Speculative Noise"
                    
                    col_p1, col_p2, col_p3 = st.columns(3)
                    
                    with col_p1:
                        st.markdown("🎯 **DeMark Next-Day Range**")
                        st.markdown(f"• **Projected High**: ₹{proj_h:,.2f}")
                        st.markdown(f"• **Projected Low**: ₹{proj_l:,.2f}")
                        st.markdown(f"• **Previous Close**: ₹{last_c:,.2f}")
                        
                    with col_p2:
                        st.markdown("📐 **Camarilla Key Pivots**")
                        st.markdown(f"• **Resistance R4**: ₹{r4:,.2f} *(Breakout target)*")
                        st.markdown(f"• **Resistance R3**: ₹{r3:,.2f} *(Mean-reversion short)*")
                        st.markdown(f"• **Support S3**: ₹{s3:,.2f} *(Mean-reversion long)*")
                        st.markdown(f"• **Support S4**: ₹{s4:,.2f} *(Breakdown target)*")
                        
                    with col_p3:
                        st.markdown("📊 **Volume Profile Prep**")
                        st.markdown(f"• **Session Volume**: {last_v:,.0f}")
                        st.markdown(f"• **Volume Z-Score**: `{vol_z:+.2f}`")
                        st.markdown(f"• **Institutional Scan**: **{vol_status}**")
                        
            except Exception as ex_cycle:
                st.warning(f"Error computing cycles: {ex_cycle}")

    # Footer
    st.caption("🧭 Sector Scope | AI Trade System V2 | Data refreshes every 60s")
