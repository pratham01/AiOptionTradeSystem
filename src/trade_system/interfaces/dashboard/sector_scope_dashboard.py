import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

LOGGER = logging.getLogger(__name__)
logger = LOGGER
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
from trade_system.domains.analysis.application.analysis.reversal_scanner import DailyReversalScanner, DailyReversalSetup
from trade_system.domains.analysis.application.analysis.breakout_breakdown_proximity_screener import (
    BreakoutBreakdownProximityScreener,
    ProximitySetup,
)
from trade_system.domains.analysis.application.analysis.fo_pcr_screener import (
    FOPCRScreener,
    StockPCRInfo,
)
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

@st.cache_data(ttl=86400)
def _fetch_historical_15m(target_date_str):
    """Fetch immutable historical data (up to the day before target_date) and cache for 24 hours."""
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date(:latest_date, '-10 days') 
          AND timestamp < date(:latest_date)
        ORDER BY symbol, timestamp ASC
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"latest_date": target_date_str})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    return df

@st.cache_data(ttl=60)
def _fetch_today_15m(target_date_str):
    """Fetch only the target day's data, cached for 60 seconds (for live intraday updates)."""
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date(:latest_date) 
          AND timestamp < date(:latest_date, '+1 day')
        ORDER BY symbol, timestamp ASC
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"latest_date": target_date_str})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    
    # Get last candle timestamp
    last_candle_ts = None
    if not df.empty:
        last_candle_ts = df[df['symbol'] != 'NSE:NIFTY50-INDEX']['timestamp'].max()
    return df, last_candle_ts

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
        
    df_hist = _fetch_historical_15m(selected_date_str)
    df_today, last_candle_ts = _fetch_today_15m(selected_date_str)
    
    df = pd.concat([df_hist, df_today], ignore_index=True)
    return selected_date_str, df, last_candle_ts

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_symbol_15m_cached(symbol: str, target_date_str: str) -> pd.DataFrame:
    """Fetch 15m candles for a single symbol from Fyers API, cached for 60s."""
    try:
        from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
        from trade_system.shared.config import Settings
        settings = Settings.load()
        manager = get_broker_manager(settings)
        fyers_broker_health = manager.brokers.get('fyers')
        broker = fyers_broker_health.broker if fyers_broker_health else None
        
        if broker:
            df_sym = broker.fetch_history(
                symbol=symbol,
                resolution="15",
                range_from=target_date_str,
                range_to=target_date_str,
                date_format="1"
            )
            if df_sym is not None and not df_sym.empty:
                df_sym['symbol'] = symbol
                df_sym['timestamp'] = pd.to_datetime(df_sym['timestamp'], errors='coerce').dt.tz_localize(None)
                return df_sym[['symbol', 'timestamp', 'close']]
    except Exception as e:
        logger.warning("Failed to fetch 15m history for %s: %s", symbol, e)
    return pd.DataFrame()


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
        
    db_update_time = pd.to_datetime(last_candle_ts).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(last_candle_ts) else "N/A"
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
    df_filtered = df[df['timestamp'].dt.date <= target_date].copy()
    df_filtered['trade_date'] = df_filtered['timestamp'].dt.date
    df_filtered['trade_time'] = df_filtered['timestamp'].dt.time
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
        
        today_candles = grp_sorted[grp_sorted['trade_date'] == target_date]
        prev_candles_all = grp_sorted[grp_sorted['trade_date'] < target_date]
        
        # Today's volume: sum of intraday candle volumes (or live quote)
        if is_today and quotes and symbol in quotes:
            volume_today = quotes[symbol].volume or 0
        elif not today_candles.empty:
            volume_today = int(today_candles['volume'].sum())
        
        # Determine the current time cutoff based on today's available candles or current time
        current_time_limit = None
        if not today_candles.empty:
            current_time_limit = today_candles['trade_time'].max()
        elif is_today:
            current_time_limit = datetime.now().time()
            
        # Historical average volume up to the same time of day over the last 10 trading days
        if not prev_candles_all.empty and current_time_limit is not None:
            prev_candles_all = prev_candles_all.copy()
            # Filter historical candles to only include those up to the current time limit
            prev_candles_filtered = prev_candles_all[prev_candles_all['trade_time'] <= current_time_limit].copy()
            
            if not prev_candles_filtered.empty:
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
                if (pchange is None or pchange == 0.0) and close_prev > 0 and close_last > 0:
                    pchange = ((close_last - close_prev) / close_prev) * 100
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
                    if (pchange is None or pchange == 0.0) and close_prev > 0 and close_last > 0:
                        pchange = ((close_last - close_prev) / close_prev) * 100
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
                    if (pchange is None or pchange == 0.0) and prev_close > 0 and ltp > 0:
                        pchange = ((ltp - prev_close) / prev_close) * 100
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
    def _compute_vwap_rsi(df_all: pd.DataFrame, symbols: list, target_date, is_today: bool = False, quotes: dict = None) -> dict:
        """Returns {symbol: {vwap, vs_vwap, daily_rsi}} for all symbols in list."""
        def _clean_sym(s: str) -> str:
            return s.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-INDEX", "")

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
                
                expanded_syms = set()
                for s in symbols:
                    expanded_syms.add(s)
                    c = _clean_sym(s)
                    expanded_syms.add(f"NSE:{c}-EQ")
                    expanded_syms.add(f"BSE:{c}-INDEX")
                    expanded_syms.add(f"NSE:{c}-INDEX")

                placeholders = ', '.join([f"'{s}'" for s in expanded_syms])
                # Fetch recent 250 daily candles per symbol to ensure sufficient data for EMA warmup (matches TradingView)
                query_daily = text(f"""
                    SELECT symbol, timestamp, close 
                    FROM (
                        SELECT symbol, timestamp, close,
                               row_number() over (partition by symbol order by timestamp desc) as rn
                        FROM ohlcv_daily 
                        WHERE symbol IN ({placeholders})
                          AND date(timestamp) <= :tgt_date
                    )
                    WHERE rn <= 250
                    ORDER BY symbol, timestamp ASC
                """)
                with get_engine().connect() as conn:
                    daily_df = pd.read_sql(query_daily, conn, params={"tgt_date": target_date.strftime("%Y-%m-%d")})
                
                if not daily_df.empty:
                    for sym, group in daily_df.groupby('symbol'):
                        closes = group['close']
                        clean = _clean_sym(sym)
                        
                        # Append the live LTP as the current forming daily candle (how TradingView computes live daily RSI)
                        last_ltp = None
                        if is_today and quotes:
                            q = quotes.get(sym) or quotes.get(clean) or quotes.get(f"NSE:{clean}-EQ")
                            if q:
                                last_ltp = q.last_price or q.close or q.open
                        
                        if last_ltp is None:
                            sym_tgt = tgt_df[tgt_df['symbol'] == sym]
                            if sym_tgt.empty:
                                sym_tgt = tgt_df[tgt_df['symbol'].apply(_clean_sym) == clean]
                            if not sym_tgt.empty:
                                last_ltp = float(sym_tgt.iloc[-1]['close'])
                        
                        if last_ltp is not None:
                            last_daily_date = pd.to_datetime(group.iloc[-1]['timestamp']).date()
                            if last_daily_date < target_date:
                                closes = pd.concat([closes, pd.Series([last_ltp])], ignore_index=True)

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
                                rsi_dict[clean] = round(val, 1)
                                rsi_dict[f"NSE:{clean}-EQ"] = round(val, 1)
            except Exception as e:
                pass

        # Build fast lookup map for df_all and tgt_df by cleaned symbol
        df_all_cleaned = df_all.copy()
        df_all_cleaned['_clean'] = df_all_cleaned['symbol'].apply(_clean_sym)
        tgt_df_cleaned = tgt_df.copy()
        tgt_df_cleaned['_clean'] = tgt_df_cleaned['symbol'].apply(_clean_sym)

        for sym in symbols:
            clean = _clean_sym(sym)
            sym_df_full = df_all_cleaned[df_all_cleaned['_clean'] == clean].sort_values('timestamp')
            sym_df = tgt_df_cleaned[tgt_df_cleaned['_clean'] == clean].sort_values('timestamp')

            vwap_val = None
            vs_vwap = None
            close_last = None

            quote_obj = None
            if quotes:
                quote_obj = quotes.get(sym) or quotes.get(clean) or quotes.get(f"NSE:{clean}-EQ")

            if not sym_df.empty:
                sym_df = sym_df.copy()
                sym_df['tp'] = (sym_df['high'] + sym_df['low'] + sym_df['close']) / 3.0
                vol_sum = sym_df['volume'].sum()
                if vol_sum > 0:
                    cum_vol = sym_df['volume'].cumsum()
                    safe_cum_vol = cum_vol.replace(0, float('nan'))
                    vwap_series = (sym_df['tp'] * sym_df['volume']).cumsum() / safe_cum_vol
                    vwap_val = float(vwap_series.dropna().iloc[-1]) if not vwap_series.dropna().empty else float(sym_df['tp'].iloc[-1])
                else:
                    vwap_val = float(sym_df['tp'].iloc[-1])
                close_last = float(sym_df['close'].iloc[-1])

            # Fallback for live quotes if target date bars not in DB
            if vwap_val is None and quote_obj:
                q_ltp = quote_obj.last_price or quote_obj.close
                q_open = quote_obj.open or q_ltp
                q_high = quote_obj.high or q_ltp
                q_low = quote_obj.low or q_ltp
                if q_high and q_low and q_ltp:
                    vwap_val = round(float(q_high + q_low + q_ltp) / 3.0, 2)
                    close_last = float(q_ltp)

            # Fallback to historical full dataframe if still None
            if vwap_val is None and not sym_df_full.empty:
                sym_df_full = sym_df_full.copy()
                sym_df_full['tp'] = (sym_df_full['high'] + sym_df_full['low'] + sym_df_full['close']) / 3.0
                vwap_val = float(sym_df_full['tp'].iloc[-1])
                close_last = float(sym_df_full['close'].iloc[-1])

            if quote_obj and (quote_obj.last_price or quote_obj.close):
                close_last = float(quote_obj.last_price or quote_obj.close)

            if vwap_val is not None and close_last is not None and vwap_val > 0:
                vs_vwap = ((close_last - vwap_val) / vwap_val * 100)
            elif vwap_val is not None:
                vs_vwap = 0.0

            daily_rsi_val = rsi_dict.get(sym) or rsi_dict.get(clean) or rsi_dict.get(f"NSE:{clean}-EQ")

            entry = {"vwap": vwap_val, "vs_vwap": vs_vwap, "daily_rsi": daily_rsi_val}
            result[sym] = entry
            result[clean] = entry
            result[f"NSE:{clean}-EQ"] = entry

        return result

    # Pre-calculate global leaderboards
    top_gainers = merged_closes.sort_values(by='pChange', ascending=False).head(10)
    top_losers = merged_closes.sort_values(by='pChange', ascending=True).head(10)

    def _compute_entry_times(merged_df: pd.DataFrame, target_date) -> dict:
        """Compute the earliest 15m timestamp at which each symbol first entered
        the top-10 gainers or bottom-10 losers list.

        To produce accurate rankings across the full F&O universe (~208 symbols),
        this function:
        1. Loads 15m intraday candles for symbols that have them (~82 symbols).
        2. For the remaining symbols (available only via live quotes), it holds
           their current ``pChange`` constant across all timestamps.
        3. Ranks all symbols together at every 15m bar.
        """
        try:
            from trade_system.domains.market_data.infrastructure.database.connection import get_engine
            from sqlalchemy import text
            query = text("""
                SELECT symbol, timestamp, close
                FROM ohlcv_15m
                WHERE date(timestamp) = :tgt_date
            """)
            with get_engine().connect() as conn:
                tgt_df = pd.read_sql(query, conn, params={"tgt_date": target_date.strftime("%Y-%m-%d")})
        except Exception:
            tgt_df = pd.DataFrame()

        # Identify gainer and loser symbols from live quotes
        top_gainers_syms = merged_df.sort_values(by='pChange', ascending=False).head(10)['symbol'].tolist()
        top_losers_syms = merged_df.sort_values(by='pChange', ascending=True).head(10)['symbol'].tolist()
        top_symbols = set(top_gainers_syms + top_losers_syms)

        # Identify which of these top symbols are missing from the local 15m DB
        db_syms = set(tgt_df['symbol'].unique()) if not tgt_df.empty else set()
        missing_top_syms = top_symbols - db_syms - {s for s in top_symbols if 'INDEX' in s}

        # Dynamically fetch 15m history from broker API for missing top symbols
        if missing_top_syms:
            today_str = target_date.strftime("%Y-%m-%d")
            fetched_dfs = []
            for sym in missing_top_syms:
                df_sym = _fetch_symbol_15m_cached(sym, today_str)
                if not df_sym.empty:
                    fetched_dfs.append(df_sym)
            if fetched_dfs:
                new_data = pd.concat(fetched_dfs, ignore_index=True)
                if tgt_df.empty:
                    tgt_df = new_data
                else:
                    tgt_df = pd.concat([tgt_df, new_data], ignore_index=True)

        if tgt_df.empty or merged_df.empty:
            return {}

        tgt_df['timestamp'] = pd.to_datetime(tgt_df['timestamp'], errors='coerce')

        # Build a prev_closes series from merged_df
        prev_closes = merged_df.set_index('symbol')['close_prev'].to_dict()
        prev_series = pd.Series(prev_closes)

        # Pivot 15m candles: rows = timestamps, columns = symbols, values = close
        pivot_closes = tgt_df.pivot_table(index='timestamp', columns='symbol', values='close')
        pivot_closes = pivot_closes.sort_index().ffill()

        # Compute pChange for DB symbols at every timestamp
        db_syms = pivot_closes.columns.intersection(prev_series.index)
        if db_syms.empty:
            return {}

        pchange_db = ((pivot_closes[db_syms] - prev_series[db_syms]) / prev_series[db_syms]) * 100

        # For non-DB symbols, use their current pChange (constant across all bars)
        all_syms_in_merged = set(merged_df['symbol'].tolist())
        non_db_syms = all_syms_in_merged - set(db_syms) - {s for s in all_syms_in_merged if 'INDEX' in s}
        if non_db_syms:
            pchange_map = merged_df.set_index('symbol')['pChange'].to_dict()
            non_db_pchange = pd.DataFrame(
                {sym: pchange_map.get(sym, 0.0) for sym in non_db_syms},
                index=pchange_db.index,
            )
            pchange_df = pd.concat([pchange_db, non_db_pchange], axis=1)
        else:
            pchange_df = pchange_db

        # Rank across ALL symbols at each timestamp
        gainer_ranks = pchange_df.rank(axis=1, ascending=False, method='min')
        loser_ranks  = pchange_df.rank(axis=1, ascending=True,  method='min')

        entry_times = {}
        for sym in pchange_df.columns:
            g_times = gainer_ranks.index[gainer_ranks[sym] <= 10]
            l_times = loser_ranks.index[loser_ranks[sym] <= 10]

            g_entry = g_times.min().strftime("%H:%M") if len(g_times) > 0 else "—"
            l_entry = l_times.min().strftime("%H:%M") if len(l_times) > 0 else "—"

            entry_times[sym] = {"gainer_entry": g_entry, "loser_entry": l_entry}

        return entry_times

    if not merged_closes.empty:
        entry_times_dict = _compute_entry_times(merged_closes, target_date)
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

    _all_scope_symbols = list(merged_closes['symbol'].unique()) if 'symbol' in merged_closes.columns else list(set(_gl_symbols))
    _indicators = _compute_vwap_rsi(df, _all_scope_symbols, target_date, is_today=is_today, quotes=quotes)

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
        ind = _indicators.get(row['symbol']) or _indicators.get(sym_clean) or {}
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
        ind = _indicators.get(row['symbol']) or _indicators.get(sym_clean) or {}
        rsi_v = ind.get('daily_rsi')
        vwap_v = ind.get('vwap')
        vs_vwap = ind.get('vs_vwap')
        if rsi_v is None:
            rsi_label = "—"
        elif rsi_v >= 70:
            rsi_label = f"🔥 {rsi_v:.0f}"
        elif rsi_v <= 30:
            rsi_label = f"🧊 {rsi_v:.0f}"
        else:
            rsi_label = f"{rsi_v:.0f}"
        
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
        ind = _indicators.get(item['symbol']) or _indicators.get(sym_clean) or {}
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
        ind = _indicators.get(item['symbol']) or _indicators.get(sym_clean) or {}
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
    # 3. SECTOR BOARD & RIGHT-SIDE CONSTITUENT STOCKS
    # -------------------------------------------------------------
    active_sector = st.session_state.get('selected_sector_drill') or (options[0] if options else None)
    
    col_board, col_stocks = st.columns([1.1, 0.9], gap="medium")

    with col_board:
        st.markdown("""
        <div class="section-header" style="margin-bottom: 6px;">
            <h3>🏆 Sector Board</h3>
            <span class="badge">CLICK TO VIEW CONSTITUENTS</span>
        </div>
        """, unsafe_allow_html=True)

        n_sectors = len(sector_perf)
        cols_per_row = 3
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
                    is_selected = (row['sector'] == active_sector)
                    
                    if st.button(
                        f"{row['sector']} {arrow}{row['pChange']:+.1f}%",
                        key=f"sector_card_{row['sector']}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state['selected_sector_drill'] = row['sector']
                        st.session_state['clicked_stock_sector'] = row['sector']
                        st.rerun()
                    
                    st.markdown(f'<div style="height:2px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;margin-bottom:4px;"><div style="height:100%;width:{bar_width}%;background:{bar_color};"></div></div>', unsafe_allow_html=True)

    with col_stocks:
        if active_sector:
            sector_stocks_df = merged_closes[merged_closes['sector'] == active_sector].copy()
            if not sector_stocks_df.empty:
                sector_stocks_df['Symbol'] = sector_stocks_df['symbol'].str.replace("NSE:", "").str.replace("-EQ", "")
                sector_stocks_df = sector_stocks_df.sort_values('pChange', ascending=False)
                sector_avg = sector_stocks_df['pChange'].mean()
                
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(124, 58, 237, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 0.98rem; font-weight: 700; color: #f8fafc;">📋 {active_sector} Stocks</span>
                        <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); font-size: 0.7rem; padding: 1px 7px; border-radius: 9999px; font-weight: 600;">{len(sector_stocks_df)} Stocks</span>
                    </div>
                    <span style="color: {'#22c55e' if sector_avg >= 0 else '#ef4444'}; font-weight: 700; font-size: 0.85rem;">Avg: {sector_avg:+.2f}%</span>
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
                    clean_s = sym.replace("NSE:", "").replace("-EQ", "")
                    ind = _indicators.get(sym) or _indicators.get(clean_s) or {}
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
                    height=345
                )
            else:
                st.info(f"No stock data available for {active_sector}.")
        else:
            st.info("Select a sector from the board on the left.")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. TABBED PANELS — Gainers/Losers | Chart | Scanner | Proximity | PCR | Reversals | Edge | Drill-Down | Squeeze | Cycle
    # -------------------------------------------------------------
    tab_leaderboard, tab_chart, tab_scanner, tab_proximity, tab_pcr, tab_reversal, tab_edge, tab_drilldown, tab_compression, tab_cycle = st.tabs([
        "📊 Gainers & Losers",
        "📈 Sector Chart",
        "🚨 Breakout Scanner",
        "🎯 Multi-Touch Proximity (KEI Pattern)",
        "🎲 F&O Stock PCR Radar (Overbought/Oversold)",
        "🔄 Daily Reversal Radar",
        "⚡ Intraday Edge Finder",
        "🔎 Sector Drill-Down",
        "📦 Volatility Squeeze",
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
            st.plotly_chart(fig, use_container_width=True, key="plotly_sector_chart")
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
                options=[
                    "ORB Breakout",
                    "Daily Supertrend Touch",
                    "Daily Supertrend Flip",
                    "Consolidation Breakout",
                    "Gap Fill",
                    "Mean Reversion",
                    "VWAP Pullback",
                    "Momentum Mover",
                    "Previous Day High/Low"
                ],
                default=["ORB Breakout", "Daily Supertrend Touch", "Daily Supertrend Flip", "Consolidation Breakout", "VWAP Pullback"],
                help="Filter the scanner results by one or more trading strategies"
            )
        
        strategy_map = {
            "ORB Breakout": ["ORB_BREAKOUT"],
            "Daily Supertrend Touch": ["DAILY_ST_TOUCH"],
            "Daily Supertrend Flip": ["DAILY_ST_FLIP"],
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
                    "DAILY_ST_TOUCH": "🎯 ST Touch",
                    "DAILY_ST_FLIP": "🔄 ST Flip",
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
                elif a.get('alert_type') == 'DAILY_ST_TOUCH':
                    ref_str = f"ST: ₹{a.get('supertrend_level', 0.0):.1f}"
                elif a.get('alert_type') == 'DAILY_ST_FLIP':
                    ref_str = f"ST Flip: ₹{a.get('supertrend_level', 0.0):.1f}"
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
                
                t_val = a.get('trigger_time')
                if pd.notna(t_val) and hasattr(t_val, 'strftime'):
                    trigger_time_str = t_val.strftime("%H:%M")
                elif pd.notna(t_val) and str(t_val) not in ('', 'None', 'nan'):
                    trigger_time_str = str(t_val).split()[-1][:5]
                else:
                    trigger_time_str = "—"

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

        # ── Dedicated Daily Supertrend Radar for F&O Universe ──────────────────
        st.markdown("---")
        with st.expander("🎯 Daily Supertrend Radar & Direction Flips (Full F&O Universe)", expanded=True):
            st.caption("Scans the entire F&O universe on the Daily timeframe (Period=10, Multiplier=2) for stocks touching the Supertrend line (Pullback Zone) or flipping direction (Trend Reversal).")
            
            try:
                engine = get_engine()
                with engine.connect() as conn:
                    df_daily_st = pd.read_sql(
                        text("""
                            SELECT symbol, timestamp, open, high, low, close, volume 
                            FROM ohlcv_daily 
                            WHERE timestamp >= date(:t_date, '-120 days') AND timestamp <= :t_date
                            ORDER BY symbol, timestamp ASC
                        """),
                        conn,
                        params={"t_date": target_date.isoformat()}
                    )
                
                if not df_daily_st.empty:
                    from trade_system.domains.strategy.application.indicators import calculate_supertrend
                    df_daily_st['timestamp'] = pd.to_datetime(df_daily_st['timestamp'], format='mixed')
                    
                    st_records = []
                    total_bull = 0
                    total_bear = 0
                    total_touch = 0
                    total_flip = 0

                    for sym, grp in df_daily_st.groupby('symbol'):
                        if len(grp) < 10:
                            continue
                        clean_sym = sym.replace("NSE:", "").replace("-EQ", "")
                        st_calc = calculate_supertrend(grp, period=10, multiplier=2)
                        if st_calc.empty:
                            continue
                        
                        last_bar = st_calc.iloc[-1]
                        prev_bar = st_calc.iloc[-2] if len(st_calc) >= 2 else last_bar
                        
                        ltp = float(last_bar['close'])
                        low = float(last_bar['low'])
                        high = float(last_bar['high'])
                        st_val = float(last_bar['supertrend'])
                        curr_dir = int(last_bar['supertrend_direction'])
                        prev_dir = int(prev_bar['supertrend_direction'])
                        atr_val = float(last_bar.get('atr', 0.0))
                        
                        prev_close = float(prev_bar['close']) if prev_bar is not None else ltp
                        chg_pct = ((ltp - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
                        dist_pct = ((ltp - st_val) / st_val) * 100
                        
                        if curr_dir == 1:
                            total_bull += 1
                        else:
                            total_bear += 1
                            
                        # Touch check
                        is_touch = False
                        if curr_dir == 1 and low <= st_val * 1.01 and ltp >= st_val * 0.99:
                            is_touch = True
                            total_touch += 1
                        elif curr_dir == -1 and high >= st_val * 0.99 and ltp <= st_val * 1.01:
                            is_touch = True
                            total_touch += 1
                            
                        # Flip check
                        is_flip = (curr_dir != prev_dir)
                        if is_flip:
                            total_flip += 1
                            
                        # Setup label
                        if is_flip and curr_dir == 1:
                            setup_type = "🔄 Bullish Flip (New Buy)"
                        elif is_flip and curr_dir == -1:
                            setup_type = "🔄 Bearish Flip (New Sell)"
                        elif is_touch and curr_dir == 1:
                            setup_type = "🎯 At Support (Bullish Pullback)"
                        elif is_touch and curr_dir == -1:
                            setup_type = "🎯 At Resistance (Bearish Pullback)"
                        elif curr_dir == 1:
                            setup_type = "🟢 Bullish Trend Active"
                        else:
                            setup_type = "🔴 Bearish Trend Active"

                        st_records.append({
                            "Symbol": clean_sym,
                            "Sector": fo_universe.get(sym, "OTHER") if 'fo_universe' in locals() or 'fo_universe' in globals() else "F&O",
                            "Direction": "🟢 BULLISH" if curr_dir == 1 else "🔴 BEARISH",
                            "LTP (₹)": ltp,
                            "Chg %": chg_pct,
                            "Supertrend (₹)": st_val,
                            "Dist to ST %": dist_pct,
                            "Setup": setup_type,
                            "is_touch": is_touch,
                            "is_flip": is_flip,
                            "curr_dir": curr_dir,
                            "_symbol": sym
                        })
                    
                    # Top KPI Metrics
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("🟢 Bullish ST Stocks", f"{total_bull}", f"{(total_bull / len(st_records) * 100):.0f}% of Universe" if st_records else "")
                    kpi2.metric("🔴 Bearish ST Stocks", f"{total_bear}", f"{(total_bear / len(st_records) * 100):.0f}% of Universe" if st_records else "")
                    kpi3.metric("🎯 Touching Supertrend", f"{total_touch}", "Pullback Opportunities")
                    kpi4.metric("🔄 Direction Flips Today", f"{total_flip}", "Trend Reversals")
                    
                    # Filter controls
                    f_col1, f_col2 = st.columns([2, 1])
                    with f_col1:
                        st_filter_mode = st.radio(
                            "Filter Radar",
                            options=["All Setups", "🎯 Touching ST (Pullbacks)", "🔄 Direction Flips Today", "🟢 Bullish Only", "🔴 Bearish Only"],
                            horizontal=True,
                            key="st_radar_filter_mode"
                        )
                    with f_col2:
                        sort_by_dist = st.checkbox("Sort by Closest to Supertrend", value=True)

                    df_radar = pd.DataFrame(st_records)
                    if not df_radar.empty:
                        if st_filter_mode == "🎯 Touching ST (Pullbacks)":
                            df_radar = df_radar[df_radar["is_touch"] == True]
                        elif st_filter_mode == "🔄 Direction Flips Today":
                            df_radar = df_radar[df_radar["is_flip"] == True]
                        elif st_filter_mode == "🟢 Bullish Only":
                            df_radar = df_radar[df_radar["curr_dir"] == 1]
                        elif st_filter_mode == "🔴 Bearish Only":
                            df_radar = df_radar[df_radar["curr_dir"] == -1]

                        if sort_by_dist:
                            df_radar = df_radar.sort_values(by="Dist to ST %", key=lambda x: abs(x), ascending=True)
                        else:
                            df_radar = df_radar.sort_values(by="Chg %", ascending=False)

                        display_cols = ["Symbol", "Sector", "Direction", "LTP (₹)", "Chg %", "Supertrend (₹)", "Dist to ST %", "Setup"]
                        st.dataframe(
                            df_radar[display_cols].style.format({
                                "LTP (₹)": "{:.2f}",
                                "Chg %": "{:+.2f}%",
                                "Supertrend (₹)": "{:.2f}",
                                "Dist to ST %": "{:+.2f}%"
                            }).map(
                                lambda v: "color: #22c55e; font-weight:700" if "BULLISH" in str(v) else "color: #ef4444; font-weight:700" if "BEARISH" in str(v) else "",
                                subset=["Direction"]
                            ).map(
                                lambda v: "background-color: rgba(34, 197, 94, 0.2); font-weight:700; color:#22c55e;" if "Bullish Flip" in str(v) else "background-color: rgba(239, 68, 68, 0.2); font-weight:700; color:#ef4444;" if "Bearish Flip" in str(v) else "background-color: rgba(59, 130, 246, 0.2); font-weight:700; color:#60a5fa;" if "At Support" in str(v) else "background-color: rgba(245, 158, 11, 0.2); font-weight:700; color:#fbbf24;" if "At Resistance" in str(v) else "",
                                subset=["Setup"]
                            ),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Interactive Stock Inspector
                        selected_st_sym = st.selectbox(
                            "📈 View Daily Supertrend Chart for Stock:",
                            options=df_radar["_symbol"].tolist(),
                            format_func=lambda x: f"{x.replace('NSE:', '').replace('-EQ', '')} (ST: ₹{df_radar[df_radar['_symbol'] == x]['Supertrend (₹)'].iloc[0]:.1f})",
                            key="st_radar_stock_select"
                        )

                        if selected_st_sym:
                            stk_df = df_daily_st[df_daily_st["symbol"] == selected_st_sym].sort_values("timestamp")
                            if len(stk_df) >= 10:
                                stk_calc = calculate_supertrend(stk_df, period=10, multiplier=2)
                                
                                import plotly.graph_objects as go
                                fig_st = go.Figure()
                                fig_st.add_trace(go.Candlestick(
                                    x=stk_calc["timestamp"],
                                    open=stk_calc["open"],
                                    high=stk_calc["high"],
                                    low=stk_calc["low"],
                                    close=stk_calc["close"],
                                    name="Daily Candles"
                                ))
                                fig_st.add_trace(go.Scatter(
                                    x=stk_calc["timestamp"],
                                    y=stk_calc["supertrend"],
                                    mode="lines",
                                    line=dict(color="#22c55e", width=2, dash="dot"),
                                    name="Daily Supertrend (10, 2)"
                                ))

                                fig_st.update_layout(
                                    title=f"<b>{selected_st_sym.replace('NSE:', '').replace('-EQ', '')}</b> — Daily Supertrend Support/Resistance",
                                    template="plotly_dark",
                                    xaxis_rangeslider_visible=False,
                                    height=420,
                                    margin=dict(l=20, r=20, t=40, b=20),
                                )
                                st.plotly_chart(fig_st, use_container_width=True)
                else:
                    st.info("No daily candle data found for the selected date.")
            except Exception as st_err:
                st.error(f"Error loading Supertrend Radar: {st_err}")

    # ---- TAB: Multi-Touch Proximity (KEI Pattern Radar) ----
    with tab_proximity:
        st.markdown("""
        <div class="section-header" style="margin-top: 0.2rem; margin-bottom: 0.8rem;">
            <h3>🎯 Multi-Touch Support & Resistance Proximity Radar</h3>
            <span class="badge" style="background: rgba(124, 58, 237, 0.2); color: #c084fc; border: 1px solid rgba(124, 58, 237, 0.35);">
                KEI PATTERN • BREAKDOWN FLOORS & BREAKOUT CEILINGS
            </span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ How does the Multi-Touch Proximity Radar (KEI Pattern) work?", expanded=False):
            st.markdown("""
            **Pattern Concept (Observed in stocks like KEI):**
            - **Liquidity Absorption:** When a stock tests a horizontal level $\ge 3$ times, the orders at that level get consumed.
            - **Descending/Ascending Squeeze:** In a breakdown setup, each bounce produces lower highs, squeezing price into support. In a breakout, higher lows compress into resistance.
            - **Explosive Expansion:** When the level gives way, price breaks out or breaks down with sharp momentum.
            - **Scanner Purpose:** Pinpoints stocks **right before** or **at the exact point** of breakdown/breakout so you can capture high R:R setups with defined risk.
            """)

        # Control filters
        col_f1, col_f2, col_f3, col_f4 = st.columns([1.2, 1.2, 1.2, 1.5])
        with col_f1:
            prox_lookback = st.slider("Lookback (Days)", min_value=15, max_value=60, value=30, step=5, key="prox_lookback")
        with col_f2:
            prox_min_touches = st.slider("Min Tests (Touches)", min_value=2, max_value=6, value=3, step=1, key="prox_touches")
        with col_f3:
            prox_max_dist = st.slider("Max Distance %", min_value=0.5, max_value=5.0, value=2.5, step=0.5, key="prox_dist")
        with col_f4:
            prox_type_filter = st.selectbox(
                "Filter Category",
                ["All Setups", "🚨 At Trigger Point (<= 0.5%)", "📉 Breakdown Floor Setups", "🚀 Breakout Ceiling Setups", "⚡ Just Broken Down/Out"],
                index=0,
                key="prox_category_filter"
            )

        with st.spinner("Analyzing F&O stocks for multi-touch levels..."):
            engine = get_engine()
            try:
                with engine.connect() as conn:
                    df_prox_raw = pd.read_sql(text("""
                        SELECT symbol, timestamp, open, high, low, close, volume 
                        FROM ohlcv_daily 
                        WHERE timestamp >= date(:target_date, '-90 days') AND timestamp <= :target_date
                        ORDER BY symbol, timestamp ASC
                    """), conn, params={"target_date": target_date.isoformat()})
                df_prox_raw['timestamp'] = pd.to_datetime(df_prox_raw['timestamp'], format="mixed", errors="coerce")
                df_prox_raw = df_prox_raw.dropna(subset=['timestamp'])
            except Exception as prox_e:
                st.error(f"Error querying database for proximity radar: {prox_e}")
                df_prox_raw = pd.DataFrame()

        if df_prox_raw.empty:
            st.info("No daily candle data available to scan for proximity setups.")
        else:
            symbols_data = {sym: grp for sym, grp in df_prox_raw.groupby('symbol')}
            screener = BreakoutBreakdownProximityScreener(
                lookback_days=prox_lookback,
                cluster_tolerance_pct=0.012,
                min_touches=prox_min_touches,
                max_proximity_pct=prox_max_dist,
            )
            scan_out = screener.scan_universe(symbols_data)
            all_breakdowns = scan_out["breakdown_setups"]
            all_breakouts = scan_out["breakout_setups"]

            # Filter setups based on category
            if prox_type_filter == "🚨 At Trigger Point (<= 0.5%)":
                bdowns = [s for s in all_breakdowns if abs(s.distance_pct) <= 0.5]
                bouts = [s for s in all_breakouts if abs(s.distance_pct) <= 0.5]
            elif prox_type_filter == "📉 Breakdown Floor Setups":
                bdowns = all_breakdowns
                bouts = []
            elif prox_type_filter == "🚀 Breakout Ceiling Setups":
                bdowns = []
                bouts = all_breakouts
            elif prox_type_filter == "⚡ Just Broken Down/Out":
                bdowns = [s for s in all_breakdowns if s.setup_type == "JUST_BROKEN_DOWN"]
                bouts = [s for s in all_breakouts if s.setup_type == "JUST_BROKEN_OUT"]
            else:
                bdowns = all_breakdowns
                bouts = all_breakouts

            # Top KPI metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("🚨 Breakdown Floor Tests", len(all_breakdowns), help="Stocks testing horizontal support >= 3 times")
            with kpi2:
                st.metric("🚀 Breakout Ceiling Tests", len(all_breakouts), help="Stocks testing horizontal resistance >= 3 times")
            with kpi3:
                at_trigger_count = sum(1 for s in all_breakdowns + all_breakouts if abs(s.distance_pct) <= 0.5)
                st.metric("⚡ At Trigger Level (<0.5%)", at_trigger_count, help="Stocks currently within 0.5% of the level")
            with kpi4:
                high_comp_count = sum(1 for s in all_breakdowns + all_breakouts if s.atr_ratio < 0.85)
                st.metric("📦 High Volatility Squeeze", high_comp_count, help="5d ATR < 0.85x 20d ATR")

            st.markdown("<br>", unsafe_allow_html=True)

            # Tabbed View for Breakdowns vs Breakouts
            col_bdown_tab, col_bout_tab = st.tabs([
                f"🚨 Breakdown Candidates ({len(bdowns)})",
                f"🚀 Breakout Candidates ({len(bouts)})"
            ])

            with col_bdown_tab:
                if not bdowns:
                    st.info("No breakdown setups matching the filter.")
                else:
                    bdown_df = pd.DataFrame([s.to_dict() for s in bdowns])
                    display_cols = ["clean_symbol", "sector", "current_price", "key_level", "distance_pct", "touch_count", "compression_score", "atr_ratio", "setup_type"]
                    renamed_df = bdown_df[display_cols].rename(columns={
                        "clean_symbol": "Symbol",
                        "sector": "Sector",
                        "current_price": "LTP",
                        "key_level": "Support Floor",
                        "distance_pct": "Dist %",
                        "touch_count": "Touches",
                        "compression_score": "Score",
                        "atr_ratio": "ATR Ratio",
                        "setup_type": "Status"
                    })
                    st.dataframe(
                        renamed_df.style.format({
                            "LTP": "₹{:.2f}",
                            "Support Floor": "₹{:.2f}",
                            "Dist %": "{:+.2f}%",
                            "Touches": "{:d} tests",
                            "Score": "{:.0f}/100",
                            "ATR Ratio": "{:.2f}x"
                        }).map(
                            lambda v: "background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight:700" if isinstance(v, (int, float)) and abs(v) <= 0.5 else "color: #f87171",
                            subset=["Dist %"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                        key="prox_breakdown_table"
                    )

            with col_bout_tab:
                if not bouts:
                    st.info("No breakout setups matching the filter.")
                else:
                    bout_df = pd.DataFrame([s.to_dict() for s in bouts])
                    display_cols = ["clean_symbol", "sector", "current_price", "key_level", "distance_pct", "touch_count", "compression_score", "atr_ratio", "setup_type"]
                    renamed_bout_df = bout_df[display_cols].rename(columns={
                        "clean_symbol": "Symbol",
                        "sector": "Sector",
                        "current_price": "LTP",
                        "key_level": "Resistance Ceiling",
                        "distance_pct": "Dist %",
                        "touch_count": "Touches",
                        "compression_score": "Score",
                        "atr_ratio": "ATR Ratio",
                        "setup_type": "Status"
                    })
                    st.dataframe(
                        renamed_bout_df.style.format({
                            "LTP": "₹{:.2f}",
                            "Resistance Ceiling": "₹{:.2f}",
                            "Dist %": "{:+.2f}%",
                            "Touches": "{:d} tests",
                            "Score": "{:.0f}/100",
                            "ATR Ratio": "{:.2f}x"
                        }).map(
                            lambda v: "background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight:700" if isinstance(v, (int, float)) and abs(v) <= 0.5 else "color: #4ade80",
                            subset=["Dist %"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                        key="prox_breakout_table"
                    )

            # Interactive Chart Inspector
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.subheader("🔬 Multi-Touch Level Chart Inspector")

            combined_setups = bdowns + bouts
            if combined_setups:
                setup_map = {f"{s.clean_symbol} ({s.sector}) — {s.setup_type} | Level: ₹{s.key_level:.1f} ({s.touch_count} tests)": s for s in combined_setups}
                selected_label = st.selectbox(
                    "Select Stock to Inspect Multi-Touch S/R Level & Compression:",
                    list(setup_map.keys()),
                    index=0,
                    key="prox_chart_selector"
                )
                active_setup = setup_map[selected_label]
                sym_raw = active_setup.symbol
                df_chart = symbols_data.get(sym_raw, pd.DataFrame()).copy()

                if not df_chart.empty:
                    df_chart = df_chart.sort_values("timestamp").reset_index(drop=True)
                    df_sub = df_chart.iloc[-prox_lookback:].copy()

                    # Technical indicators for chart
                    df_sub["sma20"] = df_sub["close"].rolling(20, min_periods=5).mean()
                    df_sub["std20"] = df_sub["close"].rolling(20, min_periods=5).std()
                    df_sub["bb_upper"] = df_sub["sma20"] + 2.0 * df_sub["std20"]
                    df_sub["bb_lower"] = df_sub["sma20"] - 2.0 * df_sub["std20"]

                    from plotly.subplots import make_subplots
                    fig_prox = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.75, 0.25],
                        subplot_titles=(
                            f"<b>{active_setup.clean_symbol}</b> — {active_setup.setup_type} (Key Level: ₹{active_setup.key_level:.2f})",
                            "Volume"
                        )
                    )

                    # Candlestick Trace
                    fig_prox.add_trace(go.Candlestick(
                        x=df_sub["timestamp"],
                        open=df_sub["open"],
                        high=df_sub["high"],
                        low=df_sub["low"],
                        close=df_sub["close"],
                        name="Daily OHLC",
                        increasing_line_color="#22c55e",
                        decreasing_line_color="#ef4444",
                    ), row=1, col=1)

                    # Horizontal Key Level Line
                    level_color = "#ef4444" if active_setup.direction == "BEARISH" else "#22c55e"
                    level_name = f"Support Floor (₹{active_setup.key_level:.2f})" if active_setup.direction == "BEARISH" else f"Resistance Ceiling (₹{active_setup.key_level:.2f})"

                    fig_prox.add_hline(
                        y=active_setup.key_level,
                        line_dash="dash",
                        line_color=level_color,
                        line_width=2.5,
                        annotation_text=f"🎯 {level_name} [{active_setup.touch_count} tests]",
                        annotation_position="top left",
                        row=1, col=1,
                    )

                    # Bollinger Bands
                    fig_prox.add_trace(go.Scatter(
                        x=df_sub["timestamp"], y=df_sub["bb_upper"],
                        line=dict(color="rgba(148, 163, 184, 0.3)", width=1),
                        name="BB Upper", showlegend=False
                    ), row=1, col=1)
                    fig_prox.add_trace(go.Scatter(
                        x=df_sub["timestamp"], y=df_sub["bb_lower"],
                        line=dict(color="rgba(148, 163, 184, 0.3)", width=1),
                        fill="tonexty", fillcolor="rgba(148, 163, 184, 0.05)",
                        name="BB Lower", showlegend=False
                    ), row=1, col=1)

                    # Volume Bar
                    vol_colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df_sub["close"], df_sub["open"])]
                    fig_prox.add_trace(go.Bar(
                        x=df_sub["timestamp"], y=df_sub["volume"],
                        marker_color=vol_colors, name="Volume", showlegend=False
                    ), row=2, col=1)

                    fig_prox.update_layout(
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        height=520,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_prox, use_container_width=True)

                    # Forensics Callout
                    st.info(f"💡 **Forensics Summary:** {active_setup.analysis_summary} | Compression Score: **{active_setup.compression_score:.0f}/100** | 5d/20d ATR Ratio: **{active_setup.atr_ratio:.2f}x**")

    # ---- TAB: F&O Stock PCR Radar (Overbought / Oversold) ----
    with tab_pcr:
        st.markdown("""
        <div class="section-header" style="margin-top: 0.2rem; margin-bottom: 0.8rem;">
            <h3>🎲 F&O Stock Put-Call Ratio (PCR) & Overbought / Oversold Radar</h3>
            <span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35);">
                LIVE OPTION CHAINS • CONTRARIAN SHORT SQUEEZES & OVERBOUGHT RISKS
            </span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ How does Stock Option PCR & Overbought/Oversold Detection work?", expanded=False):
            st.markdown("""
            **Understanding Equity Option PCR (Put-Call Ratio):**
            - **Stock Options vs Indices:** Unlike Index Nifty/BankNifty where PCR ranges 0.60–1.60, individual stock options naturally have higher institutional Call writing. 
            - **🟢 OVERSOLD (PCR $\le 0.55$):** Heavy Call writing builds thick overhead resistance. However, if the stock bases or reverses upward, call writers are trapped and forced to cover, fueling explosive **Short Squeeze Rallies**.
            - **🔴 OVERBOUGHT (PCR $\ge 0.85$):** Heavy Put writing reflects euphoric long positioning. If key support breaks, long liquidation triggers sharp multi-day declines.
            - **🎯 Max Pain:** The strike price where option sellers experience minimum aggregate loss at expiry.
            """)

        # Controls & Sliders
        col_p1, col_p2, col_p3, col_p4 = st.columns([1.2, 1.2, 1.4, 1.2])
        with col_p1:
            pcr_ob_thresh = st.slider("Overbought PCR (>=)", min_value=0.70, max_value=1.50, value=0.85, step=0.05, key="pcr_ob_thresh")
        with col_p2:
            pcr_os_thresh = st.slider("Oversold PCR (<=)", min_value=0.30, max_value=0.70, value=0.55, step=0.05, key="pcr_os_thresh")
        with col_p3:
            pcr_filter_cat = st.selectbox(
                "Filter Sentiment",
                ["All Setups", "🟢 Oversold (Short Squeeze Candidates)", "🔴 Overbought (Reversal Risk)", "💎 Extreme Setups Only"],
                index=0,
                key="pcr_filter_cat"
            )
        with col_p4:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            run_pcr_scan = st.button("🔄 Scan Live Option Chains", key="btn_run_pcr_scan", use_container_width=True)

        # Cache/Session State for PCR Scan Data
        if "pcr_scan_results" not in st.session_state or run_pcr_scan:
            with st.spinner("Fetching live option chains & computing PCR across F&O universe..."):
                try:
                    pcr_screener = FOPCRScreener(
                        overbought_threshold=pcr_ob_thresh,
                        oversold_threshold=pcr_os_thresh,
                    )
                    st.session_state["pcr_scan_results"] = pcr_screener.scan_universe_pcr(max_symbols=120)
                except Exception as pcr_exc:
                    st.error(f"Failed to scan F&O option chains: {pcr_exc}")
                    st.session_state["pcr_scan_results"] = {"overbought": [], "oversold": [], "neutral": [], "all": []}

        pcr_data = st.session_state.get("pcr_scan_results", {})
        all_pcr_stocks: List[StockPCRInfo] = pcr_data.get("all", [])

        if not all_pcr_stocks:
            st.info("No option chain data available. Click '🔄 Scan Live Option Chains' to scan.")
        else:
            ob_stocks = [s for s in all_pcr_stocks if s.pcr_oi >= pcr_ob_thresh]
            os_stocks = [s for s in all_pcr_stocks if s.pcr_oi <= pcr_os_thresh]
            extreme_stocks = [s for s in all_pcr_stocks if s.pcr_oi >= 1.00 or s.pcr_oi <= 0.45]
            avg_pcr = np.mean([s.pcr_oi for s in all_pcr_stocks]) if all_pcr_stocks else 0.0

            # KPI Summary
            pk1, pk2, pk3, pk4 = st.columns(4)
            with pk1:
                st.metric("🔴 Overbought Stocks", len(ob_stocks), help=f"PCR >= {pcr_ob_thresh:.2f} (High Put Writing)")
            with pk2:
                st.metric("🟢 Oversold Stocks", len(os_stocks), help=f"PCR <= {pcr_os_thresh:.2f} (High Call Writing / Squeeze Candidates)")
            with pk3:
                st.metric("💎 Extreme Setups", len(extreme_stocks), help="PCR >= 1.00 or PCR <= 0.45")
            with pk4:
                st.metric("⚖️ Average F&O PCR", f"{avg_pcr:.2f}", help="Average Put-Call Ratio across universe")

            st.markdown("<br>", unsafe_allow_html=True)

            # Sub-Tabs for Oversold vs Overbought vs All
            tab_os, tab_ob, tab_all_pcr = st.tabs([
                f"🟢 Oversold / Squeeze Candidates ({len(os_stocks)})",
                f"🔴 Overbought / Reversal Risk ({len(ob_stocks)})",
                f"📋 All F&O Stock PCR ({len(all_pcr_stocks)})"
            ])

            def _render_pcr_dataframe(stock_list: List[StockPCRInfo], key_name: str):
                if not stock_list:
                    st.info("No stocks matching current criteria.")
                    return
                df = pd.DataFrame([s.to_dict() for s in stock_list])
                cols = ["clean_symbol", "sector", "spot_price", "pcr_oi", "pcr_volume", "sentiment_state", "total_call_oi", "total_put_oi", "max_pain_strike", "highest_ce_oi_strike", "highest_pe_oi_strike", "contrarian_bias"]
                renamed = df[cols].rename(columns={
                    "clean_symbol": "Symbol",
                    "sector": "Sector",
                    "spot_price": "Spot LTP",
                    "pcr_oi": "PCR (OI)",
                    "pcr_volume": "PCR (Vol)",
                    "sentiment_state": "Regime",
                    "total_call_oi": "Call OI",
                    "total_put_oi": "Put OI",
                    "max_pain_strike": "Max Pain",
                    "highest_ce_oi_strike": "CE Wall (Res)",
                    "highest_pe_oi_strike": "PE Wall (Supp)",
                    "contrarian_bias": "Contrarian Bias"
                })
                st.dataframe(
                    renamed.style.format({
                        "Spot LTP": "₹{:.2f}",
                        "PCR (OI)": "{:.2f}",
                        "PCR (Vol)": "{:.2f}",
                        "Call OI": "{:,.0f}",
                        "Put OI": "{:,.0f}",
                        "Max Pain": "₹{:.1f}",
                        "CE Wall (Res)": "₹{:.1f}",
                        "PE Wall (Supp)": "₹{:.1f}",
                    }).map(
                        lambda v: "background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight:700" if isinstance(v, (int, float)) and v <= 0.55 else ("background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight:700" if isinstance(v, (int, float)) and v >= 0.85 else ""),
                        subset=["PCR (OI)"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    key=key_name
                )

            with tab_os:
                _render_pcr_dataframe(os_stocks, "pcr_table_os")

            with tab_ob:
                _render_pcr_dataframe(ob_stocks, "pcr_table_ob")

            with tab_all_pcr:
                _render_pcr_dataframe(all_pcr_stocks, "pcr_table_all")

            # Option Chain Strike Distribution Chart Inspector
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.subheader("🔬 F&O Stock Option Chain & OI Distribution Inspector")

            stock_lookup = {f"{s.clean_symbol} ({s.sector}) — PCR: {s.pcr_oi:.2f} | {s.sentiment_state}": s for s in all_pcr_stocks}
            selected_pcr_label = st.selectbox(
                "Select Stock to Inspect Option Chain Strike OI & Max Pain:",
                list(stock_lookup.keys()),
                index=0,
                key="pcr_stock_chart_selector"
            )
            target_stock = stock_lookup[selected_pcr_label]

            # Fetch fresh detailed option chain for selected stock
            with st.spinner(f"Loading strike-by-strike option chain for {target_stock.clean_symbol}..."):
                try:
                    broker = get_broker()
                    res_chain = broker.fyers.optionchain(data={"symbol": target_stock.symbol, "strikecount": 15})
                    if isinstance(res_chain, dict) and res_chain.get("s") == "ok":
                        oc_list = res_chain.get("data", {}).get("optionsChain", [])
                        df_oc = pd.DataFrame(oc_list)
                    else:
                        df_oc = pd.DataFrame()
                except Exception:
                    df_oc = pd.DataFrame()

            if not df_oc.empty and "strike_price" in df_oc.columns:
                ce_df = df_oc[df_oc["option_type"] == "CE"].sort_values("strike_price")
                pe_df = df_oc[df_oc["option_type"] == "PE"].sort_values("strike_price")

                fig_oc = go.Figure()
                fig_oc.add_trace(go.Bar(
                    x=ce_df["strike_price"],
                    y=ce_df["oi"],
                    name="Call OI (Resistance)",
                    marker_color="#ef4444",
                    opacity=0.85
                ))
                fig_oc.add_trace(go.Bar(
                    x=pe_df["strike_price"],
                    y=pe_df["oi"],
                    name="Put OI (Support)",
                    marker_color="#22c55e",
                    opacity=0.85
                ))

                # Add Max Pain vertical line
                fig_oc.add_vline(
                    x=target_stock.max_pain_strike,
                    line_dash="dash",
                    line_color="#f59e0b",
                    line_width=2,
                    annotation_text=f"Max Pain: ₹{target_stock.max_pain_strike:.1f}",
                    annotation_position="top left"
                )

                # Add Spot Price vertical line
                if target_stock.spot_price > 0:
                    fig_oc.add_vline(
                        x=target_stock.spot_price,
                        line_dash="solid",
                        line_color="#38bdf8",
                        line_width=2.5,
                        annotation_text=f"Spot LTP: ₹{target_stock.spot_price:.2f}",
                        annotation_position="top right"
                    )

                fig_oc.update_layout(
                    title=f"<b>{target_stock.clean_symbol}</b> — Strike-by-Strike OI Profile (PCR: {target_stock.pcr_oi:.2f})",
                    barmode="group",
                    template="plotly_dark",
                    xaxis_title="Strike Price (₹)",
                    yaxis_title="Open Interest (Contracts)",
                    height=460,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_oc, use_container_width=True)

                st.info(f"💡 **Institutional Forensics:** {target_stock.analysis_narrative} | CE Resistance Wall: **₹{target_stock.highest_ce_oi_strike:.1f}** | PE Support Wall: **₹{target_stock.highest_pe_oi_strike:.1f}** | Expiry: **{target_stock.nearest_expiry}**")

    # ---- TAB 4: Daily Reversal Radar ----
    with tab_reversal:
        st.markdown("""
        <div class="section-header" style="margin-top: 0.2rem; margin-bottom: 0.8rem;">
            <h3>🔄 Daily Reversal Probability Radar</h3>
            <span class="badge" style="background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.35);">
                INSTITUTIONAL MEAN REVERSIONS • 5-LAYER CONFLUENCE
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ How are Daily Reversal Probabilities calculated?", expanded=False):
            st.markdown("""
            **Institutional Techniques Used to Quantify Daily Reversals (0–100% Score):**
            1. **RSI Divergence (+25 pts)**: Price makes a Lower Low (Bullish) or Higher High (Bearish) while RSI(14) diverges in extreme overbought/oversold territory.
            2. **Forensic Candlestick Rejections (+25 pts)**: Hammer, Shooting Star, Bullish/Bearish Engulfing, or Piercing line with $\ge 50\%$ rejection wick.
            3. **Bollinger Band Mean Reversion (+20 pts)**: Statistical deviation ($\pm 2.0\sigma$) outside bands snapping back inside.
            4. **Volume Climax & Absorption (+15 pts)**: Volume surge $\ge 1.3\times$ 20-day SMA confirming institutional absorption.
            5. **Liquidity Sweeps / Turtle Soup (+15 pts)**: False breakdown/breakout sweeping 20-day stop runs and closing back inside range.
            """)

        try:
            reversal_scanner = DailyReversalScanner(min_probability=40)
            reversal_setups = reversal_scanner.scan(target_date=target_date, min_probability=40)

            # Reversal Filters
            col_rev_f1, col_rev_f2, col_rev_f3 = st.columns([1, 1, 1])
            with col_rev_f1:
                rev_dir_filter = st.selectbox(
                    "Filter Direction",
                    ["All", "🟢 Bullish Only (Calls / Long)", "🔴 Bearish Only (Puts / Short)"],
                    key="rev_dir_filter_tab"
                )
            with col_rev_f2:
                rev_conf_filter = st.selectbox(
                    "Min Confidence",
                    ["All (≥40%)", "⚡ High & Very High (≥65%)", "🔥 Very High Only (≥80%)"],
                    key="rev_conf_filter_tab"
                )
            with col_rev_f3:
                rev_sector_filter = st.selectbox(
                    "Filter Sector",
                    ["All Sectors"] + options,
                    key="rev_sector_filter_tab"
                )

            filtered_reversals = list(reversal_setups)
            if rev_dir_filter == "🟢 Bullish Only (Calls / Long)":
                filtered_reversals = [s for s in filtered_reversals if s.direction == "CALL"]
            elif rev_dir_filter == "🔴 Bearish Only (Puts / Short)":
                filtered_reversals = [s for s in filtered_reversals if s.direction == "PUT"]

            if rev_conf_filter == "⚡ High & Very High (≥65%)":
                filtered_reversals = [s for s in filtered_reversals if s.probability >= 65]
            elif rev_conf_filter == "🔥 Very High Only (≥80%)":
                filtered_reversals = [s for s in filtered_reversals if s.probability >= 80]

            if rev_sector_filter != "All Sectors":
                filtered_reversals = [s for s in filtered_reversals if s.sector == rev_sector_filter]

            # Top KPIs
            n_total_rev = len(reversal_setups)
            n_bull_rev = sum(1 for s in reversal_setups if s.direction == "CALL")
            n_bear_rev = sum(1 for s in reversal_setups if s.direction == "PUT")
            n_high_conf = sum(1 for s in reversal_setups if s.probability >= 65)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Reversal Candidates", f"{n_total_rev}", help="Stocks showing multi-layer daily reversal confluence")
            k2.metric("🟢 Bullish Calls", f"{n_bull_rev}", delta=f"{n_bull_rev/n_total_rev*100:.0f}% of pool" if n_total_rev > 0 else "0%")
            k3.metric("🔴 Bearish Puts", f"{n_bear_rev}", delta=f"{n_bear_rev/n_total_rev*100:.0f}% of pool" if n_total_rev > 0 else "0%", delta_color="inverse")
            k4.metric("🔥 High Conviction (≥65%)", f"{n_high_conf}")

            if filtered_reversals:
                # Spotlight Top 3 Setups
                st.markdown("#### 🌟 Top Conviction Reversal Spotlights")
                top_spotlights = filtered_reversals[:3]
                spotlight_cols = st.columns(len(top_spotlights))

                for idx, setup in enumerate(top_spotlights):
                    clean_sym = setup.symbol.replace("NSE:", "").replace("-EQ", "")
                    is_bull = (setup.direction == "CALL")
                    theme_color = "#22c55e" if is_bull else "#ef4444"
                    dir_badge = "🟢 BUY CALL / LONG" if is_bull else "🔴 BUY PUT / SHORT"
                    
                    with spotlight_cols[idx]:
                        confluence_chips = "".join([
                            f'<span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); color:#cbd5e1; font-size:0.7rem; padding:2px 6px; border-radius:4px; margin-right:4px; display:inline-block; margin-bottom:4px;">{c}</span>'
                            for c in setup.confluences
                        ])
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, #131426, #1e1b4b); border: 1px solid {theme_color}55; border-radius: 12px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="font-size:1.15rem; font-weight:800; color:#ffffff;">{clean_sym}</span>
                                <span style="background:{theme_color}22; color:{theme_color}; border:1px solid {theme_color}44; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:8px;">{setup.probability}% PROBABILITY</span>
                            </div>
                            <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:8px;">Sector: <b style="color:#e2e8f0;">{setup.sector}</b> • {dir_badge}</div>
                            <div style="margin-bottom:8px;">{confluence_chips}</div>
                            <hr style="margin:8px 0; border-color:rgba(255,255,255,0.08);"/>
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;">
                                <span style="color:#94a3b8;">LTP:</span>
                                <span style="color:#ffffff; font-weight:700;">₹{setup.ltp:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;">
                                <span style="color:#94a3b8;">Stop Loss:</span>
                                <span style="color:#ef4444; font-weight:700;">₹{setup.stop_loss:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;">
                                <span style="color:#94a3b8;">Target 1:</span>
                                <span style="color:#22c55e; font-weight:700;">₹{setup.target_1:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;">
                                <span style="color:#94a3b8;">Target 2:</span>
                                <span style="color:#22c55e; font-weight:700;">₹{setup.target_2:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                                <span style="color:#94a3b8;">Risk-Reward:</span>
                                <span style="color:#eab308; font-weight:700;">{setup.risk_reward:.2f}R</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"📢 Alert {clean_sym}", key=f"rev_tab_alert_{setup.symbol}"):
                            try:
                                settings = Settings.load()
                                notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
                                alert_msg = (
                                    f"🔄 <b>DAILY REVERSAL ALERT: {clean_sym}</b> 🔄\n\n"
                                    f"Direction: <b>{dir_badge}</b>\n"
                                    f"Reversal Probability: <b>{setup.probability}% ({setup.confidence})</b>\n"
                                    f"Sector: <b>{setup.sector}</b>\n"
                                    f"LTP: ₹{setup.ltp:,.2f}\n\n"
                                    f"📐 <b>Levels:</b>\n"
                                    f"• Entry: ₹{setup.ltp:,.2f}\n"
                                    f"• Stop Loss: ₹{setup.stop_loss:,.2f}\n"
                                    f"• Target 1: ₹{setup.target_1:,.2f}\n"
                                    f"• Target 2: ₹{setup.target_2:,.2f}\n"
                                    f"• Risk-Reward: <b>{setup.risk_reward:.2f}R</b>\n\n"
                                    f"💡 <b>Forensics:</b>\n"
                                    f"• " + "\n• ".join(setup.confluences)
                                )
                                notifier.send(alert_msg)
                                st.toast(f"✅ Reversal alert for {clean_sym} sent!")
                            except Exception as al_err:
                                st.error(f"Failed to send alert: {al_err}")

                # Reversal Ranked Universe Table
                st.markdown("#### 📋 Full Reversal Candidate Watchlist")
                rev_table_rows = []
                for s in filtered_reversals:
                    c_sym = s.symbol.replace("NSE:", "").replace("-EQ", "")
                    rev_table_rows.append({
                        "Symbol": c_sym,
                        "Sector": s.sector,
                        "Direction": "🟢 CALL (Long)" if s.direction == "CALL" else "🔴 PUT (Short)",
                        "Prob %": s.probability,
                        "Confidence": s.confidence,
                        "LTP (₹)": s.ltp,
                        "SL (₹)": s.stop_loss,
                        "Target 1 (₹)": s.target_1,
                        "Target 2 (₹)": s.target_2,
                        "R:R": f"{s.risk_reward:.2f}R",
                        "RSI": s.metrics.get("rsi", "—"),
                        "Vol Surge": f"{s.metrics.get('vol_surge', 1.0):.1f}x",
                        "Primary Setup": s.primary_pattern,
                        "Confluences": " • ".join(s.confluences),
                        "_symbol": s.symbol
                    })

                df_rev_display = pd.DataFrame(rev_table_rows)
                st.dataframe(
                    df_rev_display.style.format({
                        "LTP (₹)": "₹{:.2f}",
                        "SL (₹)": "₹{:.2f}",
                        "Target 1 (₹)": "₹{:.2f}",
                        "Target 2 (₹)": "₹{:.2f}",
                        "Prob %": "{}%"
                    }).map(
                        lambda v: "color: #22c55e; font-weight:700" if isinstance(v, (int, float)) and v >= 75 else ("color: #eab308; font-weight:700" if isinstance(v, (int, float)) and v >= 60 else ""),
                        subset=["Prob %"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, len(df_rev_display) * 38 + 38)
                )

                # Interactive Reversal Chart Inspector
                st.markdown("#### 🔬 Daily Reversal Chart Inspector")
                inspector_symbols = [s.symbol for s in filtered_reversals]
                clean_inspector_names = [s.replace("NSE:", "").replace("-EQ", "") for s in inspector_symbols]
                
                selected_inspect_idx = st.selectbox(
                    "Select Stock to Inspect Reversal Forensics",
                    range(len(inspector_symbols)),
                    format_func=lambda i: f"{clean_inspector_names[i]} ({filtered_reversals[i].direction} — {filtered_reversals[i].probability}% Prob)",
                    key="rev_chart_inspect_select"
                )

                inspect_sym = inspector_symbols[selected_inspect_idx]
                inspect_setup = filtered_reversals[selected_inspect_idx]

                # Fetch Daily Candles for this symbol
                engine = get_engine()
                with engine.connect() as conn:
                    query_chart = text("""
                        SELECT timestamp, open, high, low, close, volume 
                        FROM ohlcv_daily 
                        WHERE symbol = :sym AND timestamp >= date(:t_date, '-90 days') AND timestamp <= :t_date
                        ORDER BY timestamp ASC
                    """)
                    df_chart = pd.read_sql(query_chart, conn, params={"sym": inspect_sym, "t_date": target_date.isoformat()})

                if not df_chart.empty:
                    df_chart["timestamp"] = pd.to_datetime(df_chart["timestamp"], format="mixed", errors="coerce")
                    df_chart["sma20"] = df_chart["close"].rolling(20).mean()
                    df_chart["std20"] = df_chart["close"].rolling(20).std()
                    df_chart["upper_bb"] = df_chart["sma20"] + 2.0 * df_chart["std20"]
                    df_chart["lower_bb"] = df_chart["sma20"] - 2.0 * df_chart["std20"]
                    df_chart["rsi"] = _compute_rsi(df_chart["close"], 14)

                    from plotly.subplots import make_subplots
                    fig_rev = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
                    
                    # Candlesticks
                    fig_rev.add_trace(go.Candlestick(
                        x=df_chart["timestamp"],
                        open=df_chart["open"],
                        high=df_chart["high"],
                        low=df_chart["low"],
                        close=df_chart["close"],
                        name=inspect_setup.symbol.replace("NSE:", "").replace("-EQ", "")
                    ), row=1, col=1)

                    # Bollinger Bands & 20 SMA
                    fig_rev.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["upper_bb"], name="Upper BB (2σ)", line=dict(color="rgba(239, 68, 68, 0.5)", width=1, dash="dash")), row=1, col=1)
                    fig_rev.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["sma20"], name="20 SMA", line=dict(color="rgba(250, 204, 21, 0.6)", width=1.2)), row=1, col=1)
                    fig_rev.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["lower_bb"], name="Lower BB (2σ)", line=dict(color="rgba(34, 197, 94, 0.5)", width=1, dash="dash")), row=1, col=1)

                    # Reversal Levels
                    fig_rev.add_hline(y=inspect_setup.stop_loss, line_dash="dash", line_color="#ef4444", annotation_text=f"SL: ₹{inspect_setup.stop_loss:.2f}", row=1, col=1)
                    fig_rev.add_hline(y=inspect_setup.target_1, line_dash="dash", line_color="#22c55e", annotation_text=f"T1: ₹{inspect_setup.target_1:.2f}", row=1, col=1)

                    # RSI Subpane
                    fig_rev.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["rsi"], name="Daily RSI(14)", line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
                    fig_rev.add_hline(y=70, line_dash="dot", line_color="rgba(239, 68, 68, 0.5)", row=2, col=1)
                    fig_rev.add_hline(y=30, line_dash="dot", line_color="rgba(34, 197, 94, 0.5)", row=2, col=1)

                    fig_rev.update_layout(
                        height=460,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_rev, use_container_width=True)

            else:
                st.info("No reversal setups match the current filters. Adjust your minimum probability or sector filter above.")

        except Exception as rev_err:
            LOGGER.error("Error running Daily Reversal Scanner: %s", rev_err, exc_info=True)
            st.warning(f"Unable to compute Daily Reversals: {rev_err}")

    # ---- TAB 5: Sector Drill-Down ----
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
                    df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], format='mixed', errors='coerce')
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
        st.caption("Confluence-based scoring system analyzing 11 layers (Sector, Relative Strength, VWAP, Volume/CVD, Timing, Supertrend, Compression, Daily Trend/Regime, Key Levels, OBV Divergence, and Candle Quality) to find high-probability setups.")
        
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
        with st.spinner(f"Analyzing 11 confluence layers for {selected_horizon} setups..."):
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
                            edge.entry_time = getattr(trig, "trigger_time", "—")
                            
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
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="color:#94a3b8; font-size:0.85rem;">Entry Time:</span>
                                <span style="color:#38bdf8; font-weight:700;">{edge.entry_time if edge.entry_time and edge.entry_time != '—' else ('⚡ Active' if edge.entry_status == 'TRIGGERED' else '⏳ Pending')}</span>
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
                            f"Status: <b>{edge.entry_status}</b>\n"
                            f"Entry Time: <b>{edge.entry_time}</b>\n\n"
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
                
                # Clean entry time formatting
                time_disp = edge.entry_time
                if not time_disp or str(time_disp) in ("—", "None", "nan", "NaT", ""):
                    if edge.entry_status == "TRIGGERED":
                        time_disp = "⚡ Active"
                    elif edge.entry_status == "APPROACHING":
                        time_disp = "🟡 Approaching"
                    else:
                        time_disp = "⏳ Pending"

                table_rows.append({
                    "Symbol": clean_sym,
                    "Sector": edge.sector,
                    "Edge Score": edge.final_score,
                    "Dir": "🟢 CALL" if edge.direction == "CALL" else "🔴 PUT",
                    "Status": edge.entry_status,
                    "Entry Time": time_disp,
                    "Entry (₹)": edge.entry_price if edge.entry_price > 0 else edge.ltp,
                    "SL (₹)": edge.stop_loss if edge.stop_loss > 0 else (round(edge.ltp - edge.atr * 1.0, 2) if edge.direction == "CALL" else round(edge.ltp + edge.atr * 1.0, 2)),
                    "T1 (₹)": edge.target_1 if edge.target_1 > 0 else (round(edge.ltp + edge.atr * 1.5, 2) if edge.direction == "CALL" else round(edge.ltp - edge.atr * 1.5, 2)),
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
                        "Entry (₹)": "{:.2f}",
                        "SL (₹)": "{:.2f}",
                        "T1 (₹)": "{:.2f}",
                        "Chg %": "{:+.2f}%",
                        "Edge Score": "{:.0f}/100"
                    }, na_rep="—").map(
                        lambda v: "color: #22c55e; font-weight:700" if v == "🟢 CALL" else "color: #ef4444; font-weight:700" if v == "🔴 PUT" else "",
                        subset=["Dir"]
                    ).map(
                        lambda v: "color: #22c55e; font-weight:700" if v == "TRIGGERED" else "color: #eab308; font-weight:700" if v == "APPROACHING" else "",
                        subset=["Status"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    selection_mode="single-row",
                    key="edge_finder_table"
                )
                
                # Check for table selection drill down
                selected_row = st.session_state.get("edge_finder_table", {}).get("selection", {}).get("rows", [])
                if selected_row:
                    row_idx = selected_row[0]
                    if 0 <= row_idx < len(enriched_edges):
                        selected_edge = enriched_edges[row_idx]
                        
                        st.markdown(f"### 🔍 Detailed Analysis & Entry Studio: **{selected_edge.symbol.replace('NSE:', '').replace('-EQ', '')}**")
                        
                        sym_df = df_base_trig[df_base_trig["symbol"] == selected_edge.symbol]

                        # ── 1. Momentum Entry Strategies Comparison ──────────────
                        st.markdown("#### 📐 Momentum Entry & Exit Strategies")
                        st.caption("Standard institutional execution frameworks for timing momentum entries with defined Risk-to-Reward.")

                        all_trigs = trigger_eval.evaluate_all(
                            direction=selected_edge.direction,
                            ltp=selected_edge.ltp,
                            atr=selected_edge.atr,
                            df_base=sym_df,
                            target_date=target_date
                        )

                        if all_trigs:
                            strat_rows = []
                            for strat_key, trig in all_trigs.items():
                                strat_name_map = {
                                    "FIBONACCI": "📐 Fibonacci Golden Pocket (50% - 61.8%)",
                                    "VWAP": "🌊 Session / Rolling VWAP Pullback",
                                    "EMA20": "📈 20 EMA Dynamic Trend Support",
                                    "BREAKOUT": "🚀 Range / ORB Breakout",
                                    "SUPERTREND": "🛡️ Supertrend Dynamic Support",
                                }
                                status_badge = "🟢 TRIGGERED" if trig.status == "TRIGGERED" else "🟡 APPROACHING" if trig.status == "APPROACHING" else "⚪ WAITING"
                                risk = abs(trig.entry_price - trig.stop_loss)
                                reward = abs(trig.target_1 - trig.entry_price)
                                rrr = reward / risk if risk > 0 else 1.5
                                
                                strat_rows.append({
                                    "Strategy": strat_name_map.get(strat_key, strat_key),
                                    "Status": status_badge,
                                    "Entry (₹)": trig.entry_price,
                                    "Stop Loss (₹)": trig.stop_loss,
                                    "Target 1 (₹)": trig.target_1,
                                    "Target 2 (₹)": trig.target_2,
                                    "RRR": f"1:{rrr:.1f}",
                                    "Setup Detail": trig.detail,
                                })

                            st.dataframe(
                                pd.DataFrame(strat_rows),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Entry (₹)": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                                    "Stop Loss (₹)": st.column_config.NumberColumn("Stop Loss", format="₹%.2f"),
                                    "Target 1 (₹)": st.column_config.NumberColumn("Target 1", format="₹%.2f"),
                                    "Target 2 (₹)": st.column_config.NumberColumn("Target 2", format="₹%.2f"),
                                }
                            )

                        # ── 2. Fibonacci Retracement & Extension Ladder ───────────
                        from trade_system.domains.strategy.application.strategies.fibonacci_retracement import FibonacciRetracementStrategy
                        fib_strat = FibonacciRetracementStrategy()
                        grid = fib_strat.calculate_grid(sym_df, direction=selected_edge.direction)

                        if grid:
                            st.markdown("#### 🪜 Fibonacci Retracement & Extension Ladder")
                            st.caption(f"Impulse Swing: Low **₹{grid.swing_low:.2f}** ➔ High **₹{grid.swing_high:.2f}** (Range: ₹{grid.impulse_range:.2f})")

                            col_fib1, col_fib2 = st.columns([1, 1.4])
                            with col_fib1:
                                fib_table_data = [
                                    {"Fib Ratio": "-61.8% / 161.8%", "Level (₹)": grid.ext_1618, "Significance": "🎯 Golden Ratio Extension (T3)"},
                                    {"Fib Ratio": "-27.2% / 127.2%", "Level (₹)": grid.ext_1272, "Significance": "🎯 First Trend Extension (T2)"},
                                    {"Fib Ratio": "0.0%", "Level (₹)": grid.level_0, "Significance": "🏁 Swing High / Initial Target (T1)"},
                                    {"Fib Ratio": "23.6%", "Level (₹)": grid.level_236, "Significance": "Shallow Retracement"},
                                    {"Fib Ratio": "38.2%", "Level (₹)": grid.level_382, "Significance": "Momentum Support"},
                                    {"Fib Ratio": "50.0%", "Level (₹)": grid.level_500, "Significance": "⚖️ Equilibrium (Halfway Retracement)"},
                                    {"Fib Ratio": "61.8%", "Level (₹)": grid.level_618, "Significance": "✨ The Golden Pocket Entry"},
                                    {"Fib Ratio": "78.6%", "Level (₹)": grid.level_786, "Significance": "🛑 Deep Discount / Invalidation SL"},
                                    {"Fib Ratio": "100.0%", "Level (₹)": grid.level_1000, "Significance": "Origin Swing Low"},
                                ]
                                df_fib = pd.DataFrame(fib_table_data)
                                st.dataframe(
                                    df_fib,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Level (₹)": st.column_config.NumberColumn("Level", format="₹%.2f"),
                                    }
                                )

                            with col_fib2:
                                # Render visual candlestick chart with Fib levels
                                if not sym_df.empty:
                                    sub_df = sym_df.tail(40).copy()
                                    fig_fib = go.Figure()
                                    fig_fib.add_trace(go.Candlestick(
                                        x=sub_df["timestamp"],
                                        open=sub_df["open"],
                                        high=sub_df["high"],
                                        low=sub_df["low"],
                                        close=sub_df["close"],
                                        name="Price"
                                    ))
                                    # Add horizontal fib lines
                                    colors = {
                                        "0.0%": "#22c55e",
                                        "38.2%": "#38bdf8",
                                        "50.0%": "#eab308",
                                        "61.8%": "#a855f7",
                                        "78.6%": "#ef4444",
                                        "127.2%": "#10b981",
                                    }
                                    for lvl_name, lvl_val in [
                                        ("0.0% (T1)", grid.level_0),
                                        ("50.0% (Eq)", grid.level_500),
                                        ("61.8% (Golden)", grid.level_618),
                                        ("78.6% (SL)", grid.level_786),
                                        ("127.2% (T2)", grid.ext_1272),
                                    ]:
                                        fig_fib.add_hline(
                                            y=lvl_val,
                                            line_dash="dot",
                                            line_color=colors.get(lvl_name.split()[0], "#94a3b8"),
                                            annotation_text=f"{lvl_name}: ₹{lvl_val:.1f}",
                                            annotation_position="bottom right",
                                        )

                                    fig_fib.update_layout(
                                        title=f"Fibonacci Retracement Grid — {selected_edge.symbol.replace('NSE:', '').replace('-EQ', '')}",
                                        height=360,
                                        margin=dict(l=10, r=10, t=35, b=10),
                                        template="plotly_dark",
                                        xaxis_rangeslider_visible=False,
                                    )
                                    st.plotly_chart(fig_fib, use_container_width=True)

                        # ── 3. Confluence Layers Breakdown ───────────────────────
                        st.markdown("#### 🛡️ Confluence Layer Breakdown")
                        for layer_key, result in selected_edge.layers.items():
                            layer_label = layer_key.replace("_", " ").title()
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
                    df_cycle['timestamp'] = pd.to_datetime(df_cycle['timestamp'], format='mixed', errors='coerce')
                    
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
