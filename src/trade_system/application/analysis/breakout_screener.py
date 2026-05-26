"""
Breakout Screener — Replicates Chartink/Tradefinder Sector Breakout Logic natively.
Supports both Long Breakouts (leading sectors) and Short Breakdowns (lagging sectors) with high-probability filters.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from datetime import date, datetime
import pandas as pd
from sqlalchemy import text
from typing import Any, Dict, List

from trade_system.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)

class BreakoutScreener:
    """
    Screens the F&O universe using local SQLite 15-minute and Daily candles
    to identify Sector Breakouts (Long) and breakdowns (Short) with high-probability filters.
    """
    
    def __init__(self):
        self.engine = get_engine()
        self.fo_metadata = self._load_fo_metadata()
        
    def _load_fo_metadata(self) -> Dict[str, str]:
        config_path = Path("config/fo_universe.json")
        if not config_path.exists():
            return {}
        with open(config_path, "r") as f:
            return json.load(f)
            
    def scan_for_breakouts(
        self, 
        top_sectors_count: int = 3,
        target_date: date | None = None,
        use_vwap_filter: bool = True,
        use_wick_filter: bool = True,
        use_index_filter: bool = True,
        current_time: datetime | None = None
    ) -> List[Dict[str, Any]]:
        LOGGER.info("Starting Intraday Sector Breakout/Breakdown Scan...")
        
        # 1. Fetch recent 15m candles
        if current_time is not None:
            if not isinstance(current_time, str):
                current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                current_time_str = current_time
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM ohlcv_15m 
                WHERE timestamp >= date(:current_time, '-10 days') AND timestamp <= :current_time
                ORDER BY symbol, timestamp ASC
            """)
            params = {"current_time": current_time_str}
        else:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM ohlcv_15m 
                WHERE timestamp >= date('now', '-10 days')
                ORDER BY symbol, timestamp ASC
            """)
            params = {}
            
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
            
        if df.empty:
            LOGGER.warning("No 15m candle data found for breakout scan.")
            return []
            
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resolve target date (today)
        if target_date is not None:
            today = pd.to_datetime(target_date).date()
        else:
            # We filter out index symbols to find the latest stock trade date
            non_idx_df = df[~df['symbol'].str.contains("INDEX")]
            if non_idx_df.empty:
                LOGGER.warning("No stock data found to determine target date.")
                return []
            today = non_idx_df['timestamp'].max().date()
            
        LOGGER.info(f"Scanning data for date: {today}")
        
        # Filter dataframe up to the end of target date
        df = df[df['timestamp'].dt.date <= today]
        
        # Add Sector Info
        df['sector'] = df['symbol'].apply(lambda s: self.fo_metadata.get(s, "UNKNOWN"))
        
        # 2. Calculate Sector Performance
        today_df = df[df['timestamp'].dt.date == today]
        prev_df = df[df['timestamp'].dt.date < today]
        
        if today_df.empty or prev_df.empty:
            LOGGER.warning("Insufficient multi-day data for breakout scan.")
            return []
            
        # Latest closes for stocks today
        latest_closes = today_df[~today_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()
        # Closes for stocks on previous day
        prev_closes = prev_df[~prev_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()
        
        merged_closes = pd.merge(latest_closes, prev_closes, on=['symbol', 'sector'], suffixes=('_last', '_prev'))
        merged_closes['pChange'] = ((merged_closes['close_last'] - merged_closes['close_prev']) / merged_closes['close_prev']) * 100
        
        sector_perf = merged_closes.groupby('sector')['pChange'].mean().reset_index()
        sector_perf = sector_perf[sector_perf['sector'] != 'UNKNOWN']
        
        # Leading Sectors (Long)
        sector_perf_desc = sector_perf.sort_values(by='pChange', ascending=False)
        leading_sectors = sector_perf_desc.head(top_sectors_count)['sector'].tolist()
        
        # Lagging Sectors (Short)
        sector_perf_asc = sector_perf.sort_values(by='pChange', ascending=True)
        lagging_sectors = sector_perf_asc.head(top_sectors_count)['sector'].tolist()
        
        LOGGER.info(f"Leading Sectors (Long): {leading_sectors}")
        LOGGER.info(f"Lagging Sectors (Short): {lagging_sectors}")
        
        # 3. Load Index Data for sentiment checks
        nifty_candles = df[(df['symbol'] == 'NSE:NIFTY50-INDEX') & (df['timestamp'].dt.date == today)].sort_values('timestamp')
        banknifty_candles = df[(df['symbol'] == 'NSE:NIFTYBANK-INDEX') & (df['timestamp'].dt.date == today)].sort_values('timestamp')
        
        alerts = []
        
        # 4. Scan F&O universe
        scan_symbols = df[~df['symbol'].str.contains("INDEX")].copy()
        
        for symbol, group in scan_symbols.groupby('symbol'):
            group = group.sort_values('timestamp')
            if len(group) < 20:
                continue
                
            # Calculate 20-period Volume SMA
            group['vol_sma_20'] = group['volume'].rolling(window=20).mean()
            
            # Get today's candles for this stock
            today_candles = group[group['timestamp'].dt.date == today].copy()
            if len(today_candles) < 5:
                continue
                
            # Calculate session VWAP
            cum_pv = (today_candles['close'] * today_candles['volume']).cumsum()
            cum_vol = today_candles['volume'].cumsum()
            today_candles['vwap'] = cum_pv / cum_vol
            
            # Opening Range (First 4 candles of the day = 60 mins for 15m timeframe)
            orb_candles = today_candles.head(4)
            orb_high = orb_candles['high'].max()
            orb_low = orb_candles['low'].min()
            
            latest = today_candles.iloc[-1]
            current_time = latest['timestamp']
            
            # Get volume SMA from the previous candle
            prev_vol_sma = today_candles.iloc[-2]['vol_sma_20'] if len(today_candles) > 1 else group.iloc[-2]['vol_sma_20']
            
            # Determine direction (Long or Short) based on sector leadership
            is_long = latest['sector'] in leading_sectors
            is_short = latest['sector'] in lagging_sectors
            
            is_bypass = False
            pchange = 0.0
            
            # Calculate previous day's close and percent change
            prev_day_candles = group[group['timestamp'].dt.date < today]
            if not prev_day_candles.empty:
                prev_close = prev_day_candles.iloc[-1]['close']
                pchange = ((latest['close'] - prev_close) / prev_close) * 100
                
            vol_ratio = latest['volume'] / prev_vol_sma if pd.notna(prev_vol_sma) and prev_vol_sma > 0 else 0.0
            
            # Loosen sector constraint for extreme individual breakouts/breakdowns
            if not is_long and not is_short:
                if latest['close'] > orb_high:
                    if (pchange >= 3.0 and vol_ratio >= 2.5) or vol_ratio >= 4.0:
                        is_long = True
                        is_bypass = True
                elif latest['close'] < orb_low:
                    if (pchange <= -3.0 and vol_ratio >= 2.5) or vol_ratio >= 4.0:
                        is_short = True
                        is_bypass = True
            
            if not is_long and not is_short:
                continue
                
            # Check Breakout/Breakdown trigger
            is_triggered = False
            direction = None
            
            if is_long and latest['close'] > orb_high:
                is_triggered = True
                direction = "LONG"
            elif is_short and latest['close'] < orb_low:
                is_triggered = True
                direction = "SHORT"
                
            if not is_triggered:
                continue
                
            # Filter A: Volume Surge
            is_vol_surge = latest['volume'] > (2.0 * prev_vol_sma) if pd.notna(prev_vol_sma) else False
            if not is_vol_surge:
                continue
                
            # Filter B: VWAP Alignment
            if use_vwap_filter:
                vwap = latest['vwap']
                if direction == "LONG" and latest['close'] <= vwap:
                    continue
                elif direction == "SHORT" and latest['close'] >= vwap:
                    continue
                    
            # Filter C: Wick Rejection (Solid Candle Body check)
            if use_wick_filter:
                candle_range = latest['high'] - latest['low']
                if candle_range > 0:
                    if direction == "LONG":
                        upper_wick = latest['high'] - latest['close']
                        # Upper shadow should not exceed 35% of total candle range
                        if (upper_wick / candle_range) > 0.35:
                            continue
                    else:
                        lower_wick = latest['close'] - latest['low']
                        # Lower shadow should not exceed 35% of total candle range
                        if (lower_wick / candle_range) > 0.35:
                            continue
                            
            # Filter D: Index Sentiment Trend (Bypassed for extreme individual breakouts)
            if use_index_filter and not is_bypass:
                is_banking_fin = latest['sector'] in ['BANKING_PVT', 'BANKING_PSU', 'FINANCE']
                idx_candles = banknifty_candles if is_banking_fin else nifty_candles
                
                if not idx_candles.empty:
                    idx_today_t = idx_candles[idx_candles['timestamp'] <= current_time]
                    if not idx_today_t.empty:
                        idx_open = idx_today_t.iloc[0]['open']
                        idx_close = idx_today_t.iloc[-1]['close']
                        if direction == "LONG" and idx_close < idx_open:
                            continue
                        elif direction == "SHORT" and idx_close > idx_open:
                            continue
                            
            # If all checks pass, record alert!
            alerts.append({
                "symbol": symbol,
                "sector": latest['sector'],
                "direction": direction,
                "close": latest['close'],
                "orb_high": orb_high,
                "orb_low": orb_low,
                "volume": latest['volume'],
                "vol_sma": prev_vol_sma,
                "vwap": latest['vwap'],
                "is_bypass": is_bypass,
                "pchange": pchange
            })
            
        # Sort alerts by volume surge strength
        alerts.sort(key=lambda x: x['volume'] / x['vol_sma'] if x['vol_sma'] > 0 else 0, reverse=True)
        LOGGER.info(f"Scan complete. Found {len(alerts)} alerts.")
        return alerts

