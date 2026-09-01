"""
Breakout Screener — Multi-signal intraday breakout detection engine.

Detects three types of setups:
  1. ORB Breakout/Breakdown — Classic opening-range breakout with volume surge.
  2. Momentum Mover — Stocks with significant intraday % change (≥2%) regardless of ORB.
  3. Previous-Day High/Low Breakout — Close exceeding prior session's high or low.

Sector leadership is used as a scoring boost, NOT a hard filter by default.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from datetime import date, datetime
import pandas as pd
from sqlalchemy import text
from typing import Any, Dict, List

from trade_system.domains.market_data.infrastructure.database.connection import get_engine

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

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
            
    def scan_for_breakouts(
        self, 
        top_sectors_count: int = 3,
        target_date: date | None = None,
        use_vwap_filter: bool = True,
        use_wick_filter: bool = True,
        use_index_filter: bool = True,
        use_sector_filter: bool = False,
        current_time: datetime | None = None,
        vol_surge_threshold: float = 1.5,
        **kwargs
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
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
        
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

        # Fetch daily data for all symbols up to 365 days lookback
        query_daily = text("""
            SELECT symbol, timestamp, open, high, low, close, volume 
            FROM ohlcv_daily 
            WHERE timestamp >= date(:target_date, '-365 days') AND timestamp <= :target_date
            ORDER BY symbol, timestamp ASC
        """)
        try:
            with self.engine.connect() as conn:
                df_daily = pd.read_sql(query_daily, conn, params={"target_date": today.isoformat()})
            # Convert timestamp to datetime
            df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], format='mixed')
            daily_groups = {sym: grp.sort_values('timestamp') for sym, grp in df_daily.groupby('symbol')}
        except Exception as e:
            LOGGER.warning(f"Failed to fetch daily candles for 52W/weekly/ST touch filters: {e}")
            daily_groups = {}
            
        # Parse filter settings from kwargs/arguments
        use_52w_filter = kwargs.get("use_52w_filter", False)
        use_weekly_high_filter = kwargs.get("use_weekly_high_filter", False)
        use_daily_st_touch_filter = kwargs.get("use_daily_st_touch_filter", False)
        use_abnormal_vol_filter = kwargs.get("use_abnormal_vol_filter", False)
        
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
        LOGGER.info(f"Lagging Sectors (Short): {lagging_sectors}")        # 3. Load Index Data for sentiment checks
        nifty_candles = df[(df['symbol'] == 'NSE:NIFTY50-INDEX') & (df['timestamp'].dt.date == today)].sort_values('timestamp')
        banknifty_candles = df[(df['symbol'] == 'NSE:NIFTYBANK-INDEX') & (df['timestamp'].dt.date == today)].sort_values('timestamp')
        
        # Load daily consolidation watchlist levels
        consolidation_levels = {}
        try:
            query_consolidation = text("""
                SELECT symbol, resistance, support 
                FROM consolidation_watchlist_stocks 
                WHERE date = (
                    SELECT max(date) 
                    FROM consolidation_watchlist_stocks 
                    WHERE date <= :target_date
                )
            """)
            with self.engine.connect() as conn:
                res = conn.execute(query_consolidation, {"target_date": today.isoformat()}).fetchall()
                for row in res:
                    consolidation_levels[row[0]] = {"resistance": row[1], "support": row[2]}
            LOGGER.info(f"Loaded {len(consolidation_levels)} consolidation watchlists for {today}")
        except Exception as e:
            LOGGER.warning(f"Failed to fetch consolidation watchlists: {e}")

        alerts = []
        
        # 4. Scan F&O universe — prepare per-stock data once
        scan_symbols = df[~df['symbol'].str.contains("INDEX")].copy()
        seen_symbols: set[str] = set()  # Deduplicate across scan types
        
        for symbol, group in scan_symbols.groupby('symbol'):
            group = group.sort_values('timestamp')
            if len(group) < 20:
                continue
                
            # Calculate 20-period Volume SMA and shift it to get the previous candle's volume SMA
            group['vol_sma_20'] = group['volume'].rolling(window=20).mean()
            group['prev_vol_sma'] = group['vol_sma_20'].shift(1)

            # Get today's candles for this stock
            today_candles = group[group['timestamp'].dt.date == today].copy()
            if len(today_candles) < 5:
                continue
                
            # Calculate daily levels
            grp_d = daily_groups.get(symbol)
            is_52w_high = False
            is_weekly_high = False
            touches_daily_st = False
            
            if grp_d is not None and not grp_d.empty:
                latest_close = float(today_candles.iloc[-1]['close'])
                
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
                
                # Daily Supertrend Touch & Direction Flip
                flips_daily_st = False
                st_val = 0.0
                st_direction = None
                try:
                    from trade_system.domains.strategy.application.indicators import calculate_supertrend
                    st_d = calculate_supertrend(grp_d, period=10, multiplier=2)
                    if not st_d.empty:
                        last_st_d = st_d.iloc[-1]
                        prev_st_d = st_d.iloc[-2] if len(st_d) >= 2 else last_st_d
                        st_val = float(last_st_d['supertrend'])
                        curr_dir = int(last_st_d['supertrend_direction'])
                        prev_dir = int(prev_st_d['supertrend_direction'])
                        st_direction = "LONG" if curr_dir == 1 else "SHORT"

                        # Touch: Price pulls back to within 1.0% of the Daily Supertrend line
                        if curr_dir == 1 and float(last_st_d['low']) <= st_val * 1.01 and latest_close >= st_val * 0.99:
                            touches_daily_st = True
                        elif curr_dir == -1 and float(last_st_d['high']) >= st_val * 0.99 and latest_close <= st_val * 1.01:
                            touches_daily_st = True

                        # Flip: Direction changed from previous daily candle
                        if curr_dir != prev_dir:
                            flips_daily_st = True
                except Exception as ex:
                    LOGGER.debug(f"Daily ST check failed for {symbol}: {ex}")
            
            # Apply daily filters
            if use_52w_filter and not is_52w_high:
                continue
            if use_weekly_high_filter and not is_weekly_high:
                continue
            if use_daily_st_touch_filter and not touches_daily_st:
                continue

            # Calculate session VWAP
            cum_pv = (today_candles['close'] * today_candles['volume']).cumsum()
            cum_vol = today_candles['volume'].cumsum()
            today_candles['vwap'] = cum_pv / cum_vol
            
            # Opening Range (First 4 candles of the day = 60 mins for 15m timeframe)
            orb_candles = today_candles.head(4)
            orb_high = orb_candles['high'].max()
            orb_low = orb_candles['low'].min()
            
            # Calculate previous close and previous day high/low once per stock
            prev_close = None
            prev_day_high = None
            prev_day_low = None
            prev_day_candles = group[group['timestamp'].dt.date < today]
            if not prev_day_candles.empty:
                prev_close = prev_day_candles.iloc[-1]['close']
                # Previous day's high/low (last trading day)
                prev_day = prev_day_candles['timestamp'].dt.date.max()
                prev_day_only = prev_day_candles[prev_day_candles['timestamp'].dt.date == prev_day]
                if not prev_day_only.empty:
                    prev_day_high = prev_day_only['high'].max()
                    prev_day_low = prev_day_only['low'].min()
                
            best_alert = None
            # ─────────────────────────────────────────────────────────────────
            # Scan Type 1: ORB Breakout/Breakdown
            # ─────────────────────────────────────────────────────────────────
            for idx in range(4, len(today_candles)):
                candle = today_candles.iloc[idx]
                current_time = candle['timestamp']
                prev_vol_sma = candle['prev_vol_sma']
                
                # Determine direction (Long or Short) based on sector leadership
                if use_sector_filter:
                    is_long = candle['sector'] in leading_sectors
                    is_short = candle['sector'] in lagging_sectors
                else:
                    is_long = True
                    is_short = True
                
                is_bypass = False
                pchange = 0.0
                if prev_close is not None:
                    pchange = ((candle['close'] - prev_close) / prev_close) * 100
                    
                vol_ratio = candle['volume'] / prev_vol_sma if pd.notna(prev_vol_sma) and prev_vol_sma > 0 else 0.0
                
                # Loosen sector constraint for individual breakouts/breakdowns
                # Lowered thresholds: pchange >= 1.5% with vol >= 1.5x, OR pchange >= 3%, OR vol >= 3.0x
                if not is_long and not is_short:
                    if candle['close'] > orb_high:
                        if (pchange >= 1.5 and vol_ratio >= 1.5) or pchange >= 3.0 or vol_ratio >= 3.0:
                            is_long = True
                            is_bypass = True
                    elif candle['close'] < orb_low:
                        if (pchange <= -1.5 and vol_ratio >= 1.5) or pchange <= -3.0 or vol_ratio >= 3.0:
                            is_short = True
                            is_bypass = True
                
                if not is_long and not is_short:
                    continue
                    
                # Check Breakout/Breakdown trigger
                is_triggered = False
                direction = None
                
                if is_long and candle['close'] > orb_high:
                    is_triggered = True
                    direction = "LONG"
                elif is_short and candle['close'] < orb_low:
                    is_triggered = True
                    direction = "SHORT"
                    
                if not is_triggered:
                    continue
                    
                # Filter A: Volume Surge
                is_vol_surge = candle['volume'] > (vol_surge_threshold * prev_vol_sma) if pd.notna(prev_vol_sma) else False
                if not is_vol_surge:
                    continue
                    
                # Filter B: VWAP Alignment
                if use_vwap_filter:
                    vwap = candle['vwap']
                    if direction == "LONG" and candle['close'] <= vwap:
                        continue
                    elif direction == "SHORT" and candle['close'] >= vwap:
                        continue
                        
                # Filter C: Wick Rejection (Solid Candle Body check)
                if use_wick_filter:
                    candle_range = candle['high'] - candle['low']
                    if candle_range > 0:
                        if direction == "LONG":
                            upper_wick = candle['high'] - candle['close']
                            if (upper_wick / candle_range) > 0.35:
                                  continue
                        else:
                            lower_wick = candle['close'] - candle['low']
                            if (lower_wick / candle_range) > 0.35:
                                continue
                                
                # Filter D: Index Sentiment Trend (Bypassed for individual breakouts)
                if use_index_filter and not is_bypass:
                    is_banking_fin = candle['sector'] in ['BANKING_PVT', 'BANKING_PSU', 'FINANCE']
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
                                
                # Record the alert candidate
                alert_candidate = {
                    "symbol": symbol,
                    "sector": candle['sector'],
                    "direction": direction,
                    "close": candle['close'],
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "volume": candle['volume'],
                    "vol_sma": prev_vol_sma,
                    "vwap": candle['vwap'],
                    "is_bypass": is_bypass,
                    "pchange": pchange,
                    "trigger_time": current_time,
                    "alert_type": "ORB_BREAKOUT",
                    "is_52w_high": is_52w_high,
                    "is_weekly_high": is_weekly_high,
                    "touches_daily_st": touches_daily_st
                }
                
                # Keep the candidate with the highest volume surge ratio for this symbol
                if best_alert is None or (alert_candidate['volume'] / alert_candidate['vol_sma'] > best_alert['volume'] / best_alert['vol_sma']):
                    best_alert = alert_candidate
            
            if best_alert:
                alerts.append(best_alert)
                seen_symbols.add(symbol)

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 5: Consolidation Breakout / Breakdown
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols:
                resistance = None
                support = None
                
                # Check DB watchlist levels first
                if symbol in consolidation_levels:
                    resistance = consolidation_levels[symbol]["resistance"]
                    support = consolidation_levels[symbol]["support"]
                else:
                    # Check intraday consolidation on the fly (last 15 bars prior to today)
                    if len(group) >= 16:
                        recent_15 = group.iloc[-16:-1]
                        high_max = recent_15['high'].max()
                        low_min = recent_15['low'].min()
                        range_pct = (high_max - low_min) / low_min * 100
                        if range_pct <= 1.2:
                            resistance = high_max
                            support = low_min
                            
                if resistance is not None and support is not None:
                    min_ratio = vol_surge_threshold if use_abnormal_vol_filter else 1.2
                    for c_idx in range(len(today_candles)):
                        c_bar = today_candles.iloc[c_idx]
                        c_close = float(c_bar['close'])
                        c_vol_sma = c_bar['prev_vol_sma']
                        c_vol_ratio = c_bar['volume'] / c_vol_sma if pd.notna(c_vol_sma) and c_vol_sma > 0 else 0.0
                        c_pchange = ((c_close - prev_close) / prev_close) * 100 if prev_close is not None else 0.0

                        if c_close > resistance and c_vol_ratio >= min_ratio:
                            alerts.append({
                                "symbol": symbol,
                                "sector": c_bar['sector'],
                                "direction": "LONG",
                                "close": c_close,
                                "orb_high": orb_high,
                                "orb_low": orb_low,
                                "volume": c_bar['volume'],
                                "vol_sma": c_vol_sma if pd.notna(c_vol_sma) else 0.0,
                                "vwap": c_bar['vwap'],
                                "is_bypass": False,
                                "pchange": c_pchange,
                                "trigger_time": c_bar['timestamp'],
                                "alert_type": "CONSOLIDATION_BREAKOUT",
                                "resistance": resistance,
                                "support": support,
                                "is_52w_high": is_52w_high,
                                "is_weekly_high": is_weekly_high,
                                "touches_daily_st": touches_daily_st
                            })
                            seen_symbols.add(symbol)
                            break
                        elif c_close < support and c_vol_ratio >= min_ratio:
                            alerts.append({
                                "symbol": symbol,
                                "sector": c_bar['sector'],
                                "direction": "SHORT",
                                "close": c_close,
                                "orb_high": orb_high,
                                "orb_low": orb_low,
                                "volume": c_bar['volume'],
                                "vol_sma": c_vol_sma if pd.notna(c_vol_sma) else 0.0,
                                "vwap": c_bar['vwap'],
                                "is_bypass": False,
                                "pchange": c_pchange,
                                "trigger_time": c_bar['timestamp'],
                                "alert_type": "CONSOLIDATION_BREAKOUT",
                                "resistance": resistance,
                                "support": support,
                                "is_52w_high": is_52w_high,
                                "is_weekly_high": is_weekly_high,
                                "touches_daily_st": touches_daily_st
                            })
                            seen_symbols.add(symbol)
                            break

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 4: Gap Fill (verge of gap filling)
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols and prev_close is not None:
                today_open = float(today_candles.iloc[0]['open'])
                gap_pct = ((today_open - prev_close) / prev_close) * 100
                
                if abs(gap_pct) >= 0.75:
                    for c_idx in range(len(today_candles)):
                        c_bar = today_candles.iloc[c_idx]
                        c_close = float(c_bar['close'])
                        c_pchange = ((c_close - prev_close) / prev_close) * 100
                        
                        if gap_pct >= 0.75:
                            if prev_close < c_close < today_open and (today_open - c_close) >= 0.3 * (today_open - prev_close):
                                alerts.append({
                                    "symbol": symbol,
                                    "sector": c_bar['sector'],
                                    "direction": "SHORT",
                                    "close": c_close,
                                    "orb_high": orb_high,
                                    "orb_low": orb_low,
                                    "volume": c_bar['volume'],
                                    "vol_sma": c_bar['prev_vol_sma'] if pd.notna(c_bar['prev_vol_sma']) else 0.0,
                                    "vwap": c_bar['vwap'],
                                    "is_bypass": False,
                                    "pchange": c_pchange,
                                    "trigger_time": c_bar['timestamp'],
                                    "alert_type": "GAP_FILL",
                                    "gap_pct": gap_pct,
                                    "unfilled_gap_pct": ((c_close - prev_close) / prev_close) * 100,
                                    "is_52w_high": is_52w_high,
                                    "is_weekly_high": is_weekly_high,
                                    "touches_daily_st": touches_daily_st
                                })
                                seen_symbols.add(symbol)
                                break
                        elif gap_pct <= -0.75:
                            if today_open < c_close < prev_close and (c_close - today_open) >= 0.3 * (prev_close - today_open):
                                alerts.append({
                                    "symbol": symbol,
                                    "sector": c_bar['sector'],
                                    "direction": "LONG",
                                    "close": c_close,
                                    "orb_high": orb_high,
                                    "orb_low": orb_low,
                                    "volume": c_bar['volume'],
                                    "vol_sma": c_bar['prev_vol_sma'] if pd.notna(c_bar['prev_vol_sma']) else 0.0,
                                    "vwap": c_bar['vwap'],
                                    "is_bypass": False,
                                    "pchange": c_pchange,
                                    "trigger_time": c_bar['timestamp'],
                                    "alert_type": "GAP_FILL",
                                    "gap_pct": gap_pct,
                                    "unfilled_gap_pct": ((prev_close - c_close) / prev_close) * 100,
                                    "is_52w_high": is_52w_high,
                                    "is_weekly_high": is_weekly_high,
                                    "touches_daily_st": touches_daily_st
                                })
                                seen_symbols.add(symbol)
                                break

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 6: Mean Reversion (Overbought / Oversold Reversals)
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols:
                rsi_series = self._calculate_rsi(group['close'], period=14)
                if not rsi_series.empty:
                    # Align RSI with today's candles
                    today_indices = today_candles.index
                    for c_idx, real_idx in enumerate(today_indices):
                        c_bar = today_candles.iloc[c_idx]
                        c_close = float(c_bar['close'])
                        c_vwap = float(c_bar['vwap'])
                        c_pchange = ((c_close - prev_close) / prev_close) * 100 if prev_close is not None else 0.0
                        rsi_val = float(rsi_series.loc[real_idx]) if real_idx in rsi_series.index else 50.0

                        if rsi_val <= 25 and c_close <= c_vwap * 0.98:
                            alerts.append({
                                "symbol": symbol,
                                "sector": c_bar['sector'],
                                "direction": "LONG",
                                "close": c_close,
                                "orb_high": orb_high,
                                "orb_low": orb_low,
                                "volume": c_bar['volume'],
                                "vol_sma": c_bar['prev_vol_sma'] if pd.notna(c_bar['prev_vol_sma']) else 0.0,
                                "vwap": c_vwap,
                                "is_bypass": False,
                                "pchange": c_pchange,
                                "trigger_time": c_bar['timestamp'],
                                "alert_type": "MEAN_REVERSION",
                                "rsi": rsi_val,
                                "is_52w_high": is_52w_high,
                                "is_weekly_high": is_weekly_high,
                                "touches_daily_st": touches_daily_st
                            })
                            seen_symbols.add(symbol)
                            break
                        elif rsi_val >= 75 and c_close >= c_vwap * 1.02:
                            alerts.append({
                                "symbol": symbol,
                                "sector": c_bar['sector'],
                                "direction": "SHORT",
                                "close": c_close,
                                "orb_high": orb_high,
                                "orb_low": orb_low,
                                "volume": c_bar['volume'],
                                "vol_sma": c_bar['prev_vol_sma'] if pd.notna(c_bar['prev_vol_sma']) else 0.0,
                                "vwap": c_vwap,
                                "is_bypass": False,
                                "pchange": c_pchange,
                                "trigger_time": c_bar['timestamp'],
                                "alert_type": "MEAN_REVERSION",
                                "rsi": rsi_val,
                                "is_52w_high": is_52w_high,
                                "is_weekly_high": is_weekly_high,
                                "touches_daily_st": touches_daily_st
                            })
                            seen_symbols.add(symbol)
                            break

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 7: VWAP Pullback / Trend Support
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols:
                for c_idx in range(len(today_candles)):
                    c_bar = today_candles.iloc[c_idx]
                    c_close = float(c_bar['close'])
                    c_low = float(c_bar['low'])
                    c_high = float(c_bar['high'])
                    c_vwap = float(c_bar['vwap'])
                    c_pchange = ((c_close - prev_close) / prev_close) * 100 if prev_close is not None else 0.0

                    if c_pchange >= 1.2 and c_low <= c_vwap * 1.002 and c_close >= c_vwap:
                        alerts.append({
                            "symbol": symbol,
                            "sector": c_bar['sector'],
                            "direction": "LONG",
                            "close": c_close,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "volume": c_bar['volume'],
                            "vol_sma": c_bar['prev_vol_sma'] if pd.notna(c_bar['prev_vol_sma']) else 0.0,
                            "vwap": c_vwap,
                            "is_bypass": False,
                            "pchange": c_pchange,
                            "trigger_time": c_bar['timestamp'],
                            "alert_type": "VWAP_PULLBACK",
                            "is_52w_high": is_52w_high,
                            "is_weekly_high": is_weekly_high,
                            "touches_daily_st": touches_daily_st
                        })
                        seen_symbols.add(symbol)
                        break
                    elif c_pchange <= -1.2 and c_high >= c_vwap * 0.998 and c_close <= c_vwap:
                        alerts.append({
                            "symbol": symbol,
                            "sector": c_bar['sector'],
                            "direction": "SHORT",
                            "close": c_close,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "volume": c_bar['volume'],
                            "vol_sma": c_bar['prev_vol_sma'] if pd.notna(c_bar['prev_vol_sma']) else 0.0,
                            "vwap": c_vwap,
                            "is_bypass": False,
                            "pchange": c_pchange,
                            "trigger_time": c_bar['timestamp'],
                            "alert_type": "VWAP_PULLBACK",
                            "is_52w_high": is_52w_high,
                            "is_weekly_high": is_weekly_high,
                            "touches_daily_st": touches_daily_st
                        })
                        seen_symbols.add(symbol)
                        break

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 3: Previous-Day High/Low Breakout
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols and prev_close is not None:
                min_ratio = vol_surge_threshold if use_abnormal_vol_filter else 1.2
                for c_idx in range(len(today_candles)):
                    c_bar = today_candles.iloc[c_idx]
                    c_close = float(c_bar['close'])
                    c_vol_sma = c_bar['prev_vol_sma']
                    c_vol_ratio = c_bar['volume'] / c_vol_sma if pd.notna(c_vol_sma) and c_vol_sma > 0 else 0.0
                    c_pchange = ((c_close - prev_close) / prev_close) * 100

                    if prev_day_high is not None and c_close > prev_day_high and c_vol_ratio >= min_ratio:
                        alerts.append({
                            "symbol": symbol,
                            "sector": c_bar['sector'],
                            "direction": "LONG",
                            "close": c_close,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "volume": c_bar['volume'],
                            "vol_sma": c_vol_sma if pd.notna(c_vol_sma) else 0.0,
                            "vwap": c_bar['vwap'],
                            "is_bypass": False,
                            "pchange": c_pchange,
                            "trigger_time": c_bar['timestamp'],
                            "alert_type": "PREV_DAY_HIGH",
                            "is_52w_high": is_52w_high,
                            "is_weekly_high": is_weekly_high,
                            "touches_daily_st": touches_daily_st
                        })
                        seen_symbols.add(symbol)
                        break
                    elif prev_day_low is not None and c_close < prev_day_low and c_vol_ratio >= min_ratio:
                        alerts.append({
                            "symbol": symbol,
                            "sector": c_bar['sector'],
                            "direction": "SHORT",
                            "close": c_close,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "volume": c_bar['volume'],
                            "vol_sma": c_vol_sma if pd.notna(c_vol_sma) else 0.0,
                            "vwap": c_bar['vwap'],
                            "is_bypass": False,
                            "pchange": c_pchange,
                            "trigger_time": c_bar['timestamp'],
                            "alert_type": "PREV_DAY_LOW",
                            "is_52w_high": is_52w_high,
                            "is_weekly_high": is_weekly_high,
                            "touches_daily_st": touches_daily_st
                        })
                        seen_symbols.add(symbol)
                        break

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 2: Momentum Mover (catches big movers without ORB breach)
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols and prev_close is not None:
                min_ratio = vol_surge_threshold if use_abnormal_vol_filter else 1.2
                for c_idx in range(len(today_candles)):
                    c_bar = today_candles.iloc[c_idx]
                    c_close = float(c_bar['close'])
                    c_vwap = float(c_bar['vwap'])
                    c_vol_sma = c_bar['prev_vol_sma']
                    c_vol_ratio = c_bar['volume'] / c_vol_sma if pd.notna(c_vol_sma) and c_vol_sma > 0 else 0.0
                    c_pchange = ((c_close - prev_close) / prev_close) * 100

                    if c_pchange >= 2.0 and c_close > c_vwap and c_vol_ratio >= min_ratio:
                        alerts.append({
                            "symbol": symbol,
                            "sector": c_bar['sector'],
                            "direction": "LONG",
                            "close": c_close,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "volume": c_bar['volume'],
                            "vol_sma": c_vol_sma if pd.notna(c_vol_sma) else 0.0,
                            "vwap": c_vwap,
                            "is_bypass": False,
                            "pchange": c_pchange,
                            "trigger_time": c_bar['timestamp'],
                            "alert_type": "MOMENTUM_MOVER",
                            "is_52w_high": is_52w_high,
                            "is_weekly_high": is_weekly_high,
                            "touches_daily_st": touches_daily_st
                        })
                        seen_symbols.add(symbol)
                        break
                    elif c_pchange <= -2.0 and c_close < c_vwap and c_vol_ratio >= min_ratio:
                        alerts.append({
                            "symbol": symbol,
                            "sector": c_bar['sector'],
                            "direction": "SHORT",
                            "close": c_close,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "volume": c_bar['volume'],
                            "vol_sma": c_vol_sma if pd.notna(c_vol_sma) else 0.0,
                            "vwap": c_vwap,
                            "is_bypass": False,
                            "pchange": c_pchange,
                            "trigger_time": c_bar['timestamp'],
                            "alert_type": "MOMENTUM_MOVER",
                            "is_52w_high": is_52w_high,
                            "is_weekly_high": is_weekly_high,
                            "touches_daily_st": touches_daily_st
                        })
                        seen_symbols.add(symbol)
                        break

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 8: Daily Supertrend Direction Flip (Trend Reversal)
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols and flips_daily_st and st_direction is not None:
                first_candle = today_candles.iloc[0]
                latest_pchange = ((first_candle['close'] - prev_close) / prev_close) * 100 if prev_close is not None else 0.0
                alerts.append({
                    "symbol": symbol,
                    "sector": first_candle['sector'],
                    "direction": st_direction,
                    "close": float(today_candles.iloc[-1]['close']),
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "volume": first_candle['volume'],
                    "vol_sma": first_candle['prev_vol_sma'] if pd.notna(first_candle['prev_vol_sma']) else 0.0,
                    "vwap": first_candle['vwap'],
                    "is_bypass": True,
                    "pchange": latest_pchange,
                    "trigger_time": first_candle['timestamp'],
                    "alert_type": "DAILY_ST_FLIP",
                    "supertrend_level": st_val,
                    "is_52w_high": is_52w_high,
                    "is_weekly_high": is_weekly_high,
                    "touches_daily_st": touches_daily_st
                })
                seen_symbols.add(symbol)

            # ─────────────────────────────────────────────────────────────────
            # Scan Type 9: Daily Supertrend Touch (Pullback Zone)
            # ─────────────────────────────────────────────────────────────────
            if symbol not in seen_symbols and touches_daily_st and st_direction is not None:
                # Find earliest candle touching ST level
                touch_candle = today_candles.iloc[0]
                for c_idx in range(len(today_candles)):
                    c_bar = today_candles.iloc[c_idx]
                    if st_direction == "LONG" and float(c_bar['low']) <= st_val * 1.01:
                        touch_candle = c_bar
                        break
                    elif st_direction == "SHORT" and float(c_bar['high']) >= st_val * 0.99:
                        touch_candle = c_bar
                        break

                latest_pchange = ((touch_candle['close'] - prev_close) / prev_close) * 100 if prev_close is not None else 0.0
                alerts.append({
                    "symbol": symbol,
                    "sector": touch_candle['sector'],
                    "direction": st_direction,
                    "close": float(today_candles.iloc[-1]['close']),
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "volume": touch_candle['volume'],
                    "vol_sma": touch_candle['prev_vol_sma'] if pd.notna(touch_candle['prev_vol_sma']) else 0.0,
                    "vwap": touch_candle['vwap'],
                    "is_bypass": True,
                    "pchange": latest_pchange,
                    "trigger_time": touch_candle['timestamp'],
                    "alert_type": "DAILY_ST_TOUCH",
                    "supertrend_level": st_val,
                    "is_52w_high": is_52w_high,
                    "is_weekly_high": is_weekly_high,
                    "touches_daily_st": True
                })
                seen_symbols.add(symbol)
            
        # Sort alerts: ORB / Flip first, then by volume surge strength
        type_priority = {
            "DAILY_ST_FLIP": 0,
            "DAILY_ST_TOUCH": 1,
            "ORB_BREAKOUT": 2,
            "CONSOLIDATION_BREAKOUT": 3,
            "GAP_FILL": 4,
            "VWAP_PULLBACK": 5,
            "MEAN_REVERSION": 6,
            "PREV_DAY_HIGH": 7,
            "PREV_DAY_LOW": 7,
            "MOMENTUM_MOVER": 8
        }
        alerts.sort(key=lambda x: (
            type_priority.get(x.get('alert_type', 'ORB_BREAKOUT'), 9),
            -(x['volume'] / x['vol_sma'] if x['vol_sma'] > 0 else 0)
        ))
        LOGGER.info(f"Scan complete. Found {len(alerts)} alerts (ORB: {sum(1 for a in alerts if a.get('alert_type') == 'ORB_BREAKOUT')}, ST_Flip: {sum(1 for a in alerts if a.get('alert_type') == 'DAILY_ST_FLIP')}, ST_Touch: {sum(1 for a in alerts if a.get('alert_type') == 'DAILY_ST_TOUCH')}).")
        return alerts

