"""
EarlyMorningAgent — Captures institutional momentum in the first 90 minutes.
Identifies 'Gap & Go' and 'Opening Range Breakout' (ORB) setups.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from typing import Dict, List, Any, Optional

import pandas as pd
from trade_system.core import TradeSuggestion, TradeDirection, TradeHorizon, OptionParams
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.repository import get_market_data, log_agent_thought
from trade_system.application.analysis.sector_rotation import SectorRotationAnalyzer
from trade_system.infrastructure.data.fo_universe import FO_METADATA, get_fo_universe
from sqlalchemy.orm import Session

LOGGER = logging.getLogger(__name__)

class EarlyMorningAgent:
    """
    Scans the F&O universe for early morning institutional setups.
    Active between 09:15 and 10:30 IST.
    """

    def __init__(self, broker: Any = None):
        self.broker = broker
        self.engine = get_engine()
        self.rotation_analyzer = SectorRotationAnalyzer()
        self._opening_ranges: dict[str, dict] = {} # symbol -> {high, low, timestamp}

    def _thought(self, message: str, symbol: str | None = None, action: str | None = None):
        LOGGER.info(f"[EarlyMorningAgent] {message}")
        try:
            with Session(self.engine) as session:
                log_agent_thought(session, "EarlyMorningAgent", message, symbol, action)
        except: pass

    async def scan_for_setups(self, market_data: dict[str, pd.DataFrame]) -> List[TradeSuggestion]:
        """
        Main entry point for early morning scanning.
        market_data: dict of symbol -> 1-minute or 5-minute DataFrames.
        """
        now = datetime.now()
        current_time = now.time()
        
        # Only active between 09:15 and 10:45 IST (allowing some buffer)
        if current_time < dt_time(9, 15) or current_time > dt_time(10, 45):
            return []

        suggestions = []
        
        # 1. Get Sector Leadership for ORB Confirmation
        leadership_df = self.rotation_analyzer.get_sector_leadership(lookback_days=15)
        leading_sectors = set()
        lagging_sectors = set()
        if not leadership_df.empty:
            leading_sectors = set(leadership_df[leadership_df['Status'].str.contains("LEADING")]['Sector'].tolist())
            lagging_sectors = set(leadership_df[leadership_df['Status'].str.contains("LAGGING")]['Sector'].tolist())

        for symbol, df in market_data.items():
            if df.empty: continue
            
            # Ensure we are looking at today's bars
            today_df = df[df.index.date == now.date()]
            if today_df.empty: continue

            # --- SETUP 1: GAP & GO (Check within first 15 mins) ---
            if dt_time(9, 20) <= current_time <= dt_time(9, 45):
                gap_sugg = await self._check_gap_and_go(symbol, today_df)
                if gap_sugg: suggestions.append(gap_sugg)

            # --- SETUP 2: 15-MINUTE ORB (Check after 09:30) ---
            if current_time >= dt_time(9, 30):
                orb_sugg = await self._check_orb(symbol, today_df, leading_sectors, lagging_sectors)
                if orb_sugg: suggestions.append(orb_sugg)

        return suggestions

    async def _check_gap_and_go(self, symbol: str, today_df: pd.DataFrame) -> Optional[TradeSuggestion]:
        """Detects if a stock gapped and is showing immediate continuation."""
        try:
            # We need yesterday's close for gap calculation
            with Session(self.engine) as session:
                prev_day = get_market_data(session, symbol, "D", limit=1)
                if not prev_day: return None
                prev_close = prev_day[0].close

            first_bar = today_df.iloc[0]
            gap_pct = ((first_bar['open'] - prev_close) / prev_close) * 100
            
            # Condition: Gap > 1.0% and < 4.0%
            if abs(gap_pct) < 1.0 or abs(gap_pct) > 4.0: return None
            
            # Confirmation: First 5-min candle (or first few 1-min) direction
            # For simplicity, let's check if current price has broken the first candle's high/low
            first_candle_high = today_df.iloc[:5]['high'].max()
            first_candle_low = today_df.iloc[:5]['low'].min()
            current_price = today_df.iloc[-1]['close']
            
            # Bullish Gap & Go
            if gap_pct > 0 and current_price > first_candle_high:
                self._thought(f"Gap & Go detected (Bullish): {gap_pct:.2f}% gap up with breakout.", symbol, "GAP_AND_GO")
                return self._build_suggestion(symbol, TradeDirection.CALL, current_price, first_candle_low, "GAP_AND_GO")
            
            # Bearish Gap & Go
            if gap_pct < 0 and current_price < first_candle_low:
                self._thought(f"Gap & Go detected (Bearish): {gap_pct:.2f}% gap down with breakdown.", symbol, "GAP_AND_GO")
                return self._build_suggestion(symbol, TradeDirection.PUT, current_price, first_candle_high, "GAP_AND_GO")

        except Exception as e:
            LOGGER.error(f"Error checking Gap & Go for {symbol}: {e}")
        return None

    def _get_nifty_trend(self) -> str:
        """Determines Nifty 50 short-term direction based on 5-period SMA of 1-minute close prices."""
        try:
            with Session(self.engine) as session:
                bars = get_market_data(session, "NSE:NIFTY50-INDEX", "1", limit=10)
                if len(bars) >= 5:
                    closes = [b.close for b in bars]
                    sma_5 = sum(closes[-5:]) / 5.0
                    latest_close = closes[-1]
                    if latest_close > sma_5:
                        return "BULLISH"
                    elif latest_close < sma_5:
                        return "BEARISH"
        except Exception as e:
            LOGGER.error(f"Failed to get Nifty trend: {e}")
        return "NEUTRAL"

    def _calculate_daily_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculates the daily ATR for the given symbol."""
        try:
            with Session(self.engine) as session:
                bars = get_market_data(session, symbol, "D", limit=period + 1)
                if len(bars) < 5:
                    return None
                
                highs = [b.high for b in bars]
                lows = [b.low for b in bars]
                closes = [b.close for b in bars]
                
                trs = []
                for i in range(1, len(bars)):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    )
                    trs.append(tr)
                return sum(trs) / len(trs)
        except Exception as e:
            LOGGER.error(f"Failed to calculate ATR for {symbol}: {e}")
        return None

    def _load_watchlist(self) -> dict[str, dict]:
        """Loads today's priority watchlist if available."""
        try:
            import json
            from pathlib import Path
            watchlist_dir = Path("data/watchlist")
            today_str = datetime.now().strftime("%Y%m%d")
            path = watchlist_dir / f"watchlist_{today_str}.json"
            
            if path.exists():
                data = json.loads(path.read_text())
                return {item["symbol"]: item for item in data}
        except Exception as e:
            LOGGER.error(f"Failed to load watchlist: {e}")
        return {}

    async def _check_orb(self, symbol: str, today_df: pd.DataFrame, leaders: set, laggards: set) -> Optional[TradeSuggestion]:
        """Detects 15-minute Opening Range Breakout or NR7/IB yesterday's bracket breakout with advanced filters."""
        try:
            latest_bar = today_df.iloc[-1]
            current_price = latest_bar['close']
            
            # Check watchlist for compression coiling (NR7 / Inside Bar)
            watchlist = self._load_watchlist()
            watchlist_item = watchlist.get(symbol)
            is_compression_stock = False
            
            if watchlist_item:
                is_compression_stock = any(p in ["NR7 Compression", "Inside Bar"] for p in watchlist_item.get("patterns", []))
                
            pattern_name = "ORB"
            range_df = None
            
            if is_compression_stock:
                with Session(self.engine) as session:
                    daily_bars = get_market_data(session, symbol, "D", limit=1)
                if daily_bars:
                    prev_day = daily_bars[0]
                    orb_high = prev_day.high
                    orb_low = prev_day.low
                    pattern_name = "NR7/IB Bracket"
                else:
                    is_compression_stock = False
                    
            if not is_compression_stock:
                # Fallback to standard 15m range (09:15 to 09:30)
                range_df = today_df.between_time("09:15", "09:30")
                if range_df.empty or len(range_df) < 2: return None
                orb_high = range_df['high'].max()
                orb_low = range_df['low'].min()
            
            # Check if this is a raw breakout candle
            is_breakout_call = current_price > orb_high
            is_breakout_put = current_price < orb_low
            
            if not is_breakout_call and not is_breakout_put:
                return None
                
            # Avoid repeated alerts
            if symbol in self._opening_ranges and self._opening_ranges[symbol].get('alerted'):
                return None

            sector = FO_METADATA.get(symbol, "Other")
            direction = TradeDirection.CALL if is_breakout_call else TradeDirection.PUT
            sl = orb_low if direction == TradeDirection.CALL else orb_high
            risk = abs(current_price - sl)
            target = current_price + (risk * 2) if direction == TradeDirection.CALL else current_price - (risk * 2)
            
            # Apply Filters in order and log rejection reason if any fails
            rejection_reason = None
            
            # Filter 1: Nifty Index Trend Alignment
            nifty_trend = self._get_nifty_trend()
            if direction == TradeDirection.CALL and nifty_trend != "BULLISH":
                rejection_reason = f"Index trend is not BULLISH (current Nifty: {nifty_trend})"
            elif direction == TradeDirection.PUT and nifty_trend != "BEARISH":
                rejection_reason = f"Index trend is not BEARISH (current Nifty: {nifty_trend})"
                
            # Filter 2: ATR Exhaustion Filter (Opening Range size <= 35% ATR)
            if not rejection_reason:
                atr = self._calculate_daily_atr(symbol)
                if atr is not None:
                    range_size = orb_high - orb_low
                    if range_size > 0.35 * atr:
                        rejection_reason = f"Opening range size ({range_size:.2f}) exceeds 35% of daily ATR ({atr:.2f})"
                        
            # Filter 3: Volume Expansion (RVOL >= 1.5x)
            if not rejection_reason:
                if is_compression_stock:
                    avg_vol = today_df['volume'].iloc[:-1].mean() if len(today_df) > 1 else latest_bar['volume']
                else:
                    avg_vol = range_df['volume'].mean() if range_df is not None else latest_bar['volume']
                latest_vol = latest_bar['volume']
                rvol = latest_vol / avg_vol if avg_vol > 0 else 1.0
                if rvol < 1.5:
                    rejection_reason = f"Breakout volume expansion is too low (RVOL: {rvol:.2f}x, required: 1.5x)"
                    
            # Filter 4: VWAP Alignment
            if not rejection_reason:
                temp_df = today_df.copy()
                temp_df['tp'] = (temp_df['high'] + temp_df['low'] + temp_df['close']) / 3.0
                temp_df['tp_vol'] = temp_df['tp'] * temp_df['volume']
                cum_tp_vol = temp_df['tp_vol'].cumsum().iloc[-1]
                cum_vol = temp_df['volume'].cumsum().iloc[-1]
                vwap = cum_tp_vol / cum_vol if cum_vol > 0 else current_price
                if direction == TradeDirection.CALL and current_price <= vwap:
                    rejection_reason = f"Price ({current_price:.2f}) is below VWAP ({vwap:.2f})"
                elif direction == TradeDirection.PUT and current_price >= vwap:
                    rejection_reason = f"Price ({current_price:.2f}) is above VWAP ({vwap:.2f})"
                    
            # Filter 5: Strict Sector Alignment
            if not rejection_reason:
                if direction == TradeDirection.CALL:
                    if leaders and sector not in leaders:
                        rejection_reason = f"Sector {sector} is not leading (Leading: {list(leaders)})"
                else:
                    if laggards and sector not in laggards:
                        rejection_reason = f"Sector {sector} is not lagging (Lagging: {list(laggards)})"

            # Handle Filter Rejection
            if rejection_reason:
                import json
                params = {
                    "type": "ORB_REJECTED",
                    "symbol": symbol,
                    "direction": direction.value,
                    "entry": float(current_price),
                    "sl": float(sl),
                    "target": float(target),
                    "reason": rejection_reason,
                    "timestamp": datetime.now().isoformat()
                }
                self._thought(json.dumps(params), symbol=symbol, action="ORB_REJECTED")
                self._opening_ranges[symbol] = {'alerted': True}
                return None
                
            # If all filters pass, generate and return suggestion
            self._thought(f"15M ORB {'Breakout' if direction == TradeDirection.CALL else 'Breakdown'} Confirmed: {symbol} sector {sector}.", symbol, "ORB")
            self._opening_ranges[symbol] = {'alerted': True}
            return self._build_suggestion(symbol, direction, current_price, sl, "ORB")

        except Exception as e:
            LOGGER.error(f"Error checking ORB for {symbol}: {e}")
        return None

    def _build_suggestion(self, symbol: str, direction: TradeDirection, price: float, sl: float, pattern: str) -> TradeSuggestion:
        import uuid
        risk = abs(price - sl)
        target = price + (risk * 2) if direction == TradeDirection.CALL else price - (risk * 2)
        
        return TradeSuggestion(
            id=str(uuid.uuid4()),
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            horizon=TradeHorizon.INTRADAY,
            entry_zone_low=price * 0.999,
            entry_zone_high=price * 1.001,
            target=round(target, 2),
            stop_loss=round(sl, 2),
            option_params=OptionParams(direction=direction),
            confidence=0.75,
            narrative=f"Institutional {pattern} setup. Morning momentum surge detected.",
            setup_features=None,
            sector=FO_METADATA.get(symbol, "Other"),
            tags=[pattern.lower(), "early_morning"]
        )
