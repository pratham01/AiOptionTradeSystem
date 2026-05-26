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

    async def _check_orb(self, symbol: str, today_df: pd.DataFrame, leaders: set, laggards: set) -> Optional[TradeSuggestion]:
        """Detects 15-minute Opening Range Breakout."""
        try:
            # 1. Define the 15m Range (09:15 to 09:30)
            range_df = today_df.between_time("09:15", "09:30")
            if range_df.empty or len(range_df) < 10: return None
            
            orb_high = range_df['high'].max()
            orb_low = range_df['low'].min()
            current_price = today_df.iloc[-1]['close']
            
            # Avoid repeated alerts
            if symbol in self._opening_ranges and self._opening_ranges[symbol].get('alerted'):
                return None

            sector = FO_METADATA.get(symbol, "Other")
            
            # Bullish ORB
            if current_price > orb_high:
                if sector in leaders or not leaders: # Confirm with sector heat
                    self._thought(f"15M ORB Breakout: {symbol} sector {sector} is leading.", symbol, "ORB")
                    self._opening_ranges[symbol] = {'alerted': True}
                    return self._build_suggestion(symbol, TradeDirection.CALL, current_price, orb_low, "ORB")

            # Bearish ORB
            if current_price < orb_low:
                if sector in laggards or not laggards:
                    self._thought(f"15M ORB Breakdown: {symbol} sector {sector} is lagging.", symbol, "ORB")
                    self._opening_ranges[symbol] = {'alerted': True}
                    return self._build_suggestion(symbol, TradeDirection.PUT, current_price, orb_high, "ORB")

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
