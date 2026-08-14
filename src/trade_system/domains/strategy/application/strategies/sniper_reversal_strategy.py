"""Sniper Reversal Strategy at Supply/Demand Zones."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from trade_system.domains.strategy.application.strategies.base import BaseStrategy, StrategyContext, TradeSignal

LOGGER = logging.getLogger(__name__)


class SniperReversalStrategy(BaseStrategy):
    """
    Sniper Reversal Strategy.
    
    Identifies high-conviction counter-trend exhaustion reversals:
    1. Price penetrates a Key Zone (PDH, PDL, VAH, VAL, Weekly High/Low).
    2. Forms a pronounced rejection wick (>35% of total candle range).
    3. Reverses back inside the zone with immediate confirmation.
    """
    name: str = "sniper_reversal"
    version: str = "2.0"
    supported_timeframes = ["1m", "3m", "5m", "15m"]

    def __init__(self, min_wick_pct: float = 0.35, zone_buffer_pct: float = 0.002, **kwargs) -> None:
        super().__init__(min_wick_pct=min_wick_pct, zone_buffer_pct=zone_buffer_pct, **kwargs)
        self.min_wick_pct = min_wick_pct
        self.zone_buffer_pct = zone_buffer_pct

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        df = data.copy()
        df["signal"] = 0
        return df

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        if context.history_df is None or len(context.history_df) < 2:
            return None

        df = context.history_df.copy()
        last_bar = df.iloc[-1]

        o, h, l, c = float(last_bar["open"]), float(last_bar["high"]), float(last_bar["low"]), float(last_bar["close"])
        candle_range = h - l
        if candle_range <= 0:
            return None

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        upper_wick_pct = upper_wick / candle_range
        lower_wick_pct = lower_wick / candle_range

        zones = context.daily_zones or {}
        pdh = zones.get("PDH", 0.0)
        pdl = zones.get("PDL", 0.0)
        vah = zones.get("VAH", 0.0)
        val = zones.get("VAL", 0.0)

        # Bullish Reversal at Demand Zone (PDL / VAL)
        if lower_wick_pct >= self.min_wick_pct and c > o:
            near_pdl = pdl > 0 and abs(l - pdl) / pdl <= self.zone_buffer_pct
            near_val = val > 0 and abs(l - val) / val <= self.zone_buffer_pct
            if near_pdl or near_val:
                zone_name = "PDL" if near_pdl else "VAL"
                zone_lvl = pdl if near_pdl else val
                sl = l * 0.998
                risk = c - sl
                return TradeSignal(
                    symbol=context.symbol,
                    timestamp=last_bar.name if isinstance(last_bar.name, pd.Timestamp) else pd.Timestamp.now(),
                    direction="CALL",
                    action="BUY_CALL",
                    entry_price=round(c, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(c + (2.0 * risk), 2),
                    target_2=round(c + (3.0 * risk), 2),
                    confidence=0.90,
                    strategy_name=self.name,
                    timeframe=context.timeframe,
                    confluence_factors=[f"Demand Zone Retest ({zone_name})", f"Rejection Wick ({lower_wick_pct:.0%})"],
                    metadata={"zone_name": zone_name, "zone_level": zone_lvl, "wick_pct": lower_wick_pct},
                )

        # Bearish Reversal at Supply Zone (PDH / VAH)
        elif upper_wick_pct >= self.min_wick_pct and c < o:
            near_pdh = pdh > 0 and abs(h - pdh) / pdh <= self.zone_buffer_pct
            near_vah = vah > 0 and abs(h - vah) / vah <= self.zone_buffer_pct
            if near_pdh or near_vah:
                zone_name = "PDH" if near_pdh else "VAH"
                zone_lvl = pdh if near_pdh else vah
                sl = h * 1.002
                risk = sl - c
                return TradeSignal(
                    symbol=context.symbol,
                    timestamp=last_bar.name if isinstance(last_bar.name, pd.Timestamp) else pd.Timestamp.now(),
                    direction="PUT",
                    action="BUY_PUT",
                    entry_price=round(c, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(c - (2.0 * risk), 2),
                    target_2=round(c - (3.0 * risk), 2),
                    confidence=0.90,
                    strategy_name=self.name,
                    timeframe=context.timeframe,
                    confluence_factors=[f"Supply Zone Retest ({zone_name})", f"Rejection Wick ({upper_wick_pct:.0%})"],
                    metadata={"zone_name": zone_name, "zone_level": zone_lvl, "wick_pct": upper_wick_pct},
                )

        return None

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        return f"Sniper Reversal on {symbol} @ ₹{row.get('close', 0):.2f}"
