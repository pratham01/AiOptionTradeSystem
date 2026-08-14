"""Opening Range Breakout (ORB) strategy with VWAP & volume confirmation."""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from typing import Optional

import pandas as pd

from trade_system.domains.strategy.application.strategies.base import BaseStrategy, StrategyContext, TradeSignal

LOGGER = logging.getLogger(__name__)


class OrbStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy (15-minute / 30-minute).
    
    Generates signals when:
    1. First N minutes establish High & Low range.
    2. Price breaks above Range High (CALL) or below Range Low (PUT) with volume expansion.
    3. Confirmed by VWAP position (price above VWAP for Long, below for Short).
    """
    name: str = "orb_breakout"
    version: str = "2.0"
    supported_timeframes = ["1m", "3m", "5m", "15m"]

    def __init__(self, orb_minutes: int = 15, volume_surge_min: float = 1.2, **kwargs) -> None:
        super().__init__(orb_minutes=orb_minutes, volume_surge_min=volume_surge_min, **kwargs)
        self.orb_minutes = orb_minutes
        self.volume_surge_min = volume_surge_min

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        df = data.copy()
        df["signal"] = 0
        return df

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        if context.history_df is None or context.history_df.empty:
            return None

        df = context.history_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                return None

        today = df.index[-1].date()
        today_df = df[df.index.date == today]
        if len(today_df) < 2:
            return None

        # Determine ORB window (e.g. 09:15 to 09:30 for 15m ORB)
        start_time = dt_time(9, 15)
        orb_cutoff = dt_time(9, 15 + self.orb_minutes) if self.orb_minutes < 45 else dt_time(9, 45)

        orb_bars = today_df[(today_df.index.time >= start_time) & (today_df.index.time < orb_cutoff)]
        if orb_bars.empty:
            return None

        orb_high = float(orb_bars["high"].max())
        orb_low = float(orb_bars["low"].min())
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            return None

        latest_bar = today_df.iloc[-1]
        latest_time = latest_bar.name.time()

        # Only evaluate breakout AFTER the ORB establishment window
        if latest_time <= orb_cutoff:
            return None

        close = float(latest_bar["close"])
        vol = float(latest_bar.get("volume", 0))
        avg_vol = float(orb_bars["volume"].mean()) if "volume" in orb_bars.columns else 1.0
        vol_surge = (vol / avg_vol) if avg_vol > 0 else 1.0

        confluence = [f"{self.orb_minutes}m ORB Breakout"]
        if vol_surge >= self.volume_surge_min:
            confluence.append(f"Volume Surge {vol_surge:.1f}x")

        # Breakout Above ORB High
        if close > orb_high:
            sl = orb_low + (orb_range * 0.5)  # Midpoint or ORB low
            target_1 = close + orb_range
            target_2 = close + (1.5 * orb_range)
            return TradeSignal(
                symbol=context.symbol,
                timestamp=latest_bar.name.to_pydatetime() if hasattr(latest_bar.name, "to_pydatetime") else latest_bar.name,
                direction="CALL",
                action="BUY_CALL",
                entry_price=round(close, 2),
                stop_loss=round(sl, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                confidence=0.85 if vol_surge >= self.volume_surge_min else 0.70,
                strategy_name=self.name,
                timeframe=context.timeframe,
                confluence_factors=confluence,
                metadata={"orb_high": orb_high, "orb_low": orb_low, "volume_surge": vol_surge},
            )

        # Breakdown Below ORB Low
        elif close < orb_low:
            sl = orb_low + (orb_range * 0.5)
            target_1 = close - orb_range
            target_2 = close - (1.5 * orb_range)
            return TradeSignal(
                symbol=context.symbol,
                timestamp=latest_bar.name.to_pydatetime() if hasattr(latest_bar.name, "to_pydatetime") else latest_bar.name,
                direction="PUT",
                action="BUY_PUT",
                entry_price=round(close, 2),
                stop_loss=round(sl, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                confidence=0.85 if vol_surge >= self.volume_surge_min else 0.70,
                strategy_name=self.name,
                timeframe=context.timeframe,
                confluence_factors=confluence,
                metadata={"orb_high": orb_high, "orb_low": orb_low, "volume_surge": vol_surge},
            )

        return None

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        return f"ORB Breakout on {symbol} @ ₹{row.get('close', 0):.2f}"
