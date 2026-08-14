"""Supertrend multi-timeframe flip and pullback strategy."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from trade_system.domains.strategy.application.indicators.supertrend import SupertrendIndicator
from trade_system.domains.strategy.application.strategies.base import BaseStrategy, StrategyContext, TradeSignal

LOGGER = logging.getLogger(__name__)


class SupertrendStrategy(BaseStrategy):
    """
    Supertrend crossover & touch strategy.
    
    Generates signals when:
    1. Base timeframe (e.g., 3m/5m) flips direction.
    2. Optional higher-timeframe (15m) filter is aligned.
    3. Price closes on the correct side of the Supertrend line.
    """
    name: str = "supertrend"
    version: str = "2.0"
    supported_timeframes = ["1m", "3m", "5m", "15m"]

    def __init__(self, period: int = 7, multiplier: float = 3.0, require_htf_align: bool = False, **kwargs) -> None:
        super().__init__(period=period, multiplier=multiplier, require_htf_align=require_htf_align, **kwargs)
        self.indicator = SupertrendIndicator(period=period, multiplier=multiplier)
        self.require_htf_align = require_htf_align

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or len(data) < self.indicator.period + 1:
            return pd.DataFrame()
        df = self.indicator.calculate(data)
        if "supertrend_direction" not in df.columns:
            return df

        df["signal"] = 0
        st_dir = df["supertrend_direction"]
        # Signal on direction flip
        df.loc[(st_dir == 1) & (st_dir.shift(1) == -1), "signal"] = 1   # Buy / Call
        df.loc[(st_dir == -1) & (st_dir.shift(1) == 1), "signal"] = -1  # Sell / Put
        return df

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        if context.history_df is None or len(context.history_df) < self.indicator.period + 2:
            return None

        signals_df = self.generate_signals(context.history_df)
        if signals_df.empty:
            return None

        last_row = signals_df.iloc[-1]
        sig = int(last_row.get("signal", 0))
        if sig == 0:
            return None

        close = float(last_row["close"])
        st_val = float(last_row.get("supertrend", close))
        direction = "CALL" if sig == 1 else "PUT"
        action = "BUY_CALL" if sig == 1 else "BUY_PUT"

        # Check Higher Timeframe Alignment if provided
        confluence = ["Supertrend Flip"]
        confidence = 0.85
        if context.htf_df is not None and not context.htf_df.empty:
            htf_calc = self.indicator.calculate(context.htf_df)
            if not htf_calc.empty:
                htf_dir = int(htf_calc.iloc[-1].get("supertrend_direction", 0))
                if (sig == 1 and htf_dir == 1) or (sig == -1 and htf_dir == -1):
                    confluence.append("15m Supertrend Aligned")
                    confidence = 0.95
                elif self.require_htf_align and htf_dir != sig:
                    return None  # Suppress if HTF conflicts

        # Target and SL calculation based on ST level
        risk = max(abs(close - st_val), close * 0.002)
        sl = st_val if direction == "CALL" else st_val
        target_1 = close + (2.0 * risk) if direction == "CALL" else close - (2.0 * risk)
        target_2 = close + (3.5 * risk) if direction == "CALL" else close - (3.5 * risk)

        ts = last_row.name if isinstance(last_row.name, pd.Timestamp) else pd.Timestamp.now()

        return TradeSignal(
            symbol=context.symbol,
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            direction=direction,
            action=action,
            entry_price=round(close, 2),
            stop_loss=round(sl, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            confidence=confidence,
            strategy_name=self.name,
            timeframe=context.timeframe,
            confluence_factors=confluence,
            metadata={"supertrend_value": st_val},
        )

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        sig = row.get("signal", 0)
        direction = "UP (BULLISH)" if sig == 1 else "DOWN (BEARISH)"
        return f"SuperTrend Flip on {symbol}: Direction -> {direction} @ ₹{row.get('close', 0):.2f}"
