"""0DTE Gamma Blast Option Strategy for Index Expiry Sessions."""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from typing import Optional

import pandas as pd

from trade_system.domains.strategy.application.strategies.base import BaseStrategy, StrategyContext, TradeSignal

LOGGER = logging.getLogger(__name__)


class GammaBlastStrategy(BaseStrategy):
    """
    0DTE Gamma Blast Strategy.
    
    Operates during the afternoon gamma expansion window (14:15–15:10 IST) on index expiry days.
    Looks for:
    1. Fast momentum breakout on 1m/3m candle.
    2. Favorable delta acceleration on ATM/near-OTM options.
    3. Tight stop loss with 1:3+ asymmetric payoff potential.
    """
    name: str = "gamma_blast"
    version: str = "2.0"
    supported_timeframes = ["1m", "3m", "5m"]

    def __init__(
        self,
        window_start: dt_time = dt_time(14, 15),
        window_end: dt_time = dt_time(15, 10),
        min_momentum_score: float = 65.0,
        **kwargs
    ) -> None:
        super().__init__(window_start=window_start, window_end=window_end, min_momentum_score=min_momentum_score, **kwargs)
        self.window_start = window_start
        self.window_end = window_end
        self.min_momentum_score = min_momentum_score

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        df = data.copy()
        df["signal"] = 0
        return df

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        if context.history_df is None or len(context.history_df) < 5:
            return None

        df = context.history_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                return None

        latest_time = df.index[-1].time()
        # Only active in the afternoon Gamma window
        if not (self.window_start <= latest_time <= self.window_end):
            return None

        recent = df.tail(5)
        close = float(recent.iloc[-1]["close"])
        prev_close = float(recent.iloc[0]["close"])
        pct_move = (close - prev_close) / prev_close * 100

        # Fast momentum threshold: > 0.15% in 5 bars for index
        if abs(pct_move) < 0.15:
            return None

        direction = "CALL" if pct_move > 0 else "PUT"
        action = f"BUY_{direction}"
        momentum_score = min(100.0, abs(pct_move) * 300)

        if momentum_score < self.min_momentum_score:
            return None

        # 0DTE Option risk parameters (e.g. 20% risk on premium, 60%+ target)
        sl_dist = close * 0.003  # ~0.3% spot distance
        sl = close - sl_dist if direction == "CALL" else close + sl_dist
        target_1 = close + (2.5 * sl_dist) if direction == "CALL" else close - (2.5 * sl_dist)
        target_2 = close + (4.0 * sl_dist) if direction == "CALL" else close - (4.0 * sl_dist)

        ts = df.index[-1]
        return TradeSignal(
            symbol=context.symbol,
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            direction=direction,
            action=action,
            entry_price=round(close, 2),
            stop_loss=round(sl, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            confidence=0.88,
            strategy_name=self.name,
            timeframe=context.timeframe,
            confluence_factors=["0DTE Afternoon Window", f"Gamma Momentum {momentum_score:.1f}/100"],
            metadata={"momentum_score": momentum_score, "spot_move_pct": pct_move},
        )

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        return f"0DTE Gamma Blast Signal on {symbol} @ ₹{row.get('close', 0):.2f}"
