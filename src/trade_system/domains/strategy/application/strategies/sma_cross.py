from __future__ import annotations

import pandas as pd

from trade_system.domains.trading.domain.models.domain import Signal
from trade_system.domains.strategy.application.strategies.base import BaseStrategy


class SmaCrossStrategy(BaseStrategy):
    name = "sma_cross"

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().sort_values("timestamp").reset_index(drop=True)
        df["fast_sma"] = df["close"].rolling(self.fast).mean()
        df["slow_sma"] = df["close"].rolling(self.slow).mean()
        df["final_signal"] = 0
        
        # Signal logic
        df.loc[df["fast_sma"] > df["slow_sma"], "final_signal"] = 1
        df.loc[df["fast_sma"] < df["slow_sma"], "final_signal"] = -1
        
        # We only want the flip points for messages usually, but for generate_signals we return the state
        return df

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        action = "BUY" if row['final_signal'] == 1 else "SELL"
        return f"SMA Cross | {symbol} | {action} | Price: {row['close']:.2f}"

    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        df = candles.copy().sort_values("timestamp").reset_index(drop=True)
        df["fast_sma"] = df["close"].rolling(self.fast).mean()
        df["slow_sma"] = df["close"].rolling(self.slow).mean()
        return df

    def on_bar(self, window: pd.DataFrame):
        from trade_system.domains.trading.domain.models.legacy import Signal as LegacySignal
        if len(window) < 2:
            return None
        last = window.iloc[-1]
        prev = window.iloc[-2]
        
        # Cross up
        if prev["fast_sma"] <= prev["slow_sma"] and last["fast_sma"] > last["slow_sma"]:
            return LegacySignal(action="BUY", reason="SMA Fast Crosses Over Slow")
            
        # Cross down
        if prev["fast_sma"] >= prev["slow_sma"] and last["fast_sma"] < last["slow_sma"]:
            return LegacySignal(action="SELL", reason="SMA Fast Crosses Under Slow")
            
        return None
