"""Strategy Pattern Base Classes, Data Structures, and Lifecycle Contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StrategyContext:
    """Encapsulated market and instrument context passed to a strategy evaluation."""
    symbol: str
    timeframe: str = "15m"
    current_bar: Optional[pd.Series] = None
    history_df: Optional[pd.DataFrame] = None
    htf_df: Optional[pd.DataFrame] = None
    option_chain_df: Optional[pd.DataFrame] = None
    daily_zones: Dict[str, float] = field(default_factory=dict)
    market_regime: int = 0  # 1=bull, -1=bear, 0=neutral
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Standardized trading signal emitted by any strategy."""
    symbol: str
    timestamp: datetime
    direction: str  # "CALL" / "PUT" / "BUY" / "SELL"
    action: str     # "BUY_CALL", "BUY_PUT", "EXIT", "HOLD"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float = 0.0
    confidence: float = 1.0  # 0.0 to 1.0
    strategy_name: str = ""
    timeframe: str = "15m"
    confluence_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def risk_reward_ratio(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk > 0 and self.target_1 > 0:
            reward = abs(self.target_1 - self.entry_price)
            return round(reward / risk, 2)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "direction": self.direction,
            "action": self.action,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "risk_reward_ratio": self.risk_reward_ratio,
            "confluence_factors": self.confluence_factors,
            "metadata": self.metadata,
        }


class BaseStrategy(ABC):
    """
    Abstract base strategy interface conforming to the Strategy Design Pattern.
    """
    name: str = "base_strategy"
    version: str = "1.0"
    supported_timeframes: List[str] = ["1m", "3m", "5m", "15m", "D"]

    def __init__(self, **params: Any) -> None:
        self.params = params

    def initialize(self, symbol: str, timeframe: str) -> None:
        """Lifecycle hook called when attaching strategy to a symbol stream."""
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates buy/sell signals on historical/batch OHLCV data."""
        pass

    @abstractmethod
    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        """Returns a formatted message for the signal."""
        pass

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        """
        Real-time evaluation hook called on live completed bars or ticks.
        Default implementation converts data to dataframe and calls generate_signals.
        """
        if context.history_df is None or context.history_df.empty:
            return None
        signals_df = self.generate_signals(context.history_df)
        if signals_df is None or signals_df.empty:
            return None
        last_row = signals_df.iloc[-1]
        sig_val = last_row.get("signal", 0)
        if sig_val == 0:
            return None

        direction = "CALL" if sig_val > 0 else "PUT"
        close_p = float(last_row.get("close", 0.0))
        return TradeSignal(
            symbol=context.symbol,
            timestamp=last_row.name if isinstance(last_row.name, datetime) else datetime.now(),
            direction=direction,
            action=f"BUY_{direction}",
            entry_price=close_p,
            stop_loss=close_p * (0.99 if direction == "CALL" else 1.01),
            target_1=close_p * (1.02 if direction == "CALL" else 0.98),
            confidence=0.8,
            strategy_name=self.name,
            timeframe=context.timeframe,
        )


class Strategy(ABC):
    """Legacy interface adapter for backward compatibility."""
    @abstractmethod
    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Prepares candle data for the strategy."""
        pass

    @abstractmethod
    def on_bar(self, window: pd.DataFrame):
        """Decides what to do based on the latest candle window."""
        pass
