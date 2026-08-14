"""11-Layer Confluence Intraday Edge Strategy for F&O Universe."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from trade_system.domains.analysis.application.analysis.intraday_edge_scorer import IntradayEdgeScorer, EdgeScore
from trade_system.domains.strategy.application.strategies.base import BaseStrategy, StrategyContext, TradeSignal

LOGGER = logging.getLogger(__name__)


class IntradayEdgeStrategy(BaseStrategy):
    """
    11-Layer Confluence Intraday Edge Strategy for F&O Stocks.
    
    Evaluates:
    1. Sector Momentum
    2. Relative Strength
    3. VWAP Location
    4. Volume Confirmation
    5. Momentum Timing
    6. Supertrend Alignment
    7. Compression Release
    8. Daily Trend Alignment (Regime Gate + RSI)
    9. Key Level Proximity (PDH/PDL/52W)
    10. OBV Divergence (Smart Money Flow)
    11. Candle Quality & Wick Rejection
    """
    name: str = "intraday_edge"
    version: str = "2.0"
    supported_timeframes = ["15m"]

    def __init__(self, min_score: float = 72.0, horizon: str = "INTRADAY", **kwargs) -> None:
        super().__init__(min_score=min_score, horizon=horizon, **kwargs)
        self.min_score = min_score
        self.scorer = IntradayEdgeScorer(horizon=horizon)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        df = data.copy()
        df["signal"] = 0
        return df

    def evaluate(self, context: StrategyContext) -> Optional[TradeSignal]:
        if context.history_df is None or len(context.history_df) < 5:
            return None

        # Delegate to 11-layer scoring pipeline
        close = float(context.history_df.iloc[-1]["close"])
        return None  # IntradayEdgeScorer.scan() batch interface handles full universe ranking

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        return f"Intraday Edge on {symbol} @ ₹{row.get('close', 0):.2f}"
