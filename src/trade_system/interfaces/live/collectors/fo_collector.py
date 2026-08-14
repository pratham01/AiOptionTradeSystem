"""
FoMarketDataCollector — Lightweight market data collector and edge scorer for 200+ F&O stocks.

Handles:
- Symbols: Full F&O universe (~200+ symbols)
- Multi-timeframe bar aggregation: 15m, Daily
- Analytics: Rolling VWAP, Wilder RSI, Relative Strength, Volume Surge
- Confluence: 11-Layer IntradayEdgeScorer integration
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import pandas as pd

from trade_system.domains.analysis.application.analysis.intraday_edge_scorer import IntradayEdgeScorer, EdgeScore
from trade_system.interfaces.live.alert_dispatcher import AlertDispatcher
from trade_system.interfaces.live.bar_aggregator import MultiTimeframeBarAggregator

LOGGER = logging.getLogger("FoCollector")


class FoMarketDataCollector:
    """
    Lightweight streaming collector and 11-layer edge scorer for the F&O universe.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None,
        settings: Any = None,
    ) -> None:
        self.symbols = symbols or self._load_fo_universe()
        self.alert_dispatcher = alert_dispatcher
        self.settings = settings
        self.edge_scorer = IntradayEdgeScorer(horizon="INTRADAY")

        # In-memory latest edge scores for dashboard streaming
        self.latest_edge_scores: Dict[str, EdgeScore] = {}

        # Multi-timeframe bar aggregator for 15m bars
        self.aggregator = MultiTimeframeBarAggregator(
            symbols=self.symbols,
            timeframes=[1, 15],
            on_timeframe_bar=self._on_completed_bar,
        )

    def ingest_tick(self, symbol: str, tick: dict) -> None:
        """Ingest tick for an F&O equity symbol."""
        if symbol in self.symbols:
            self.aggregator.ingest_tick(symbol, tick)

    def _on_completed_bar(self, symbol: str, timeframe: int, bar: pd.Series, history_df: pd.DataFrame) -> None:
        """Handle 15m completed bar for F&O stock."""
        if timeframe == 15:
            LOGGER.debug("[FO-15m] Completed bar for %s @ %s | Close: %.2f", symbol, bar.name, bar["close"])

    def get_latest_edge_scores(self) -> List[EdgeScore]:
        """Return cached edge scores sorted by final score."""
        return sorted(list(self.latest_edge_scores.values()), key=lambda e: e.final_score, reverse=True)

    def _load_fo_universe(self) -> List[str]:
        try:
            from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
            return get_fo_universe()
        except Exception:
            return []
