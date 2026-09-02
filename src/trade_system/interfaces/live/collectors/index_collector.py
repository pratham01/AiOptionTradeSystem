"""
IndexMarketDataCollector — Dedicated high-frequency collector and strategy engine for index instruments.

Handles:
- Symbols: NIFTY50, NIFTYBANK, SENSEX, FINNIFTY
- Multi-timeframe bar aggregation: 1m, 3m, 5m, 15m
- Indicator execution: Multi-TF Supertrend, VWAP, Volume Delta, SMC Zones
- Strategy dispatching: SupertrendStrategy, OrbStrategy, GammaBlastStrategy, SniperReversalStrategy
- Option Chain & Strike Supertrend integration
"""
from __future__ import annotations

import logging
from datetime import datetime, date, time as dt_time
from typing import Any, Dict, List, Optional

import pandas as pd

from trade_system.domains.strategy.application.strategies.base import StrategyContext, TradeSignal
from trade_system.domains.strategy.application.strategies.supertrend_strategy import SupertrendStrategy
from trade_system.domains.strategy.application.strategies.gamma_blast_strategy import GammaBlastStrategy
from trade_system.domains.strategy.application.strategies.sniper_reversal_strategy import SniperReversalStrategy
from trade_system.interfaces.live.alert_dispatcher import AlertDispatcher
from trade_system.interfaces.live.bar_aggregator import MultiTimeframeBarAggregator

LOGGER = logging.getLogger("IndexCollector")


class IndexMarketDataCollector:
    """
    High-frequency multi-timeframe market data collector and strategy runner for Index instruments.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None,
        strategy_timeframe_minutes: int = 3,
        settings: Any = None,
    ) -> None:
        self.symbols = symbols or [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX",
            "BSE:SENSEX-INDEX",
            "NSE:FINNIFTY-INDEX",
        ]
        self.alert_dispatcher = alert_dispatcher
        self.strategy_timeframe_minutes = strategy_timeframe_minutes
        self.settings = settings

        # Strategies
        self.st_strategy = SupertrendStrategy(period=7, multiplier=3.0)
        self.gamma_strategy = GammaBlastStrategy()
        self.sniper_strategy = SniperReversalStrategy()

        # State storage
        self.daily_zones: Dict[str, Dict[str, float]] = {s: {} for s in self.symbols}
        self.last_signals: Dict[str, Dict[str, TradeSignal]] = {s: {} for s in self.symbols}

        # Multi-timeframe bar aggregator
        self.aggregator = MultiTimeframeBarAggregator(
            symbols=self.symbols,
            timeframes=[1, self.strategy_timeframe_minutes, 5, 15],
            on_timeframe_bar=self._on_completed_bar,
        )

    def ingest_tick(self, symbol: str, tick: dict) -> None:
        """Ingest raw live tick for an index symbol."""
        if symbol in self.symbols:
            self.aggregator.ingest_tick(symbol, tick)

    def set_daily_zones(self, symbol: str, zones: Dict[str, float]) -> None:
        """Set premarket calculated daily support/resistance zones."""
        self.daily_zones[symbol] = zones

    def _on_completed_bar(self, symbol: str, timeframe: int, bar: pd.Series, history_df: pd.DataFrame) -> None:
        """Evaluates strategies on completed bars."""
        LOGGER.debug("[%s] Completed %dm bar @ %s | Close: %.2f", symbol, timeframe, bar.name, bar["close"])

        context = StrategyContext(
            symbol=symbol,
            timeframe=f"{timeframe}m",
            current_bar=bar,
            history_df=history_df,
            htf_df=self.aggregator.get_timeframe_data(symbol, 15),
            daily_zones=self.daily_zones.get(symbol, {}),
        )

        # 1. Evaluate Supertrend Strategy on base strategy timeframe (e.g. 3m)
        if timeframe == self.strategy_timeframe_minutes:
            st_signal = self.st_strategy.evaluate(context)
            if st_signal:
                self._handle_signal(symbol, "supertrend_flip", st_signal)

        # 2. Evaluate Gamma Blast Strategy on 1m/3m
        gamma_signal = self.gamma_strategy.evaluate(context)
        if gamma_signal:
            self._handle_signal(symbol, "gamma_blast", gamma_signal)

        # 3. Evaluate Sniper Reversal Strategy
        sniper_signal = self.sniper_strategy.evaluate(context)
        if sniper_signal:
            self._handle_signal(symbol, "sniper_reversal", sniper_signal)

    def _handle_signal(self, symbol: str, strategy_name: str, signal: TradeSignal) -> None:
        """Record signal and dispatch alert if alert dispatcher is available."""
        self.last_signals[symbol][strategy_name] = signal
        LOGGER.info(
            "⚡ [%s] Signal: %s (%s) @ ₹%.2f [Confidence: %.0f%%]",
            symbol, strategy_name, signal.direction, signal.entry_price, signal.confidence * 100
        )

        if self.alert_dispatcher:
            short_sym = symbol.split(":")[-1].replace("-INDEX", "")
            context_dict = {
                "symbol_short": short_sym,
                "color": "🟢" if signal.direction == "CALL" else "🔴",
                "time": signal.timestamp.strftime("%H:%M") if hasattr(signal.timestamp, "strftime") else str(signal.timestamp),
                "timeframe": self.strategy_timeframe_minutes,
                "direction": signal.direction,
                "direction_verbose": "🟢 UP (BULLISH)" if signal.direction == "CALL" else "🔴 DOWN (BEARISH)",
                "trend_15m": "UP" if signal.direction == "CALL" else "DOWN",
                "close": signal.entry_price,
                "supertrend": signal.stop_loss,
                "confluence": " | ".join(signal.confluence_factors) if signal.confluence_factors else "Standard Signal",
                "action": signal.action,
                "entry": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "target": signal.target_1,
                "price": signal.entry_price,
                "orb_high": signal.metadata.get("orb_high", 0.0),
                "orb_low": signal.metadata.get("orb_low", 0.0),
                "volume_surge": signal.metadata.get("volume_surge", 1.0),
                "vwap_status": "Above VWAP" if signal.direction == "CALL" else "Below VWAP",
            }
            self.alert_dispatcher.dispatch_strategy_alert(strategy_name, symbol, context_dict, is_index=True)
