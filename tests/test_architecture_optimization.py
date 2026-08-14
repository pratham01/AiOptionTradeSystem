"""Comprehensive Unit Tests for Modular Architecture Optimization."""
from __future__ import annotations

import unittest
from datetime import datetime, date
import pandas as pd
import numpy as np

from trade_system.shared.config import Settings
from trade_system.shared.exceptions import TradeSystemError, StrategyNotFoundError, DataSanityError
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.shared.notifications.templates import TelegramTemplateManager
from trade_system.domains.strategy.application.strategies.base import StrategyContext, TradeSignal
from trade_system.domains.strategy.application.strategies.strategy_registry import StrategyFactory, registered_strategies
from trade_system.domains.strategy.application.strategies.supertrend_strategy import SupertrendStrategy
from trade_system.domains.strategy.application.strategies.orb_strategy import OrbStrategy
from trade_system.domains.market_data.application.data_sanity_manager import DataSanityManager
from trade_system.interfaces.live.bar_aggregator import MultiTimeframeBarAggregator
from trade_system.interfaces.dashboard.streaming_adapter import LiveStateStreamingAdapter
from trade_system.domains.advisory.application.agent.base_agent import BaseAutonomousAgent, ThoughtProcess, AgentAction


class TestArchitectureOptimization(unittest.TestCase):

    def setUp(self):
        self.settings = Settings.load()

    def test_1_fyers_auth_singleton(self):
        """Verify FyersAuthService implements Thread-Safe Singleton pattern."""
        service1 = FyersAuthService(self.settings)
        service2 = FyersAuthService(self.settings)
        self.assertIs(service1, service2, "FyersAuthService should return the identical singleton instance.")

    def test_2_telegram_template_manager(self):
        """Verify Telegram template engine correctly formats strategy messages."""
        mgr = TelegramTemplateManager()
        ctx = {
            "symbol_short": "NIFTY",
            "time": "09:30",
            "direction": "UP",
            "timeframe": 3,
            "trend_15m": "UP",
            "close": 24500.50,
            "supertrend": 24450.00,
            "confluence": "High Zone Retest",
            "action": "BUY CALL",
            "color": "🟢",
            "strikes_block": "\n  🟢 24500CE ₹120.5",
        }
        msg = mgr.format_message("supertrend_flip", ctx)
        self.assertIn("NIFTY", msg)
        self.assertIn("24500.50", msg)
        self.assertIn("BUY CALL", msg)

    def test_3_strategy_factory(self):
        """Verify StrategyFactory discovers and instantiates all strategies."""
        available = StrategyFactory.list_available_strategies()
        self.assertIn("supertrend", available)
        self.assertIn("orb_breakout", available)
        self.assertIn("gamma_blast", available)
        self.assertIn("sniper_reversal", available)
        self.assertIn("intraday_edge", available)

        st_strat = StrategyFactory.create_strategy("supertrend", period=10, multiplier=2.0)
        self.assertIsInstance(st_strat, SupertrendStrategy)

        with self.assertRaises(StrategyNotFoundError):
            StrategyFactory.create_strategy("non_existent_strategy_xyz")

    def test_4_supertrend_strategy_evaluation(self):
        """Verify SupertrendStrategy evaluates StrategyContext and generates TradeSignal."""
        strat = SupertrendStrategy(period=7, multiplier=3.0)
        dates = pd.date_range("2026-08-14 09:15", periods=30, freq="3min")
        # Simulate uptrend breakout
        prices = [24000 + (i * 10) for i in range(30)]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 5 for p in prices],
            "low": [p - 5 for p in prices],
            "close": [p + 2 for p in prices],
            "volume": [1000] * 30,
        }, index=dates)

        ctx = StrategyContext(symbol="NSE:NIFTY50-INDEX", timeframe="3m", history_df=df)
        signals_df = strat.generate_signals(df)
        self.assertIn("supertrend", signals_df.columns)
        self.assertIn("supertrend_direction", signals_df.columns)

    def test_5_data_sanity_manager(self):
        """Verify DataSanityManager validates dataframes and corrects inconsistencies."""
        mgr = DataSanityManager()
        # Faulty dataframe with negative volume and inverted high/low
        faulty_df = pd.DataFrame({
            "open": [100.0, 102.0],
            "high": [90.0, 105.0],   # inverted high on row 0
            "low": [110.0, 99.0],    # inverted low on row 0
            "close": [95.0, 103.0],
            "volume": [-50, 1000],   # negative volume
        })
        clean_df = mgr.validate_dataframe(faulty_df, symbol="TEST")
        self.assertEqual(len(clean_df), 2)
        # High must be >= low
        self.assertTrue((clean_df["high"] >= clean_df["low"]).all())
        # Volume must be >= 0
        self.assertTrue((clean_df["volume"] >= 0).all())

    def test_6_multi_timeframe_bar_aggregator(self):
        """Verify MultiTimeframeBarAggregator ingests ticks and aggregates timeframes."""
        emitted_bars = []

        def on_bar(sym, tf, bar, df):
            emitted_bars.append((sym, tf, bar["close"]))

        agg = MultiTimeframeBarAggregator(
            symbols=["NSE:NIFTY50-INDEX"],
            timeframes=[1, 3],
            on_timeframe_bar=on_bar,
        )

        # Ingest 2 minutes of ticks
        agg.ingest_tick("NSE:NIFTY50-INDEX", {"timestamp": datetime(2026, 8, 14, 9, 15, 10), "ltp": 24000.0, "volume": 100})
        agg.ingest_tick("NSE:NIFTY50-INDEX", {"timestamp": datetime(2026, 8, 14, 9, 15, 50), "ltp": 24010.0, "volume": 200})
        # Boundary cross to minute 9:16
        agg.ingest_tick("NSE:NIFTY50-INDEX", {"timestamp": datetime(2026, 8, 14, 9, 16, 5), "ltp": 24015.0, "volume": 300})

        self.assertEqual(agg.last_tick_price["NSE:NIFTY50-INDEX"], 24015.0)

    def test_7_streaming_adapter(self):
        """Verify LiveStateStreamingAdapter provides stream health metrics."""
        adapter = LiveStateStreamingAdapter()
        adapter.record_tick("NSE:NIFTY50-INDEX", 24100.0)
        price = adapter.get_latest_price("NSE:NIFTY50-INDEX")
        self.assertEqual(price, 24100.0)

        health = adapter.get_stream_health()
        self.assertIn("status_label", health)

    def test_8_base_autonomous_agent(self):
        """Verify BaseAutonomousAgent protocol."""
        class MockAgent(BaseAutonomousAgent):
            def observe(self, market_context: Any) -> None:
                self.observed = True

            def reason(self) -> ThoughtProcess:
                return ThoughtProcess(agent_name=self.name, thought_summary="Bullish continuation expected")

            def act(self) -> AgentAction:
                return AgentAction(agent_name=self.name, action_type="TRADE_SIGNAL", payload={"direction": "CALL"})

        agent = MockAgent(name="MockAnalyst")
        agent.observe({"spot": 24000})
        thought = agent.reason()
        action = agent.act()

        self.assertEqual(thought.agent_name, "MockAnalyst")
        self.assertEqual(action.action_type, "TRADE_SIGNAL")


if __name__ == "__main__":
    unittest.main()
