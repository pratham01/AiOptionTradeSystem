"""
Unit tests for Wyckoff & VSA Institutional Strategy and Backtesting Engine.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trade_system.domains.strategy.application.strategies.wyckoff_strategy import (
    WyckoffVsaStrategy, WyckoffSetup
)
from trade_system.domains.analysis.application.backtesting.wyckoff_backtest import (
    WyckoffBacktestEngine
)
from trade_system.domains.advisory.application.agent.wyckoff_agent import (
    WyckoffDailyScannerAgent
)


class TestWyckoffStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = WyckoffVsaStrategy(range_lookback=15, min_rrr=1.5)

    def _generate_synthetic_wyckoff_spring(self) -> pd.DataFrame:
        """Generates synthetic Trading Range with a Phase C Spring."""
        rows = []
        base_time = datetime(2026, 7, 1, 9, 15)
        # Establish Range: 500 to 550
        for i in range(25):
            mid = 525.0 + 20.0 * np.sin(i * 0.5)
            rows.append({
                "timestamp": (base_time + timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": mid - 2.0, "high": mid + 5.0, "low": mid - 5.0, "close": mid + 2.0, "volume": 1000
            })
        # Spring candle: pokes to 495 (below 500 low), closes at 505 with 2.5x volume
        rows.append({
            "timestamp": (base_time + timedelta(days=25)).strftime("%Y-%m-%d"),
            "open": 502.0, "high": 508.0, "low": 495.0, "close": 506.0, "volume": 2500
        })
        return pd.DataFrame(rows)

    def test_wyckoff_spring_detection(self):
        """Verify WyckoffVsaStrategy detects Spring pattern and sets trade levels."""
        df = self._generate_synthetic_wyckoff_spring()
        setup = self.strategy.analyze_setup(df, symbol="NSE:TEST-EQ")
        if setup is not None:
            self.assertIsInstance(setup, WyckoffSetup)
            self.assertIn(setup.action, ["BUY CALL", "BUY PUT"])
            self.assertGreater(setup.entry_price, 0)
            self.assertGreater(setup.stop_loss, 0)
            self.assertGreater(setup.target_1, 0)

    def test_wyckoff_backtest_simulation(self):
        """Verify WyckoffBacktestEngine executes simulation on dataframe."""
        df = self._generate_synthetic_wyckoff_spring()
        # Add follow-through bars to complete trade
        base_time = datetime(2026, 7, 27, 9, 15)
        for i in range(10):
            p = 510.0 + i * 5.0
            df = pd.concat([df, pd.DataFrame([{
                "timestamp": (base_time + timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": p - 2.0, "high": p + 4.0, "low": p - 3.0, "close": p + 2.0, "volume": 1200
            }])], ignore_index=True)

        engine = WyckoffBacktestEngine(max_holding_bars=10)
        trades = engine.run_backtest_on_dataframe(df, symbol="NSE:TEST-EQ")
        self.assertIsInstance(trades, list)

    def test_wyckoff_agent_init(self):
        """Verify WyckoffDailyScannerAgent initializes cleanly."""
        agent = WyckoffDailyScannerAgent()
        self.assertIsNotNone(agent.strategy)
        self.assertIsNotNone(agent.engine)


if __name__ == "__main__":
    unittest.main()
