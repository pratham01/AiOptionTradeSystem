"""
Unit tests for FibonacciRetracementStrategy and Multi-Strategy SmartEntryTrigger.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trade_system.domains.strategy.application.strategies.fibonacci_retracement import (
    FibonacciRetracementStrategy, FibonacciGrid, FibonacciSetup
)
from trade_system.domains.analysis.application.analysis.smart_entry_trigger import SmartEntryTrigger


class TestFibonacciStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = FibonacciRetracementStrategy(swing_lookback=25)

    def _generate_synthetic_bullish_swing(self) -> pd.DataFrame:
        """Generates synthetic impulse move from 1000 to 1100, then pulling back to 1045 (Golden Pocket)."""
        rows = []
        base_time = datetime(2026, 8, 1, 9, 15)

        # 1. Base at 1000.0 (Swing Low)
        for i in range(5):
            rows.append({
                "timestamp": base_time + timedelta(days=i),
                "open": 998.0, "high": 1005.0, "low": 995.0, "close": 1000.0, "volume": 1000
            })

        # 2. Impulse leg up to 1100.0 (Swing High)
        for i in range(5, 10):
            p = 1000.0 + (i - 4) * 20.0  # 1020, 1040, 1060, 1080, 1100
            rows.append({
                "timestamp": base_time + timedelta(days=i),
                "open": p - 10.0, "high": p + 2.0, "low": p - 12.0, "close": p, "volume": 2500
            })

        # 3. Pullback into Golden Pocket (50.0% = 1050, 61.8% = 1038.2) -> Close at 1045.0
        rows.append({
            "timestamp": base_time + timedelta(days=10),
            "open": 1070.0, "high": 1075.0, "low": 1042.0, "close": 1045.0, "volume": 1200
        })

        return pd.DataFrame(rows)

    def test_fibonacci_grid_calculation(self):
        """Verify calculation of exact Fibonacci Retracement & Extension levels."""
        df = self._generate_synthetic_bullish_swing()
        grid = self.strategy.calculate_grid(df, direction="CALL")

        self.assertIsNotNone(grid)
        self.assertEqual(grid.swing_low, 995.0)
        self.assertEqual(grid.swing_high, 1102.0)
        self.assertAlmostEqual(grid.impulse_range, 107.0, places=1)

        # Check 50.0% and 61.8% levels
        expected_50 = 1102.0 - (0.500 * 107.0)
        expected_618 = 1102.0 - (0.618 * 107.0)
        self.assertAlmostEqual(grid.level_500, expected_50, places=1)
        self.assertAlmostEqual(grid.level_618, expected_618, places=1)

        # Check Golden Pocket state
        self.assertTrue(grid.is_in_golden_pocket)
        self.assertEqual(grid.current_zone, "GOLDEN_POCKET")

    def test_fibonacci_setup_evaluation(self):
        """Verify evaluate_setup produces valid trade setup."""
        df = self._generate_synthetic_bullish_swing()
        setup = self.strategy.evaluate_setup(df, symbol="NSE:TEST-EQ", direction="CALL", atr=10.0)

        self.assertIsNotNone(setup)
        self.assertEqual(setup.action, "BUY CALL")
        self.assertEqual(setup.status, "TRIGGERED")
        self.assertGreater(setup.entry_price, 0)
        self.assertGreater(setup.stop_loss, 0)
        self.assertGreater(setup.target_1, setup.entry_price)
        self.assertGreater(setup.target_2, setup.target_1)

    def test_smart_entry_trigger_evaluate_all(self):
        """Verify SmartEntryTrigger returns multiple strategies including Fibonacci and EMA."""
        df = self._generate_synthetic_bullish_swing()
        trigger = SmartEntryTrigger(horizon="INTRADAY")
        all_trigs = trigger.evaluate_all(direction="CALL", ltp=1045.0, atr=10.0, df_base=df)

        self.assertIn("FIBONACCI", all_trigs)
        fib_trig = all_trigs["FIBONACCI"]
        self.assertEqual(fib_trig.trigger_type, "FIB_GOLDEN_ZONE")
        self.assertEqual(fib_trig.status, "TRIGGERED")


if __name__ == "__main__":
    unittest.main()
