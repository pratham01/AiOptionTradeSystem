"""
Unit tests for Daily Fair Value Gap (FVG) Strategy and Scanner Agent.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trade_system.domains.strategy.application.strategies.fvg_strategy import (
    FairValueGapStrategy, FvgTradeSetup, DailyFairValueGap
)
from trade_system.domains.advisory.application.agent.fvg_agent import FvgDailyScannerAgent


class TestFairValueGapStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = FairValueGapStrategy(min_gap_atr_mult=0.2, min_rrr=1.5)

    def _generate_synthetic_bullish_fvg_data(self) -> pd.DataFrame:
        """Generates synthetic daily bars creating a Bullish FVG (BISI) and a 50% CE retest."""
        rows = []
        base_time = datetime(2026, 8, 1, 9, 15)

        # Baseline candles
        for i in range(10):
            p = 1000.0 + i * 2.0
            rows.append({
                "timestamp": (base_time + timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": p - 2.0, "high": p + 3.0, "low": p - 3.0, "close": p + 1.0, "volume": 1000
            })

        # Candle 1: High = 1030.0
        rows.append({
            "timestamp": (base_time + timedelta(days=10)).strftime("%Y-%m-%d"),
            "open": 1025.0, "high": 1030.0, "low": 1020.0, "close": 1028.0, "volume": 1500
        })

        # Candle 2 (Impulse): 1028 -> 1070
        rows.append({
            "timestamp": (base_time + timedelta(days=11)).strftime("%Y-%m-%d"),
            "open": 1029.0, "high": 1075.0, "low": 1027.0, "close": 1070.0, "volume": 4000
        })

        # Candle 3: Low = 1050.0 (Gap between C1 High 1030 and C3 Low 1050 = 20 pts)
        # CE level = (1030 + 1050) / 2 = 1040.0
        rows.append({
            "timestamp": (base_time + timedelta(days=12)).strftime("%Y-%m-%d"),
            "open": 1072.0, "high": 1090.0, "low": 1050.0, "close": 1085.0, "volume": 2000
        })

        # Candle 4 (Pullback / Retest of FVG zone and 50% CE): Low = 1038, Close = 1045
        rows.append({
            "timestamp": (base_time + timedelta(days=13)).strftime("%Y-%m-%d"),
            "open": 1060.0, "high": 1062.0, "low": 1038.0, "close": 1048.0, "volume": 1200
        })

        return pd.DataFrame(rows)

    def test_detect_bullish_fvg_and_ce_level(self):
        """Verify detect_all_fvgs identifies BISI and computes accurate 50% CE."""
        df = self._generate_synthetic_bullish_fvg_data()
        fvgs = self.strategy.detect_all_fvgs(df)

        self.assertGreaterEqual(len(fvgs), 1)
        bull_fvg = next((f for f in fvgs if f.fvg_type == "BULLISH" and f.gap_size == 20.0), None)
        self.assertIsNotNone(bull_fvg)
        self.assertEqual(bull_fvg.bottom, 1030.0)
        self.assertEqual(bull_fvg.top, 1050.0)
        self.assertEqual(bull_fvg.ce_level, 1040.0)

    def test_analyze_setup_returns_fvg_trade_setup(self):
        """Verify analyze_setup generates trade setup on FVG retest."""
        df = self._generate_synthetic_bullish_fvg_data()
        setup = self.strategy.analyze_setup(df, symbol="NSE:TEST-EQ")

        self.assertIsNotNone(setup)
        self.assertIsInstance(setup, FvgTradeSetup)
        self.assertEqual(setup.action, "BUY CALL")
        self.assertEqual(setup.consequent_encroachment, 1040.0)
        self.assertGreater(setup.entry_price, 0)
        self.assertGreater(setup.stop_loss, 0)
        self.assertGreater(setup.target_1, 0)
        self.assertGreaterEqual(setup.risk_reward_ratio, 1.5)

    def test_fvg_agent_init(self):
        """Verify FvgDailyScannerAgent initializes cleanly."""
        agent = FvgDailyScannerAgent()
        self.assertIsNotNone(agent.strategy)
        self.assertIsNotNone(agent.conflict_resolver)


if __name__ == "__main__":
    unittest.main()
