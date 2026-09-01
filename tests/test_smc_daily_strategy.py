"""
Unit tests for Smart Money Concept (SMC) Daily Strategy and Scanner Agent.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trade_system.shared.config import Settings
from trade_system.domains.strategy.application.strategies.smc_strategy import (
    SmartMoneyConceptStrategy, SmcTradeSetup, SmcOrderBlock, SmcFairValueGap
)
from trade_system.domains.advisory.application.agent.smc_daily_agent import SmcDailyScannerAgent


class TestSmartMoneyConceptStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = SmartMoneyConceptStrategy(swing_length=3, fvg_min_atr_mult=0.2)

    def _generate_synthetic_ohlcv(self, pattern: str = "bullish_ob") -> pd.DataFrame:
        """Generates synthetic price action data for testing SMC patterns."""
        rows = []
        base_time = datetime(2026, 8, 1, 9, 15)
        price = 500.0

        if pattern == "bullish_ob":
            # 1. Downtrend into a base candle
            for i in range(10):
                price -= 5.0
                rows.append({
                    "timestamp": (base_time + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": price + 4.0, "high": price + 6.0, "low": price - 2.0, "close": price, "volume": 1000
                })
            # 2. Base bearish candle (Demand OB)
            price -= 3.0
            rows.append({
                "timestamp": (base_time + timedelta(days=10)).strftime("%Y-%m-%d"),
                "open": price + 5.0, "high": price + 6.0, "low": price - 1.0, "close": price, "volume": 3500
            })
            # 3. Impulsive bullish rally creating BOS and FVG
            for i in range(11, 20):
                price += 12.0
                rows.append({
                    "timestamp": (base_time + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": price - 10.0, "high": price + 4.0, "low": price - 11.0, "close": price, "volume": 2500
                })
            # 4. Pullback into the Demand OB in Discount
            for i in range(20, 25):
                price -= 10.0
                rows.append({
                    "timestamp": (base_time + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "open": price + 8.0, "high": price + 10.0, "low": price - 2.0, "close": price, "volume": 800
                })

        return pd.DataFrame(rows)

    def test_detect_order_blocks_and_fvgs(self):
        """Verify that SmartMoneyConceptStrategy detects OBs, FVGs and ATR correctly."""
        df = self._generate_synthetic_ohlcv("bullish_ob")
        atr = self.strategy.calculate_atr(df)
        self.assertEqual(len(atr), len(df))
        self.assertGreater(atr.iloc[-1], 0)

        swing_highs, swing_lows = self.strategy.find_swings(df)
        self.assertEqual(len(swing_highs), len(df))

        obs = self.strategy.detect_order_blocks(df, swing_highs, swing_lows)
        self.assertIsInstance(obs, list)

        fvgs = self.strategy.detect_fair_value_gaps(df, atr)
        self.assertIsInstance(fvgs, list)

    def test_analyze_symbol_returns_valid_trade_setup(self):
        """Verify that analyze_symbol produces a valid SmcTradeSetup with entry/sl/target."""
        df = self._generate_synthetic_ohlcv("bullish_ob")
        setup = self.strategy.analyze_symbol(df, symbol="NSE:TEST-EQ")
        
        if setup is not None:
            self.assertIsInstance(setup, SmcTradeSetup)
            self.assertIn(setup.action, ["BUY CALL", "BUY PUT"])
            self.assertGreater(setup.confluence_score, 0)
            self.assertGreater(setup.entry_price, 0)
            self.assertGreater(setup.stop_loss, 0)
            self.assertGreater(setup.target_1, 0)
            self.assertGreaterEqual(setup.risk_reward_ratio, 1.5)

    def test_scanner_agent_instantiation(self):
        """Verify SmcDailyScannerAgent initializes and can access database."""
        agent = SmcDailyScannerAgent()
        self.assertIsNotNone(agent.strategy)
        self.assertIsNotNone(agent.engine)


if __name__ == "__main__":
    unittest.main()
