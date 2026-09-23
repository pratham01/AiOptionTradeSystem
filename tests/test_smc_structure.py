"""
Unit tests for Institutional SMC Structure Engine (Photon Trading Framework).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trade_system.domains.strategy.application.indicators.smc_structure import (
    SMCStructureEngine, SMCStructureState, StructureTrend, PointType, BreakType
)


class TestSMCStructureEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SMCStructureEngine(swing_length=3, internal_length=2)

    def _create_bullish_trend_with_pullback(self) -> pd.DataFrame:
        """
        Creates a synthetic price series:
        1. Swing High at bar 5 (price = 110)
        2. Swing Low at bar 8 (price = 95)
        3. Break of Structure (BOS) rallying to 130 at bar 15 (Swing Low 95 becomes Strong Low)
        4. Internal pullback: minor lower highs (128 -> 124) and lower lows (122 -> 118)
        5. Internal CHoCH: price explodes to 126, breaking 124!
        """
        rows = []
        base_time = datetime(2026, 9, 1, 9, 15)
        
        # Phase 1: Climb to 110, swing low to 95, BOS to 130
        prices = [100, 102, 105, 108, 110, 108, 105, 98, 95, 98, 104, 112, 120, 128, 130]
        # Phase 2: Pullback leg drops to 121, bounces to 125, drops to 118 (forms clean internal high at 125)
        prices += [124, 121, 125, 120, 118]
        # Phase 3: Internal CHoCH breakout above 125 -> 127, 129
        prices += [122, 127, 129]

        for i, p in enumerate(prices):
            rows.append({
                "timestamp": (base_time + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": p - 1.0,
                "high": p + 1.5,
                "low": p - 1.5,
                "close": p,
                "volume": 1000 + i * 50,
            })

        return pd.DataFrame(rows)

    def test_bullish_swing_bos_and_strong_low(self):
        """Verify that a breakout above swing high creates a Bullish Swing BOS and Strong Low."""
        df = self._create_bullish_trend_with_pullback()
        state = self.engine.analyze(df)

        self.assertIsInstance(state, SMCStructureState)
        self.assertEqual(state.swing_trend, StructureTrend.BULLISH)
        self.assertIsNotNone(state.strong_protected_level)
        self.assertIsNotNone(state.weak_target_level)
        # Strong protected level must be <= 96.0 (around the 95 pivot)
        self.assertLessEqual(state.strong_protected_level, 96.0)

    def test_internal_choch_realignment(self):
        """Verify that an internal CHoCH triggers is_internal_realigned=True."""
        df = self._create_bullish_trend_with_pullback()
        state = self.engine.analyze(df)

        # Check breaks
        break_types = [b.break_type for b in state.recent_breaks]
        self.assertIn(BreakType.SWING_BOS, break_types)
        self.assertIn(BreakType.INTERNAL_CHOCH_BULL, break_types)
        self.assertTrue(state.is_internal_realigned)

    def test_empty_or_short_dataframe(self):
        """Verify safe fallback on short data."""
        short_df = pd.DataFrame({"high": [100], "low": [90], "close": [95], "open": [92], "timestamp": ["2026-09-01"]})
        state = self.engine.analyze(short_df)
        self.assertEqual(state.swing_trend, StructureTrend.SIDEWAYS)
        self.assertIsNone(state.strong_protected_level)


if __name__ == "__main__":
    unittest.main()
