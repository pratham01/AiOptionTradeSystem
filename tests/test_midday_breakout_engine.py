"""
Unit tests for MiddayBreakoutEngine (Afternoon Squeeze & Breakout Radar).
"""
from __future__ import annotations

import unittest
from datetime import datetime, time as dt_time, timedelta
import pandas as pd
import numpy as np

from trade_system.domains.analysis.application.analysis.midday_breakout_engine import (
    MiddayBreakoutEngine, MiddayBreakoutSetup
)


class TestMiddayBreakoutEngine(unittest.TestCase):

    def _create_synthetic_day(
        self,
        breakout_type: str = "bullish",
        coil_width_pct: float = 0.7,
    ) -> pd.DataFrame:
        """
        Creates a day of 15m bars from 09:15 to 14:00:
        - 09:15 to 10:15: Morning volatility
        - 10:30 to 12:45: Tight consolidation around 1000
        - 13:00 to 14:00: Breakout or breakdown
        """
        rows = []
        base_date = datetime(2026, 9, 23, 9, 15)

        # Bar times: 09:15, 09:30, 09:45, 10:00, 10:15
        morning_prices = [1000, 1010, 1008, 995, 998]
        for i, p in enumerate(morning_prices):
            rows.append({
                "timestamp": base_date + timedelta(minutes=15 * i),
                "open": p - 1.0, "high": p + 3.0, "low": p - 3.0, "close": p, "volume": 10000
            })

        # Midday coil: 10:30 to 12:45 (10 bars: 10:30, 10:45, 11:00, 11:15, 11:30, 11:45, 12:00, 12:15, 12:30, 12:45)
        # Half range for coil_width_pct around 1000
        half_range = 1000 * (coil_width_pct / 200.0)
        mid_high = 1000 + half_range
        mid_low = 1000 - half_range
        for i in range(5, 15):
            p = 1000.0 + (1.0 if i % 2 == 0 else -1.0)
            rows.append({
                "timestamp": base_date + timedelta(minutes=15 * i),
                "open": p, "high": mid_high, "low": mid_low, "close": p, "volume": 3000
            })

        # Afternoon: 13:00 to 14:00
        if breakout_type == "bullish":
            afternoon_prices = [mid_high + 2.0, mid_high + 8.0, mid_high + 14.0, mid_high + 18.0]
            vols = [15000, 25000, 22000, 20000]
        elif breakout_type == "bearish":
            afternoon_prices = [mid_low - 2.0, mid_low - 8.0, mid_low - 14.0, mid_low - 18.0]
            vols = [15000, 25000, 22000, 20000]
        else:  # coiled
            afternoon_prices = [1000, 1001, 1000.5, 1001]
            vols = [3000, 3100, 2900, 3000]

        for i, (p, v) in enumerate(zip(afternoon_prices, vols), start=15):
            rows.append({
                "timestamp": base_date + timedelta(minutes=15 * i),
                "open": p - 0.5, "high": p + 1.0, "low": p - 1.0, "close": p, "volume": v
            })

        return pd.DataFrame(rows)

    def test_bullish_midday_breakout(self):
        """Verify bullish breakout detection above lunch ceiling with volume."""
        df = self._create_synthetic_day("bullish", coil_width_pct=0.7)
        setup = MiddayBreakoutEngine.evaluate_symbol(
            symbol="NSE:TEST-EQ",
            sector="NIFTY IT",
            candles_df=df,
            pcr_value=1.1,
        )

        self.assertIsNotNone(setup)
        self.assertEqual(setup.status, "BULLISH_BREAKOUT")
        self.assertGreater(setup.score, 70)
        self.assertGreater(setup.ltp, setup.midday_high)
        self.assertLess(setup.stop_loss, setup.ltp)
        self.assertGreater(setup.target_1, setup.ltp)
        self.assertGreaterEqual(setup.risk_reward, 1.5)

    def test_bearish_midday_breakdown(self):
        """Verify bearish breakdown detection below lunch floor."""
        df = self._create_synthetic_day("bearish", coil_width_pct=0.7)
        setup = MiddayBreakoutEngine.evaluate_symbol(
            symbol="NSE:TEST-EQ",
            sector="NIFTY AUTO",
            candles_df=df,
            pcr_value=0.5,
        )

        self.assertIsNotNone(setup)
        self.assertEqual(setup.status, "BEARISH_BREAKDOWN")
        self.assertGreater(setup.score, 70)
        self.assertLess(setup.ltp, setup.midday_low)
        self.assertGreater(setup.stop_loss, setup.ltp)
        self.assertLess(setup.target_1, setup.ltp)

    def test_coiled_squeeze_waiting_for_trigger(self):
        """Verify coiled stocks still inside the channel are tagged as COILED_SQUEEZE."""
        df = self._create_synthetic_day("coiled", coil_width_pct=0.6)
        setup = MiddayBreakoutEngine.evaluate_symbol(
            symbol="NSE:TEST-EQ",
            sector="NIFTY PHARMA",
            candles_df=df,
        )

        self.assertIsNotNone(setup)
        self.assertEqual(setup.status, "COILED_SQUEEZE")
        self.assertLessEqual(setup.midday_range_pct, 1.0)

    def test_wide_range_ignored(self):
        """Verify stocks with wide midday range (> max_coil_pct) are filtered out."""
        df = self._create_synthetic_day("bullish", coil_width_pct=2.5)
        setup = MiddayBreakoutEngine.evaluate_symbol(
            symbol="NSE:TEST-EQ",
            sector="NIFTY METAL",
            candles_df=df,
            max_coil_pct=1.0,
        )
        self.assertIsNone(setup)


if __name__ == "__main__":
    unittest.main()
