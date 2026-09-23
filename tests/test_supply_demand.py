"""
Unit tests for Institutional Supply & Demand Engine (JeaFx Framework).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trade_system.domains.strategy.application.indicators.supply_demand import (
    SupplyDemandEngine, SupplyDemandZone, SDZoneType
)


class TestSupplyDemandEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SupplyDemandEngine(swing_length=3, fvg_min_atr_mult=0.2)

    def _create_synthetic_sd_dataset(self) -> pd.DataFrame:
        """
        Creates a dataset with:
        1. Down move to bottom at bar 8 (price = 200)
        2. Base candle at bar 9 (open=203, high=204, low=199, close=201) -> Extreme Base
        3. Bullish impulse candle at bar 10 (open=202, high=220, low=202, close=218)
        4. Continuation candle at bar 11 (open=218, high=230, low=210, close=228)
           -> Gap between candle 9 high (204) and candle 11 low (210) = 6 points (> ATR * 0.2)
        5. Decisional consolidation at bars 14-15 (price = 235), followed by breakout to 260.
        6. Opposing supply zone up at 270-275.
        """
        rows = []
        base_time = datetime(2026, 9, 1, 9, 15)

        # 0-7: Downtrend
        for i in range(8):
            p = 230 - i * 4
            rows.append({
                "timestamp": (base_time + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": p + 2.0, "high": p + 3.0, "low": p - 2.0, "close": p, "volume": 1000
            })

        # Bar 8: Lowest wick (198)
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 8)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 204.0, "high": 205.0, "low": 198.0, "close": 200.0, "volume": 1200
        })

        # Bar 9: Base candle (Extreme Demand Base)
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 9)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 201.0, "high": 204.0, "low": 199.0, "close": 200.0, "volume": 3500
        })

        # Bar 10: Impulse candle
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 10)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 202.0, "high": 220.0, "low": 202.0, "close": 218.0, "volume": 5000
        })

        # Bar 11: Gap continuation candle (low = 210, leaving 204 to 210 open FVG)
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 11)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 218.0, "high": 230.0, "low": 210.0, "close": 228.0, "volume": 4200
        })

        # Bars 12-16: Climb to 260
        for i in range(12, 17):
            p = 230 + (i - 12) * 6
            rows.append({
                "timestamp": (base_time + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": p - 2.0, "high": p + 3.0, "low": p - 3.0, "close": p, "volume": 2000
            })

        # Bars 17-20: Opposing Supply formation at 270 (Base at 18, impulse down at 19)
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 17)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 260.0, "high": 268.0, "low": 258.0, "close": 265.0, "volume": 2500
        })
        # Bar 18: Supply base candle
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 18)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 266.0, "high": 275.0, "low": 264.0, "close": 272.0, "volume": 4000
        })
        # Bar 19: Bearish impulse candle
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 19)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 271.0, "high": 271.0, "low": 250.0, "close": 252.0, "volume": 6000
        })
        # Bar 20: Bearish gap candle (high = 258, leaving 264 to 258 open FVG)
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 20)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 251.0, "high": 258.0, "low": 242.0, "close": 245.0, "volume": 3800
        })
        # Bars 21-22: Continuation bars
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 21)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 244.0, "high": 246.0, "low": 238.0, "close": 240.0, "volume": 3000
        })
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * 22)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 240.0, "high": 242.0, "low": 235.0, "close": 238.0, "volume": 2800
        })

        return pd.DataFrame(rows)

    def test_imbalance_validated_demand_and_supply_detection(self):
        """Verify that Extreme Demand and Supply zones with open imbalances are detected."""
        df = self._create_synthetic_sd_dataset()
        zones = self.engine.detect_zones(df)

        self.assertGreater(len(zones), 0)
        demand_zones = [z for z in zones if z.is_demand]
        supply_zones = [z for z in zones if z.is_supply]

        self.assertGreater(len(demand_zones), 0)
        self.assertGreater(len(supply_zones), 0)

        # Verify Extreme Demand
        ext_demand = next((z for z in demand_zones if z.zone_type == SDZoneType.EXTREME_DEMAND), None)
        self.assertIsNotNone(ext_demand)
        self.assertTrue(ext_demand.has_valid_imbalance)
        self.assertGreater(ext_demand.imbalance_size, 0)
        self.assertTrue(ext_demand.is_fresh)

        # Verify Extreme Supply
        ext_supply = next((z for z in supply_zones if z.zone_type == SDZoneType.EXTREME_SUPPLY), None)
        self.assertIsNotNone(ext_supply)
        self.assertTrue(ext_supply.has_valid_imbalance)

    def test_opposing_target_zone_discovery(self):
        """Verify Range-to-Range target discovery finds nearest opposing supply for buyers."""
        df = self._create_synthetic_sd_dataset()
        zones = self.engine.detect_zones(df)

        current_price = 225.0
        opposing = self.engine.find_opposing_target_zone(
            current_price=current_price,
            is_bullish=True,
            active_zones=zones,
        )
        self.assertIsNotNone(opposing)
        self.assertTrue(opposing.is_supply)
        self.assertGreater(opposing.bottom, current_price)


if __name__ == "__main__":
    unittest.main()
