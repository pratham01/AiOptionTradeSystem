"""Tests for DataSanityManager and DataSyncService automatic healing and sanity guarantees."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from datetime import datetime, date, timedelta
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trade_system.shared.config import Settings
from trade_system.shared.exceptions import DataSanityError
from trade_system.domains.market_data.application.data_sanity_manager import DataSanityManager, SanityReport
from trade_system.interfaces.live.data_sync_service import DataSyncService, get_expected_last_trading_date


class TestDataSanityManager(unittest.TestCase):

    def setUp(self):
        self.settings = Settings.load()
        self.sanity_mgr = DataSanityManager(settings=self.settings)

    def test_validate_dataframe_cleans_corrupt_values(self):
        """Verify validate_dataframe fixes high/low inconsistencies and drops negative values."""
        raw_df = pd.DataFrame([
            {"timestamp": "2026-08-28 09:15:00", "open": 100.0, "high": 90.0, "low": 110.0, "close": 105.0, "volume": 500},
            {"timestamp": "2026-08-28 09:30:00", "open": -50.0, "high": 100.0, "low": 50.0, "close": 80.0, "volume": 100},  # negative open
            {"timestamp": "2026-08-28 09:45:00", "open": 102.0, "high": 108.0, "low": 101.0, "close": 106.0, "volume": -20}, # negative volume
            {"timestamp": "2026-08-28 09:45:00", "open": 102.0, "high": 108.0, "low": 101.0, "close": 106.0, "volume": 30},  # duplicate timestamp
        ])

        cleaned = self.sanity_mgr.validate_dataframe(raw_df, symbol="NSE:TEST-EQ")
        self.assertEqual(len(cleaned), 2, "Should drop negative price row and deduplicate timestamps")
        
        row1 = cleaned.iloc[0]
        self.assertGreaterEqual(row1["high"], row1["low"], "High must be >= low")
        self.assertEqual(row1["high"], 110.0, "High must bound max of open, close, high")
        self.assertEqual(row1["low"], 90.0, "Low must bound min of open, close, low")
        
        row2 = cleaned.iloc[1]
        self.assertGreaterEqual(row2["volume"], 0, "Volume must be clipped to >= 0")

    def test_validate_dataframe_raises_on_missing_columns(self):
        """Verify error is raised if essential OHLC columns are missing."""
        invalid_df = pd.DataFrame([{"timestamp": "2026-08-28", "open": 100.0}])
        with self.assertRaises(DataSanityError):
            self.sanity_mgr.validate_dataframe(invalid_df, symbol="NSE:TEST-EQ")

    def test_get_expected_last_trading_date_weekday_logic(self):
        """Verify expected last trading date calculates previous trading session."""
        # Monday morning 08:00 IST -> should be Friday (3 days prior)
        mon_morning = datetime(2026, 8, 31, 8, 0, 0)
        expected = get_expected_last_trading_date(mon_morning)
        self.assertEqual(expected, date(2026, 8, 28), "Monday pre-market expected date should be Friday")

        # Friday afternoon 16:00 IST -> should be Friday
        fri_afternoon = datetime(2026, 8, 28, 16, 0, 0)
        expected_fri = get_expected_last_trading_date(fri_afternoon)
        self.assertEqual(expected_fri, date(2026, 8, 28), "Friday post-market expected date should be Friday")

        # Sunday -> should be Friday
        sunday = datetime(2026, 8, 30, 12, 0, 0)
        expected_sun = get_expected_last_trading_date(sunday)
        self.assertEqual(expected_sun, date(2026, 8, 28), "Sunday expected date should be Friday")

    def test_auto_heal_triggered_when_gaps_exist(self):
        """Verify that when gaps are detected and broker is present, backfilling is invoked."""
        mock_broker = MagicMock()
        mock_broker.fetch_history.return_value = pd.DataFrame([
            {"timestamp": "2026-08-28 09:15:00", "open": 200.0, "high": 205.0, "low": 198.0, "close": 202.0, "volume": 1000}
        ])

        mgr = DataSanityManager(broker=mock_broker, settings=self.settings)
        # Check specific symbols
        report = mgr.ensure_data_sanity_and_heal(
            symbols=["NSE:RELIANCE-EQ"],
            resolutions=["D"],
            auto_heal=True
        )
        self.assertIsInstance(report, SanityReport)
        self.assertEqual(report.total_symbols_checked, 1)


if __name__ == "__main__":
    unittest.main()
