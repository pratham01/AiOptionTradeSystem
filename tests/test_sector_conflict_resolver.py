"""
Unit tests for SectorConflictResolver and Stock-Sector Alignment Gatekeeper.
"""
from __future__ import annotations

import unittest
from trade_system.domains.analysis.application.analysis.sector_conflict_resolver import (
    SectorConflictResolver, SectorState, SectorAlignmentResult
)
from trade_system.domains.strategy.application.strategies.smc_strategy import SmcTradeSetup
from trade_system.domains.strategy.application.strategies.wyckoff_strategy import WyckoffSetup


class TestSectorConflictResolver(unittest.TestCase):

    def setUp(self):
        self.resolver = SectorConflictResolver()

    def test_get_sector_for_symbol(self):
        """Verify sector resolution for known F&O stocks."""
        self.assertEqual(self.resolver.get_sector_for_symbol("NSE:COFORGE-EQ"), "IT")
        self.assertEqual(self.resolver.get_sector_for_symbol("NSE:HDFCBANK-EQ"), "BANKING")
        self.assertEqual(self.resolver.get_sector_for_symbol("COFORGE"), "IT")

    def test_index_symbols_always_approved(self):
        """Verify that index symbols pass through without sector blockage."""
        result = self.resolver.evaluate_alignment("NSE:NIFTY50-INDEX", direction=1)
        self.assertTrue(result.should_send_telegram)
        self.assertFalse(result.has_conflict)
        self.assertEqual(result.verdict, "APPROVED_INDEX")

    def test_coforge_it_conflict_resolution(self):
        """
        Simulate the exact scenario: COFORGE gives BUY CALL, but IT sector is DOWN.
        Verify that evaluate_alignment flags conflict and suppresses Telegram broadcast.
        """
        # Inject synthetic sector states where IT is DOWN (-1.5%)
        self.resolver._cached_sector_states = {
            "IT": SectorState(
                sector="IT",
                pct_change=-1.5,
                rs_slope=-0.5,
                status="❄️ LAGGING",
                trend="BEARISH",
                stock_count=13,
                advancing_count=1,
                declining_count=12
            ),
            "TELECOM": SectorState(
                sector="TELECOM",
                pct_change=1.2,
                rs_slope=0.8,
                status="🔥 LEADING",
                trend="BULLISH",
                stock_count=3,
                advancing_count=3,
                declining_count=0
            )
        }
        self.resolver._cache_timestamp = __import__("datetime").datetime.now()

        # 1. COFORGE (IT) BUY CALL -> Must be suppressed due to IT being DOWN
        res_coforge_call = self.resolver.evaluate_alignment("NSE:COFORGE-EQ", direction=1)
        self.assertTrue(res_coforge_call.has_conflict, "Must detect conflict when buying stock in falling sector")
        self.assertFalse(res_coforge_call.should_send_telegram, "Must NOT send Telegram when sector conflict exists")
        self.assertEqual(res_coforge_call.verdict, "SUPPRESSED_SECTOR_HEADWIND")

        # 2. COFORGE (IT) BUY PUT -> Must be approved (Sector is DOWN, supporting PUT)
        res_coforge_put = self.resolver.evaluate_alignment("NSE:COFORGE-EQ", direction=-1)
        self.assertFalse(res_coforge_put.has_conflict)
        self.assertTrue(res_coforge_put.should_send_telegram)
        self.assertEqual(res_coforge_put.verdict, "APPROVED_HEADWIND")

        # 3. BHARTIARTL (TELECOM) BUY CALL -> Must be approved (Sector is UP +1.2%)
        res_telecom_call = self.resolver.evaluate_alignment("NSE:BHARTIARTL-EQ", direction=1)
        self.assertFalse(res_telecom_call.has_conflict)
        self.assertTrue(res_telecom_call.should_send_telegram)
        self.assertEqual(res_telecom_call.verdict, "APPROVED_TAILWIND")

    def test_filter_setups_for_telegram(self):
        """Verify filtering a batch of setups correctly separates valid from suppressed."""
        self.resolver._cached_sector_states = {
            "IT": SectorState(
                sector="IT",
                pct_change=-1.5,
                rs_slope=-0.5,
                status="❄️ LAGGING",
                trend="BEARISH",
                stock_count=13,
                advancing_count=1,
                declining_count=12
            ),
            "CONSUMER": SectorState(
                sector="CONSUMER",
                pct_change=0.8,
                rs_slope=0.4,
                status="🔥 LEADING",
                trend="BULLISH",
                stock_count=14,
                advancing_count=11,
                declining_count=3
            )
        }
        self.resolver._cache_timestamp = __import__("datetime").datetime.now()

        setup1 = SmcTradeSetup(
            symbol="NSE:COFORGE-EQ",
            action="BUY CALL",
            direction=1,
            setup_type="BULLISH_DEMAND",
            confluence_score=90.0,
            entry_price=2000.0,
            stop_loss=1900.0,
            target_1=2200.0,
            target_2=2300.0,
            risk_reward_ratio=2.0,
            market_structure="BULLISH",
            equilibrium_status="DISCOUNT",
            reasons=["Demand OB Retest"]
        )

        setup2 = SmcTradeSetup(
            symbol="NSE:MARICO-EQ",
            action="BUY CALL",
            direction=1,
            setup_type="BULLISH_DEMAND",
            confluence_score=95.0,
            entry_price=800.0,
            stop_loss=780.0,
            target_1=840.0,
            target_2=860.0,
            risk_reward_ratio=2.0,
            market_structure="BULLISH",
            equilibrium_status="DISCOUNT",
            reasons=["Demand OB Retest"]
        )

        valid, suppressed = self.resolver.filter_setups_for_telegram([setup1, setup2])
        self.assertEqual(len(valid), 1, "Only 1 setup without sector conflict should pass")
        self.assertEqual(valid[0].symbol, "NSE:MARICO-EQ")
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0][0].symbol, "NSE:COFORGE-EQ")


if __name__ == "__main__":
    unittest.main()
