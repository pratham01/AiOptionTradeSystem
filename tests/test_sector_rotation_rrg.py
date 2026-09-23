import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trade_system.domains.analysis.application.analysis.sector_rotation_rrg import (
    SectorRRGEngine,
    SectorRRGPoint
)


class TestSectorRRGEngine:

    def test_compute_sector_breadth(self):
        # Synthetic merged_closes
        data = [
            {"symbol": "NSE:A-EQ", "sector": "AUTO", "pChange": 2.5, "vol_surge": 1.8},
            {"symbol": "NSE:B-EQ", "sector": "AUTO", "pChange": 1.0, "vol_surge": 1.2},
            {"symbol": "NSE:C-EQ", "sector": "AUTO", "pChange": -0.5, "vol_surge": 0.8},
            {"symbol": "NSE:D-EQ", "sector": "IT", "pChange": -1.5, "vol_surge": 2.0},
            {"symbol": "NSE:E-EQ", "sector": "IT", "pChange": -2.0, "vol_surge": 1.0},
            {"symbol": "NSE:F-EQ", "sector": "UNKNOWN", "pChange": 0.5, "vol_surge": 1.0},
        ]
        df = pd.DataFrame(data)
        breadth = SectorRRGEngine.compute_sector_breadth(df)

        assert "AUTO" in breadth
        assert "IT" in breadth
        assert "UNKNOWN" not in breadth

        auto_b = breadth["AUTO"]
        assert auto_b["advances"] == 2
        assert auto_b["declines"] == 1
        assert auto_b["total_stocks"] == 3
        assert pytest.approx(auto_b["advance_pct"], 0.1) == 66.7
        assert pytest.approx(auto_b["avg_vol_surge"], 0.05) == 1.27

        it_b = breadth["IT"]
        assert it_b["advances"] == 0
        assert it_b["declines"] == 2
        assert it_b["advance_pct"] == 0.0

    def test_rrg_quadrants_classification(self):
        # 1. Leading
        assert SectorRRGEngine._classify_quadrant(102.5, 101.0) == "LEADING"
        # 2. Weakening
        assert SectorRRGEngine._classify_quadrant(103.0, 98.5) == "WEAKENING"
        # 3. Lagging
        assert SectorRRGEngine._classify_quadrant(97.0, 99.0) == "LAGGING"
        # 4. Improving
        assert SectorRRGEngine._classify_quadrant(98.5, 101.5) == "IMPROVING"

    def test_compute_rrg_intraday_series(self):
        # Generate synthetic 15m candles across 6 timestamps
        base_time = datetime(2026, 9, 22, 9, 15)
        timestamps = [base_time + timedelta(minutes=15 * i) for i in range(6)]

        rows = []
        # Benchmark: Nifty 50 (starts at 100, rises steadily to 101: +1%)
        bench_prices = [100.0, 100.2, 100.4, 100.6, 100.8, 101.0]
        for ts, p in zip(timestamps, bench_prices):
            rows.append({
                "timestamp": ts, "symbol": "NSE:NIFTY50-INDEX", "sector": "INDEX",
                "open": 100.0, "high": p + 0.1, "low": p - 0.1, "close": p, "volume": 10000
            })

        # Sector 1: OUTPERFORMING (starts at 100, rises to 103: +3% -> LEADING)
        s1_prices = [100.0, 100.5, 101.2, 102.0, 102.5, 103.0]
        for ts, p in zip(timestamps, s1_prices):
            rows.append({
                "timestamp": ts, "symbol": "NSE:AUTO1-EQ", "sector": "AUTO",
                "open": 100.0, "high": p + 0.2, "low": p - 0.2, "close": p, "volume": 5000
            })

        # Sector 2: UNDERPERFORMING (starts at 100, falls to 98: -2% -> LAGGING)
        s2_prices = [100.0, 99.6, 99.2, 98.8, 98.4, 98.0]
        for ts, p in zip(timestamps, s2_prices):
            rows.append({
                "timestamp": ts, "symbol": "NSE:IT1-EQ", "sector": "IT",
                "open": 100.0, "high": p + 0.2, "low": p - 0.2, "close": p, "volume": 5000
            })

        df_today = pd.DataFrame(rows)
        merged_closes = pd.DataFrame([
            {"symbol": "NSE:AUTO1-EQ", "sector": "AUTO", "pChange": 3.0, "vol_surge": 2.1},
            {"symbol": "NSE:IT1-EQ", "sector": "IT", "pChange": -2.0, "vol_surge": 1.4},
        ])

        rrg_results = SectorRRGEngine.compute_rrg(df_today, merged_closes, tail_bars=4)

        assert "AUTO" in rrg_results
        assert "IT" in rrg_results

        auto_pt = rrg_results["AUTO"]
        assert auto_pt.rs_ratio > 100.0
        assert auto_pt.quadrant == "LEADING"
        assert auto_pt.advances == 1
        assert len(auto_pt.history_tail) == 4

        it_pt = rrg_results["IT"]
        assert it_pt.rs_ratio < 100.0
        assert it_pt.quadrant == "LAGGING"
        assert it_pt.declines == 1

    def test_fallback_when_intraday_candles_empty(self):
        # Only merged_closes provided
        merged_closes = pd.DataFrame([
            {"symbol": "NSE:METALS1-EQ", "sector": "METALS", "pChange": 1.5, "vol_surge": 1.5},
            {"symbol": "NSE:FMCG1-EQ", "sector": "FMCG", "pChange": -1.2, "vol_surge": 0.9},
        ])

        rrg_results = SectorRRGEngine.compute_rrg(pd.DataFrame(), merged_closes)
        assert "METALS" in rrg_results
        assert rrg_results["METALS"].rs_ratio > 100.0
        assert rrg_results["METALS"].quadrant in ["LEADING", "WEAKENING"]
        assert "FMCG" in rrg_results
        assert rrg_results["FMCG"].rs_ratio < 100.0

    def test_one_timestamp_fallback_snapshot(self):
        # When today has only 1 candle (e.g. 09:15 open), it must compute valid RRG points
        merged_closes = pd.DataFrame([
            {"symbol": "NSE:BANK1-EQ", "sector": "BANKING", "pChange": 2.0, "vol_surge": 1.4},
            {"symbol": "NSE:OIL1-EQ", "sector": "OIL_GAS", "pChange": -1.0, "vol_surge": 1.1},
        ])
        df_1_candle = pd.DataFrame([
            {"symbol": "NSE:BANK1-EQ", "timestamp": pd.Timestamp("2026-09-23 09:15"), "open": 500, "close": 510, "sector": "BANKING"},
            {"symbol": "NSE:OIL1-EQ", "timestamp": pd.Timestamp("2026-09-23 09:15"), "open": 200, "close": 198, "sector": "OIL_GAS"},
        ])
        rrg_res = SectorRRGEngine.compute_rrg(df_1_candle, merged_closes)
        assert len(rrg_res) == 2
        assert "BANKING" in rrg_res
        assert rrg_res["BANKING"].advances == 1
        assert rrg_res["BANKING"].declines == 0
        assert len(rrg_res["BANKING"].history_tail) == 3

    def test_missing_sector_column_in_candles(self):
        # When today_15m_df lacks 'sector' column, compute_rrg maps it automatically from merged_closes
        merged_closes = pd.DataFrame([
            {"symbol": "NSE:INFY-EQ", "sector": "IT", "pChange": 1.5, "vol_surge": 1.2},
            {"symbol": "NSE:TCS-EQ", "sector": "IT", "pChange": 0.8, "vol_surge": 1.0},
        ])
        df_no_sec = pd.DataFrame([
            {"symbol": "NSE:INFY-EQ", "timestamp": pd.Timestamp("2026-09-23 09:15"), "open": 100, "close": 101},
            {"symbol": "NSE:INFY-EQ", "timestamp": pd.Timestamp("2026-09-23 09:30"), "open": 101, "close": 102},
            {"symbol": "NSE:TCS-EQ", "timestamp": pd.Timestamp("2026-09-23 09:15"), "open": 200, "close": 201},
            {"symbol": "NSE:TCS-EQ", "timestamp": pd.Timestamp("2026-09-23 09:30"), "open": 201, "close": 202},
        ])
        rrg_res = SectorRRGEngine.compute_rrg(df_no_sec, merged_closes)
        assert "IT" in rrg_res
        assert rrg_res["IT"].advances == 2

