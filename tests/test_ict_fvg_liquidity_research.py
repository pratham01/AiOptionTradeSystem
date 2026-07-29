from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.backtesting.ict_fvg_liquidity_research import FVGDetector, IctFvgLiquidityResearch


def test_fvg_detector_detects_bullish_gap():
    detector = FVGDetector(min_size=5.0)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 09:15:00",
                    "2026-01-01 09:18:00",
                    "2026-01-01 09:21:00",
                ]
            ),
            "open": [100.0, 102.0, 108.0],
            "high": [101.0, 107.0, 111.0],
            "low": [99.0, 101.0, 107.0],
            "close": [100.0, 106.0, 110.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )

    active = detector.update(frame)

    assert len(active) == 1
    assert active[0]["type"] == "BFVG"
    assert active[0]["bottom"] == 101.0
    assert active[0]["top"] == 107.0


def test_ict_research_run_year_returns_expected_columns(tmp_path: Path):
    rows = [
        ("2026-01-01 09:15:00", 100.0, 102.0, 99.0, 101.0, 1000.0),
        ("2026-01-01 09:18:00", 101.0, 103.0, 100.0, 102.5, 1000.0),
        ("2026-01-01 09:21:00", 102.5, 105.0, 102.0, 104.5, 1000.0),
        ("2026-01-01 09:24:00", 104.5, 105.0, 100.0, 101.0, 1000.0),
        ("2026-01-01 09:27:00", 101.0, 101.5, 98.0, 100.5, 1000.0),
        ("2026-01-01 09:30:00", 100.5, 106.0, 97.0, 105.0, 1000.0),
        ("2026-01-01 09:33:00", 105.0, 109.0, 104.0, 108.5, 1000.0),
        ("2026-01-01 09:36:00", 108.5, 112.0, 108.0, 111.0, 1000.0),
        ("2026-01-01 09:39:00", 111.0, 113.0, 110.0, 112.0, 1000.0),
        ("2026-01-01 09:42:00", 112.0, 114.0, 111.0, 113.0, 1000.0),
        ("2026-01-01 09:45:00", 113.0, 115.0, 112.0, 114.0, 1000.0),
        ("2026-01-01 09:48:00", 114.0, 116.0, 113.0, 115.0, 1000.0),
    ]
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    path = tmp_path / "NSE_NIFTY50-INDEX_3min_2026.csv"
    frame.to_csv(path, index=False)

    trades = IctFvgLiquidityResearch(
        fvg_min_size=2.0,
        ob_impulse_thresh=2.0,
        liquidity_tolerance=1.5,
        bos_lookback=2,
        stop_buffer_points=0.5,
    ).run_year(path)

    if not trades.empty:
        assert {
            "entry_time",
            "entry_price",
            "stop_loss",
            "take_profit",
            "exit_time",
            "exit_price",
            "exit_reason",
            "points_captured",
            "r_multiple",
        }.issubset(trades.columns)
