from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.backtesting.big_move_detector import UpsideBigMoveResearch


def test_upside_big_move_research_creates_threshold_summary(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-01-01 09:15:00", periods=600, freq="3min")
    close = pd.Series(range(100, 700), dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": [1000 + (i % 10) * 50 for i in range(len(timestamps))],
        }
    )
    file_path = tmp_path / "NSE_NIFTY50-INDEX_3min_2026.csv"
    frame.to_csv(file_path, index=False)

    artifacts = UpsideBigMoveResearch().run(tmp_path, tmp_path / "reports")

    assert not artifacts.summary.empty
    assert set(artifacts.summary["threshold"]) == {60, 70, 80, 90}
    assert artifacts.summary_path.exists()
    assert artifacts.report_path.exists()
