from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.analysis.candlestick_patterns import CandlestickPatternAnalyzer


def test_candlestick_pattern_analyzer_outputs(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:15:00", periods=1000, freq="3min"),
            "open": list(range(100, 1100)),
            "high": list(range(101, 1101)),
            "low": list(range(99, 1099)),
            "close": list(range(100, 600)) + list(range(599, 99, -1)),
            "volume": [0] * 1000,
        }
    )
    df.to_csv(data_dir / "NSE_NIFTY50-INDEX_3min_2024.csv", index=False)

    artifacts = CandlestickPatternAnalyzer().run(data_dir, tmp_path / "reports")

    assert artifacts.daily_path.exists()
    assert artifacts.weekly_path.exists()
    assert artifacts.monthly_path.exists()
    assert artifacts.report_path.exists()
