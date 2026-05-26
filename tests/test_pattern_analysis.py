from pathlib import Path

import pandas as pd

from trade_system.application.analysis import MarketPatternAnalyzer


def test_pattern_analyzer_outputs_files(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:15:00", periods=500, freq="3min"),
            "open": list(range(100, 600)),
            "high": list(range(101, 601)),
            "low": list(range(99, 599)),
            "close": list(range(100, 600)),
            "volume": [0] * 500,
        }
    )
    df.to_csv(data_dir / "NSE_NIFTY50-INDEX_3min_2024.csv", index=False)

    artifacts = MarketPatternAnalyzer().run(data_dir, tmp_path / "reports")

    assert artifacts.daily_features_path.exists()
    assert artifacts.multi_timeframe_levels_path.exists()
    assert artifacts.daily_summary_path.exists()
    assert artifacts.probability_report_path.exists()
