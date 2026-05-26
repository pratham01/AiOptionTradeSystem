from pathlib import Path

import pandas as pd

from trade_system.application.analysis.high_confidence_setups import HighConfidenceSetupAnalyzer


def test_high_confidence_setup_summary(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    (reports_dir / "candlestick_patterns").mkdir(parents=True)
    (reports_dir / "pattern_analysis").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    intraday = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:15:00", periods=1200, freq="3min"),
            "open": list(range(100, 1300)),
            "high": list(range(101, 1301)),
            "low": list(range(99, 1299)),
            "close": list(range(100, 700)) + list(range(699, 99, -1)),
            "volume": [0] * 1200,
        }
    )
    intraday.to_csv(data_dir / "NSE_NIFTY50-INDEX_3min_2024.csv", index=False)

    pd.DataFrame({"dummy": []}).to_csv(reports_dir / "candlestick_patterns" / "candlestick_probability_daily.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "prev_low": [100] * 10,
            "prev_high": [200] * 10,
            "prev_week_low": [95] * 10,
            "prev_week_high": [205] * 10,
            "range_vs_prev": [0.8] * 10,
        }
    ).to_csv(reports_dir / "pattern_analysis" / "nifty50_daily_pattern_features.csv", index=False)

    analyzer = HighConfidenceSetupAnalyzer()
    analyzer._build_daily_pattern_events(data_dir)
