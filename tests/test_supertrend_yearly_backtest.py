from pathlib import Path

import pandas as pd

from trade_system.application.backtesting.supertrend_yearly import SupertrendYearlyBacktester


def test_yearly_summary_builder(tmp_path: Path):
    file_path = tmp_path / "NSE_NIFTY50-INDEX_3min_2024.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:15:00", periods=80, freq="3min"),
            "open": list(range(100, 180)),
            "high": list(range(101, 181)),
            "low": list(range(99, 179)),
            "close": list(range(100, 140)) + list(range(139, 99, -1)),
            "volume": [0] * 80,
        }
    )
    df.to_csv(file_path, index=False)

    artifacts = SupertrendYearlyBacktester().run_on_directory(tmp_path, tmp_path / "reports")

    assert not artifacts.yearly_summary.empty
    assert artifacts.yearly_summary.iloc[0]["year"] == 2024
    assert (tmp_path / "reports" / "NSE_NIFTY50-INDEX_supertrend_7_3_2024_trades.csv").exists()
