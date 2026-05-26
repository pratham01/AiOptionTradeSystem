from pathlib import Path

import pandas as pd

from trade_system.application.backtesting.smc_signal_research import SmcSignalResearch


def test_smc_signal_research_runs(tmp_path: Path) -> None:
    file_path = tmp_path / "NSE_NIFTY50-INDEX_3min_2026.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:15:00", periods=400, freq="3min"),
            "open": [100 + i * 0.2 for i in range(400)],
            "high": [101 + i * 0.2 for i in range(400)],
            "low": [99 + i * 0.2 for i in range(400)],
            "close": [100.5 + i * 0.2 for i in range(400)],
            "volume": [1000] * 400,
        }
    )
    df.to_csv(file_path, index=False)

    artifacts = SmcSignalResearch().run(tmp_path, tmp_path / "reports")

    assert artifacts.summary_path.exists()
    assert artifacts.report_path.exists()
    assert not artifacts.summary.empty
