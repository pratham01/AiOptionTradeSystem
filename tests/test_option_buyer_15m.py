from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.backtesting.option_buyer_15m import OptionBuyer15mResearch


def test_option_buyer_15m_research_outputs_summary(tmp_path: Path):
    file_path = tmp_path / "NSE_NIFTY50-INDEX_3min_2024.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 09:15:00", periods=400, freq="3min"),
            "open": list(range(100, 500)),
            "high": list(range(101, 501)),
            "low": list(range(99, 499)),
            "close": list(range(100, 300)) + list(range(299, 99, -1)),
            "volume": [0] * 400,
        }
    )
    df.to_csv(file_path, index=False)

    artifacts = OptionBuyer15mResearch().run(tmp_path, tmp_path / "reports")

    assert not artifacts.summary.empty
    assert set(artifacts.summary["strategy"]) == {
        "st_flip",
        "st_flip_confirmed",
        "st_flip_confirmed_smc",
        "orb_breakout",
        "pullback_reclaim",
        "prev_day_breakout",
        "compression_breakout",
    }
