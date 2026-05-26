from pathlib import Path

import pandas as pd

from trade_system.infrastructure.data.storage import CsvDataCatalog


def test_write_historical_yearly_splits_by_year(tmp_path: Path):
    catalog = CsvDataCatalog(tmp_path)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2018-01-01 09:15:00", "2018-12-31 15:27:00", "2019-01-01 09:15:00"]
            ),
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [10, 20, 30],
            "symbol": ["NSE:NIFTY50-INDEX"] * 3,
        }
    )

    paths = catalog.write_historical_yearly(df, "NSE:NIFTY50-INDEX", "3")

    assert [path.name for path in paths] == [
        "NSE_NIFTY50-INDEX_3min_2018.csv",
        "NSE_NIFTY50-INDEX_3min_2019.csv",
    ]
    saved_2018 = pd.read_csv(tmp_path / "NSE_NIFTY50-INDEX_3min_2018.csv")
    saved_2019 = pd.read_csv(tmp_path / "NSE_NIFTY50-INDEX_3min_2019.csv")
    assert len(saved_2018) == 2
    assert len(saved_2019) == 1
    assert "symbol" not in saved_2018.columns
