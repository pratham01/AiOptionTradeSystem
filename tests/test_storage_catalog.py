from pathlib import Path

import pandas as pd

from trade_system.infrastructure.data.storage import CsvDataCatalog


def test_live_bar_file_naming_and_no_symbol_column(tmp_path: Path):
    catalog = CsvDataCatalog(tmp_path)
    path = catalog.live_bars_path("NSE:NIFTY50-INDEX", 1, "2026-03-20")

    assert path.name == "NSE_NIFTY50-INDEX_1min_2026-03-20.csv"

    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-20 09:15:00"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "symbol": "NSE:NIFTY50-INDEX",
            }
        ]
    )
    catalog.append_frame(df, path)

    saved = pd.read_csv(path)
    assert "symbol" not in saved.columns


def test_historical_file_naming(tmp_path: Path):
    catalog = CsvDataCatalog(tmp_path)
    path = catalog.historical_path("NSE:NIFTY50-INDEX", "1")
    assert path.name == "NSE_NIFTY50-INDEX_1min_historical.csv"


def test_append_frame_upserts_existing_timestamp(tmp_path: Path):
    catalog = CsvDataCatalog(tmp_path)
    path = catalog.live_bars_path("NSE:NIFTY50-INDEX", 1, "2026-03-20")

    first = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-20 09:15:00"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
            }
        ]
    )
    second = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-20 09:15:00"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.8,
                "volume": 25.0,
            }
        ]
    )

    catalog.append_frame(first, path)
    catalog.append_frame(second, path)

    saved = pd.read_csv(path, parse_dates=["timestamp"])
    assert len(saved) == 1
    assert float(saved.iloc[0]["close"]) == 100.8
    assert float(saved.iloc[0]["volume"]) == 25.0
