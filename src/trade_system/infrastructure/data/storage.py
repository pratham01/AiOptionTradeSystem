from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("/", "_")


class CsvDataCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def historical_path(self, symbol: str, resolution: str) -> Path:
        return self.root / f"{_safe_symbol(symbol)}_{_format_resolution(resolution)}_historical.csv"

    def historical_year_path(self, symbol: str, resolution: str, year: int) -> Path:
        return self.root / f"{_safe_symbol(symbol)}_{_format_resolution(resolution)}_{year}.csv"

    def live_ticks_path(self, symbol: str, trade_date: str) -> Path:
        return self.root / f"{_safe_symbol(symbol)}_ticks_{trade_date}.csv"

    def live_bars_path(self, symbol: str, timeframe_minutes: int, trade_date: str) -> Path:
        return self.root / f"{_safe_symbol(symbol)}_{timeframe_minutes}min_{trade_date}.csv"

    def append_frame(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df_to_write = self._drop_symbol_column(df)
        if path.exists() and "timestamp" in df_to_write.columns:
            existing = pd.read_csv(path, parse_dates=["timestamp"])
            merged = pd.concat([existing, df_to_write], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
            merged.to_csv(path, index=False)
        else:
            header = not path.exists()
            df_to_write.to_csv(path, mode="a", header=header, index=False)
        LOGGER.info("Wrote %s rows to %s", len(df_to_write), path)

    def write_historical(self, df: pd.DataFrame, symbol: str, resolution: str) -> Path:
        path = self.historical_path(symbol, resolution)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_to_write = self._drop_symbol_column(df)
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["timestamp"])
            merged = pd.concat([existing, df_to_write], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            merged.to_csv(path, index=False)
        else:
            df_to_write.sort_values("timestamp").to_csv(path, index=False)
        LOGGER.info("Historical dataset saved to %s", path)
        return path

    def write_historical_yearly(self, df: pd.DataFrame, symbol: str, resolution: str) -> list[Path]:
        df_to_write = self._drop_symbol_column(df)
        if df_to_write.empty:
            return []

        frame = df_to_write.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["year"] = frame["timestamp"].dt.year
        written_paths: list[Path] = []

        for year, year_df in frame.groupby("year", sort=True):
            path = self.historical_year_path(symbol, resolution, int(year))
            payload = year_df.drop(columns=["year"]).sort_values("timestamp")
            if path.exists():
                existing = pd.read_csv(path, parse_dates=["timestamp"])
                payload = pd.concat([existing, payload], ignore_index=True)
                payload = payload.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            payload.to_csv(path, index=False)
            LOGGER.info("Historical yearly dataset saved to %s", path)
            written_paths.append(path)

        return written_paths

    @staticmethod
    def _drop_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
        if "symbol" in df.columns:
            return df.drop(columns=["symbol"])
        return df.copy()


def _format_resolution(resolution: str) -> str:
    if resolution.isdigit():
        return f"{resolution}min"
    return resolution.lower()
