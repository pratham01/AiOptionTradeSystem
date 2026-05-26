from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd

from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.infrastructure.data.storage import CsvDataCatalog

LOGGER = logging.getLogger(__name__)


class HistoricalDataService:
    def __init__(self, broker: FyersBrokerClient, catalog: CsvDataCatalog) -> None:
        self.broker = broker
        self.catalog = catalog

    def collect(
        self,
        symbol: str,
        resolution: str,
        from_date: date,
        to_date: date,
        chunk_days: int = 365,
        sleep_seconds: float = 0.25,
        year_wise: bool = False,
    ) -> pd.DataFrame:
        all_frames: list[pd.DataFrame] = []
        effective_chunk_days = chunk_days
        if resolution.isdigit():
            effective_chunk_days = min(chunk_days, 20)
        current = from_date
        while current <= to_date:
            chunk_end = min(current + timedelta(days=effective_chunk_days - 1), to_date)
            LOGGER.info(
                "Fetching %s %s from %s to %s",
                symbol,
                resolution,
                current.isoformat(),
                chunk_end.isoformat(),
            )
            if resolution.isdigit():
                frame = self.broker.fetch_history(
                    symbol=symbol,
                    resolution=resolution,
                    range_from=current.isoformat(),
                    range_to=chunk_end.isoformat(),
                    date_format="1",
                )
            else:
                start_epoch = int(datetime.combine(current, datetime.min.time()).timestamp())
                end_epoch = int(datetime.combine(chunk_end, datetime.max.time()).timestamp())
                frame = self.broker.fetch_history_by_epoch(symbol, resolution, start_epoch, end_epoch)
            if not frame.empty:
                all_frames.append(frame)
            current = chunk_end + timedelta(days=1)
            if current <= to_date and sleep_seconds:
                time.sleep(sleep_seconds)

        if not all_frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        if year_wise:
            self.catalog.write_historical_yearly(combined, symbol, resolution)
        else:
            self.catalog.write_historical(combined, symbol, resolution)
        return combined
