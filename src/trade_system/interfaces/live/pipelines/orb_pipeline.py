"""
OrbPipeline — Deprecated (ORB logic removed).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
import pandas as pd

LOGGER = logging.getLogger(__name__)


class OrbPipeline:
    """Deprecated no-op ORB tracker."""

    def __init__(self, notifier=None, confirmed_notifier=None) -> None:
        self.notifier = notifier
        self.confirmed_notifier = confirmed_notifier

    def register_symbol(self, symbol: str) -> None:
        pass

    def reset(self, symbols: list[str]) -> None:
        pass

    def get_candle(self, symbol: str) -> dict | None:
        return None

    def on_bar(
        self,
        symbol: str,
        bar: pd.Series,
        minute_data: dict[str, pd.DataFrame],
        current_date: date,
    ) -> None:
        pass

    def on_tick(self, symbol: str, price: float, tick_time: datetime) -> None:
        pass
