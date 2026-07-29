"""
LiveDataService — Unified data access layer for dashboard pages.

Reads live prices from the shared live state file (written by the collector).
Falls back to broker API only when the live bot is not running.
Reads historical intraday OHLCV from SQLite.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from trade_system.shared.live_state import LiveStateReader, SymbolState
from trade_system.domains.trading.domain.ports.broker import MarketQuote

LOGGER = logging.getLogger(__name__)


class LiveDataService:
    """
    Single entry point for all dashboard data needs.

    Priority order for live prices:
      1. Live state file (if bot is running and state is fresh)
      2. Broker API fallback (if bot is not running)
    """

    def __init__(self) -> None:
        self._reader = LiveStateReader()

    def is_live_bot_running(self) -> bool:
        """Check if the live trading bot is currently running."""
        return self._reader.is_bot_running()

    def get_current_prices(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """
        Get current prices for the given symbols.

        Returns MarketQuote objects for compatibility with existing dashboard code.
        Reads from the live state file first; falls back to broker API.
        """
        # Try live state first
        if self._reader.is_fresh(max_age_seconds=15):
            states = self._reader.get_current_prices(symbols)
            if states and all(s in states for s in symbols):
                return self._states_to_quotes(states)

        # Fallback to broker API
        return self._fetch_from_broker(symbols)

    def get_system_health(self) -> dict[str, Any]:
        """Get live system health metrics."""
        return self._reader.get_system_health()

    def get_intraday_ohlcv(
        self,
        symbol: str,
        resolution: str = "1",
        target_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Read intraday OHLCV data from SQLite for a given symbol and date.

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        target = target_date or date.today()
        try:
            from trade_system.domains.market_data.infrastructure.database.connection import get_engine
            from trade_system.domains.market_data.infrastructure.database.repository import RESOLUTION_TO_TABLE
            from sqlalchemy import text

            table = RESOLUTION_TO_TABLE.get(resolution.upper(), "ohlcv_1m")
            engine = get_engine()

            query = text(
                f"SELECT timestamp, open, high, low, close, volume "
                f"FROM {table} "
                f"WHERE symbol = :symbol "
                f"AND date(timestamp) = :target_date "
                f"ORDER BY timestamp"
            )

            with engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol, "target_date": target.isoformat()})
                rows = result.fetchall()

            if not rows:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        except Exception as e:
            LOGGER.error("Failed to read intraday OHLCV for %s: %s", symbol, e)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def get_latest_state(self) -> dict[str, Any] | None:
        """Get the full raw live state for advanced dashboard use."""
        return self._reader.get_raw_state()

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _states_to_quotes(states: dict[str, SymbolState]) -> dict[str, MarketQuote]:
        """Convert SymbolState objects to MarketQuote for backward compatibility."""
        quotes: dict[str, MarketQuote] = {}
        for sym, s in states.items():
            exchange = sym.split(":")[0] if ":" in sym else "NSE"
            quotes[sym] = MarketQuote(
                symbol=sym,
                exchange=exchange,
                last_price=s.ltp,
                open=s.open,
                high=s.high,
                low=s.low,
                close=s.close,
                previous_close=s.previous_close,
                volume=s.volume,
                change=s.change,
                change_percent=s.change_percent,
                timestamp=datetime.now(),
                bid=0.0,
                ask=0.0,
                bid_qty=0,
                ask_qty=0,
                ltp=s.ltp,
            )
        return quotes

    @staticmethod
    def _fetch_from_broker(symbols: list[str]) -> dict[str, MarketQuote]:
        """Fallback: fetch quotes from broker API when live bot is not running."""
        try:
            from trade_system.interfaces.dashboard.shared_broker import fetch_live_quotes
            return fetch_live_quotes(tuple(symbols))
        except Exception as e:
            LOGGER.error("Broker API fallback failed: %s", e)
            return {}
