"""
DataSyncService — Automatically backfills missing data in the SQLite DB.

Runs at startup and fills gaps in all OHLCV tables (1m, 15m, daily) for
all configured symbols. This is the fix for the "UI shows no data" problem
when the live bot hasn't been running.

Usage:
    python -m trade_system.interfaces.live.data_sync_service
    
Or call programmatically:
    from trade_system.interfaces.live.data_sync_service import DataSyncService
    svc = DataSyncService(broker, settings)
    svc.backfill_all()
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import (
    Ohlcv1m, Ohlcv15m, OhlcvDaily
)
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Max lookback when syncing (to avoid fetching years of data on first run)
MAX_BACKFILL_DAYS = 60

# How many days back to check for gaps
GAP_CHECK_DAYS = 10

RESOLUTION_CONFIG = {
    "15": {
        "model": Ohlcv15m,
        "table": "ohlcv_15m",
        "fyers_resolution": "15",
        "description": "15-minute",
    },
    "D": {
        "model": OhlcvDaily,
        "table": "ohlcv_daily",
        "fyers_resolution": "D",
        "description": "Daily",
    },
}


def _is_trading_day(d: date) -> bool:
    """Return True if d is a weekday (simple check, ignores holidays)."""
    return d.weekday() < 5


def _trading_days_between(start: date, end: date) -> list[date]:
    """Return list of weekday dates from start to end (inclusive)."""
    result = []
    current = start
    while current <= end:
        if _is_trading_day(current):
            result.append(current)
        current += timedelta(days=1)
    return result


def _get_last_db_date(engine, table: str, symbol: str) -> date | None:
    """Get the most recent date we have data for a symbol in a table."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT MAX(date(timestamp)) FROM {table} WHERE symbol = :sym"),
                {"sym": symbol},
            ).fetchone()
            if row and row[0]:
                return date.fromisoformat(row[0])
    except Exception as e:
        LOGGER.warning("Failed to query last date for %s in %s: %s", symbol, table, e)
    return None


def _upsert_records(engine, model, records: list[dict]) -> int:
    """Upsert a list of records into the given model table."""
    if not records:
        return 0
    try:
        with Session(engine) as session:
            stmt = insert(model).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timestamp"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )
            session.execute(stmt)
            session.commit()
        return len(records)
    except Exception as e:
        LOGGER.error("Upsert failed: %s", e)
        return 0


class DataSyncService:
    """
    Backfills missing OHLCV data from Fyers API into SQLite.

    Designed to run at startup (before the live bot starts) and during
    off-market hours to keep the DB current.
    """

    def __init__(self, broker, settings: Settings) -> None:
        self.broker = broker
        self.settings = settings
        self.engine = get_engine()

    def backfill_all(self, resolutions: list[str] | None = None) -> dict[str, int]:
        """
        Backfill all gaps for all symbols across all resolutions.

        Returns dict of {resolution: total_rows_written}.
        """
        resolutions = resolutions or list(RESOLUTION_CONFIG.keys())
        symbols = self._get_all_symbols()
        totals: dict[str, int] = {}

        LOGGER.info(
            "DataSyncService: Starting backfill for %d symbols across resolutions %s",
            len(symbols), resolutions
        )

        for res in resolutions:
            cfg = RESOLUTION_CONFIG.get(res)
            if not cfg:
                continue

            LOGGER.info("--- Syncing %s data ---", cfg["description"])
            total = 0

            for symbol in symbols:
                written = self._backfill_symbol(symbol, res, cfg)
                total += written
                # Brief throttle to respect Fyers rate limits
                if written > 0:
                    time.sleep(0.4)

            LOGGER.info(
                "Finished %s sync: %d rows written across %d symbols",
                cfg["description"], total, len(symbols)
            )
            totals[res] = total

        return totals

    def _backfill_symbol(self, symbol: str, resolution: str, cfg: dict) -> int:
        """Backfill a single symbol for a given resolution. Returns rows written."""
        today = date.today()

        # Determine from_date: one day after the last date we have
        last_date = _get_last_db_date(self.engine, cfg["table"], symbol)
        if last_date:
            from_date = last_date + timedelta(days=1)
        else:
            # No data at all — limit how far back we go
            from_date = today - timedelta(days=MAX_BACKFILL_DAYS)

        # Don't fetch future dates or if already up to date
        if from_date > today:
            return 0

        # For intraday resolutions don't fetch weekends
        if resolution != "D":
            trading_days = _trading_days_between(from_date, today)
            if not trading_days:
                return 0
            from_date = trading_days[0]

        try:
            df = self.broker.fetch_history(
                symbol=symbol,
                resolution=cfg["fyers_resolution"],
                range_from=from_date.strftime("%Y-%m-%d"),
                range_to=today.strftime("%Y-%m-%d"),
                date_format="1",
            )
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["rate limit", "429", "limit exceeded", "too many"]):
                LOGGER.warning("Rate limit hit for %s/%s — sleeping 30s", symbol, resolution)
                time.sleep(30)
            else:
                LOGGER.debug("Failed to fetch %s/%s: %s", symbol, resolution, e)
            return 0

        if df is None or df.empty:
            return 0

        records = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row["timestamp"], format="mixed").to_pydatetime().replace(tzinfo=None)
                records.append({
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })
            except Exception:
                continue

        if records:
            written = _upsert_records(self.engine, cfg["model"], records)
            if written > 0:
                LOGGER.info(
                    "Backfilled %d %s bars for %s (from %s)",
                    written, cfg["description"], symbol, from_date
                )
            return written
        return 0

    def _get_all_symbols(self) -> list[str]:
        """Get the full symbol list: FO universe + index symbols."""
        try:
            fo_syms = get_fo_universe()
        except Exception:
            fo_syms = []
        index_syms = getattr(self.settings, "index_symbols", [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX",
            "NSE:FINNIFTY-INDEX",
            "BSE:SENSEX-INDEX",
        ])
        return list(dict.fromkeys(fo_syms + index_syms))  # deduplicate, preserve order

    def get_data_health_report(self) -> dict:
        """
        Check data freshness for key symbols and resolutions.
        Returns a report dict suitable for display in the dashboard.
        """
        report = {}
        key_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
        today = date.today()

        for res, cfg in RESOLUTION_CONFIG.items():
            table = cfg["table"]
            for sym in key_symbols:
                last_date = _get_last_db_date(self.engine, table, sym)
                gap_days = (today - last_date).days if last_date else None
                report[f"{sym}_{res}"] = {
                    "symbol": sym,
                    "resolution": res,
                    "last_date": last_date.isoformat() if last_date else "never",
                    "gap_days": gap_days,
                    "is_current": gap_days is not None and gap_days <= 1,
                }
        return report


def run_startup_sync() -> None:
    """
    Entry point: authenticate and run a full backfill.
    Call this at bot startup before starting the websocket.
    """
    from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
    from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
    from trade_system.shared.utils.logging_utils import configure_logging

    configure_logging()
    settings = Settings.load()
    settings.ensure_directories()

    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    if not token:
        LOGGER.error("Cannot backfill: no valid Fyers token")
        return

    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
    )

    svc = DataSyncService(broker, settings)

    # First show what's missing
    health = svc.get_data_health_report()
    for key, info in health.items():
        status = "✅ Current" if info["is_current"] else f"⚠️  Gap: {info['gap_days']} days"
        LOGGER.info("[%s] %s: %s — last date: %s", info["resolution"], info["symbol"], status, info["last_date"])

    # Backfill
    totals = svc.backfill_all()
    LOGGER.info("Backfill complete: %s", totals)


if __name__ == "__main__":
    run_startup_sync()
