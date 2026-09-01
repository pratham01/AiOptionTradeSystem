"""
DataSyncService — Automatically backfills missing data in the SQLite DB.

Runs at startup and fills gaps in OHLCV tables (15m, daily, 5m) for
all configured F&O and index symbols. Guarantees data completeness before
any live computations or indicator evaluations occur.

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
from datetime import date, datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import (
    Ohlcv1m, Ohlcv5m, Ohlcv15m, OhlcvDaily
)
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Max lookback when syncing (to avoid fetching excessive data on first run)
MAX_BACKFILL_DAYS = 60

# How many days back to check for gaps
GAP_CHECK_DAYS = 10

RESOLUTION_CONFIG = {
    "15": {
        "model": Ohlcv15m,
        "table": "ohlcv_15m",
        "fyers_resolution": "15",
        "description": "15-minute",
        "expected_bars_per_day": 25,
    },
    "D": {
        "model": OhlcvDaily,
        "table": "ohlcv_daily",
        "fyers_resolution": "D",
        "description": "Daily",
        "expected_bars_per_day": 1,
    },
    "5": {
        "model": Ohlcv5m,
        "table": "ohlcv_5m",
        "fyers_resolution": "5",
        "description": "5-minute",
        "expected_bars_per_day": 75,
    },
}


def _is_trading_day(d: date) -> bool:
    """Return True if d is a weekday (Monday=0 to Friday=4)."""
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


def get_expected_last_trading_date(now: Optional[datetime] = None) -> date:
    """
    Determine the most recent completed market trading day.
    - If today is Monday before 09:15 IST, the last completed trading day is Friday.
    - If today is a weekend, the last completed trading day is Friday.
    - If today is a weekday before 09:15 IST, the last completed trading day is yesterday.
    - If today is a weekday after 09:15 IST, today's session is in progress or completed.
    """
    dt_now = now or datetime.now(IST)
    today = dt_now.date() if hasattr(dt_now, "date") else dt_now
    current_time = dt_now.time() if hasattr(dt_now, "time") else dt_time(10, 0)
    market_open = dt_time(9, 15)

    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    elif today.weekday() == 0 and current_time < market_open:  # Monday pre-market
        return today - timedelta(days=3)
    elif current_time < market_open:  # Tue-Fri pre-market
        return today - timedelta(days=1)
    else:
        return today


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


def _get_symbol_date_gaps(engine, table: str, symbol: str, lookback_days: int = 30, expected_bars: int = 1) -> list[date]:
    """Identify missing or incomplete trading dates for a symbol within the lookback window."""
    try:
        since_date = date.today() - timedelta(days=lookback_days)
        with engine.connect() as conn:
            # Baseline reference trading days from NIFTY index
            idx_rows = conn.execute(
                text("SELECT DISTINCT date(timestamp) as dt FROM ohlcv_daily WHERE symbol = 'NSE:NIFTY50-INDEX' AND timestamp >= :since ORDER BY dt ASC"),
                {"since": since_date.isoformat()}
            ).fetchall()
            trading_days = [date.fromisoformat(r[0]) for r in idx_rows] if idx_rows else _trading_days_between(since_date, date.today())

            # Actual days and bar counts for symbol
            sym_rows = conn.execute(
                text(f"SELECT date(timestamp) as dt, count(*) as cnt FROM {table} WHERE symbol = :sym AND timestamp >= :since GROUP BY dt"),
                {"sym": symbol, "since": since_date.isoformat()}
            ).fetchall()
            sym_counts = {date.fromisoformat(r[0]): r[1] for r in sym_rows}

            today = date.today()
            missing_dates = []
            min_bars = max(1, int(expected_bars * 0.7))
            for d in trading_days:
                if d not in sym_counts:
                    missing_dates.append(d)
                elif d != today and sym_counts[d] < min_bars:
                    missing_dates.append(d)
            return sorted(missing_dates)
    except Exception as e:
        LOGGER.warning("Gap check error for %s in %s: %s", symbol, table, e)
        return []


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
    Backfills missing OHLCV data from Broker API into SQLite.

    Ensures data completeness across the full F&O universe and Index symbols
    before any computations or strategy processing can begin.
    """

    def __init__(self, broker: Any, settings: Settings) -> None:
        self.broker = broker
        self.settings = settings
        self.engine = get_engine()

    def backfill_all(self, resolutions: list[str] | None = None) -> dict[str, int]:
        """
        Backfill all gaps for all symbols (F&O universe + indices) across resolutions.

        Returns dict of {resolution: total_rows_written}.
        """
        resolutions = resolutions or ["15", "D"]
        symbols = self._get_all_symbols()
        return self.backfill_symbols(symbols=symbols, resolutions=resolutions)

    def backfill_symbols(self, symbols: list[str], resolutions: list[str] | None = None) -> dict[str, int]:
        """
        Backfill specific symbols for the specified resolutions.
        """
        resolutions = resolutions or ["15", "D"]
        totals: dict[str, int] = {}

        LOGGER.info(
            "DataSyncService: Starting backfill for %d symbols across resolutions %s",
            len(symbols), resolutions
        )

        for res in resolutions:
            cfg = RESOLUTION_CONFIG.get(res)
            if not cfg:
                continue

            LOGGER.info("--- Syncing %s data for %d symbols ---", cfg["description"], len(symbols))
            total = 0
            backfilled_count = 0

            for idx, symbol in enumerate(symbols):
                written = self._backfill_symbol(symbol, res, cfg)
                total += written
                if written > 0:
                    backfilled_count += 1
                    time.sleep(0.25)  # Rate limit throttle: 4 requests/sec safe limit

            LOGGER.info(
                "Finished %s sync: %d rows written across %d/%d symbols",
                cfg["description"], total, backfilled_count, len(symbols)
            )
            totals[res] = total

        return totals

    def _backfill_symbol(self, symbol: str, resolution: str, cfg: dict) -> int:
        """Backfill a single symbol for a given resolution. Returns rows written."""
        today = date.today()
        expected_last_date = get_expected_last_trading_date()
        expected_bars = cfg.get("expected_bars_per_day", 1)

        # Check internal gaps within the lookback window
        internal_gaps = _get_symbol_date_gaps(self.engine, cfg["table"], symbol, lookback_days=MAX_BACKFILL_DAYS, expected_bars=expected_bars)
        last_date = _get_last_db_date(self.engine, cfg["table"], symbol)

        if internal_gaps:
            from_date = min(internal_gaps)
        elif last_date:
            if last_date >= expected_last_date:
                # Already up to date and no internal gaps
                return 0
            from_date = last_date + timedelta(days=1)
        else:
            # No data at all — limit how far back we go
            from_date = today - timedelta(days=MAX_BACKFILL_DAYS)

        if from_date > today:
            return 0

        # For intraday resolutions ensure we start on a trading day
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
                LOGGER.warning("Rate limit hit for %s/%s — pausing 30s", symbol, resolution)
                time.sleep(30)
            else:
                LOGGER.debug("Failed to fetch %s/%s: %s", symbol, resolution, e)
            return 0

        if df is None or df.empty:
            return 0

        # Data validation and normalization
        records = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row["timestamp"], format="mixed").to_pydatetime().replace(tzinfo=None)
                op = float(row["open"])
                hi = float(row["high"])
                lo = float(row["low"])
                cl = float(row["close"])
                vol = float(row.get("volume", 0) or 0.0)

                # Skip invalid or corrupt price candles
                if op <= 0 or hi <= 0 or lo <= 0 or cl <= 0 or hi < lo:
                    continue

                records.append({
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": op,
                    "high": max(hi, op, cl),
                    "low": min(lo, op, cl),
                    "close": cl,
                    "volume": max(vol, 0.0),
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
        return list(dict.fromkeys(index_syms + fo_syms))

    def get_data_health_report(
        self,
        symbols: Optional[list[str]] = None,
        resolutions: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """
        Check data freshness across requested symbols (or full FO universe + indices).
        Returns a report dict with gap analysis per symbol and resolution.
        """
        report = {}
        target_symbols = symbols or self._get_all_symbols()
        target_resolutions = resolutions or ["15", "D"]
        expected_date = get_expected_last_trading_date()

        for res in target_resolutions:
            cfg = RESOLUTION_CONFIG.get(res)
            if not cfg:
                continue
            table = cfg["table"]
            expected_bars = cfg.get("expected_bars_per_day", 1)
            for sym in target_symbols:
                last_date = _get_last_db_date(self.engine, table, sym)
                internal_gaps = _get_symbol_date_gaps(self.engine, table, sym, lookback_days=GAP_CHECK_DAYS, expected_bars=expected_bars)
                is_current = (last_date is not None and last_date >= expected_date and len(internal_gaps) == 0)
                if last_date:
                    gap_days = len(internal_gaps) if internal_gaps else (max(0, len(_trading_days_between(last_date + timedelta(days=1), expected_date))) if not is_current else 0)
                else:
                    gap_days = MAX_BACKFILL_DAYS

                report[f"{sym}_{res}"] = {
                    "symbol": sym,
                    "resolution": res,
                    "last_date": last_date.isoformat() if last_date else "never",
                    "expected_date": expected_date.isoformat(),
                    "gap_days": gap_days,
                    "internal_gaps": [d.isoformat() for d in internal_gaps],
                    "is_current": is_current,
                }
        return report

    def find_symbols_needing_sync(
        self,
        symbols: Optional[list[str]] = None,
        resolutions: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """
        Identifies exact symbols that have gaps per resolution.
        Returns dict: {resolution: [symbols_with_gaps]}
        """
        health = self.get_data_health_report(symbols=symbols, resolutions=resolutions)
        gaps_by_res: dict[str, list[str]] = {}
        for entry in health.values():
            if not entry["is_current"]:
                res = entry["resolution"]
                sym = entry["symbol"]
                gaps_by_res.setdefault(res, []).append(sym)
        return gaps_by_res


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
    symbols_with_gaps = svc.find_symbols_needing_sync()
    total_gaps = sum(len(syms) for syms in symbols_with_gaps.values())

    if total_gaps == 0:
        LOGGER.info("✅ All F&O and Index historical data is current. No backfill needed.")
    else:
        LOGGER.info("⚠️ Detected gaps in %d symbol-resolution pairs. Starting backfill...", total_gaps)
        totals = svc.backfill_all()
        LOGGER.info("Backfill complete: %s", totals)


if __name__ == "__main__":
    run_startup_sync()

