"""
DataHealer — Automatic missing/bad data detection and repair.

Responsibilities
----------------
1. Detect gaps in ohlcv_15m by comparing against ohlcv_1m source data.
2. Detect stale ohlcv_daily rows (> 2 calendar days behind today).
3. Backfill ohlcv_15m from ohlcv_1m by resampling (no broker call needed).
4. Backfill ohlcv_daily from ohlcv_1m by resampling.
5. Detect and remove corrupt rows (zero/negative OHLCV, inverted high/low).
6. Log a healing report with counts of repaired, skipped, and cleaned rows.

Design Principles
-----------------
- Safe: all writes are idempotent (INSERT OR REPLACE).
- Non-blocking: can be called from a background thread.
- Broker-independent: primary healing path uses only existing DB data.
- Self-documenting: returns a HealReport dataclass for dashboard display.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

_MARKET_START = "09:15"
_MARKET_END = "15:30"

_RESAMPLE_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


@dataclass
class HealReport:
    """Summary of what DataHealer found and fixed."""
    run_at: datetime = field(default_factory=datetime.now)
    symbols_checked: int = 0
    missing_15m_rows: int = 0
    repaired_15m_rows: int = 0
    missing_daily_rows: int = 0
    repaired_daily_rows: int = 0
    corrupt_rows_removed: int = 0
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"DataHealer @ {self.run_at.strftime('%Y-%m-%d %H:%M')}\n"
            f"  Symbols: {self.symbols_checked}\n"
            f"  15m gaps found/repaired: {self.missing_15m_rows}/{self.repaired_15m_rows}\n"
            f"  Daily gaps found/repaired: {self.missing_daily_rows}/{self.repaired_daily_rows}\n"
            f"  Corrupt rows removed: {self.corrupt_rows_removed}\n"
            f"  Errors: {len(self.errors)}"
        )


class DataHealer:
    """
    Detects and repairs data gaps/corruption in the OHLCV DB tables.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Engine pointing to trade_system.db.
    broker : optional
        Optional broker client (reserved for future REST backfill).
    """

    def __init__(self, engine, broker=None) -> None:
        self._engine = engine
        self._broker = broker

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def heal(
        self,
        symbols: list = None,
        heal_15m: bool = True,
        heal_daily: bool = True,
        remove_corrupt: bool = True,
        lookback_days: int = 30,
    ) -> HealReport:
        """
        Run the full healing cycle.

        Parameters
        ----------
        symbols : list[str] | None
            Symbols to heal. If None, heals ALL symbols found in ohlcv_1m.
        heal_15m, heal_daily, remove_corrupt : bool
        lookback_days : int
            How far back (calendar days) to look for gaps.
        """
        report = HealReport()

        with self._engine.connect() as conn:
            if symbols is None:
                rows = conn.execute(
                    text("SELECT DISTINCT symbol FROM ohlcv_1m ORDER BY symbol")
                ).fetchall()
                symbols = [r[0] for r in rows]

        report.symbols_checked = len(symbols)
        LOGGER.info("DataHealer starting — %d symbols, lookback=%d days", len(symbols), lookback_days)

        since = (date.today() - timedelta(days=lookback_days)).isoformat()

        for sym in symbols:
            try:
                if remove_corrupt:
                    report.corrupt_rows_removed += self._remove_corrupt_rows(sym, since)
                if heal_15m:
                    missing, repaired = self._heal_15m(sym, since)
                    report.missing_15m_rows += missing
                    report.repaired_15m_rows += repaired
                if heal_daily:
                    missing, repaired = self._heal_daily(sym, since)
                    report.missing_daily_rows += missing
                    report.repaired_daily_rows += repaired
            except Exception as exc:
                LOGGER.error("DataHealer error for %s: %s", sym, exc)
                report.errors.append(f"{sym}: {exc}")

        LOGGER.info(report.summary())
        return report

    def quick_heal_today(self, symbols: list) -> HealReport:
        """Fast heal for today only — run at bot start / market open."""
        return self.heal(
            symbols=symbols,
            heal_15m=True,
            heal_daily=True,
            remove_corrupt=False,
            lookback_days=1,
        )

    # ------------------------------------------------------------------
    # Corrupt row removal
    # ------------------------------------------------------------------

    def _remove_corrupt_rows(self, symbol: str, since: str) -> int:
        total = 0
        for table in ["ohlcv_1m", "ohlcv_15m", "ohlcv_daily"]:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(f"""
                        DELETE FROM {table}
                        WHERE symbol = :sym
                          AND timestamp >= :since
                          AND (open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                               OR high < low)
                    """),
                    {"sym": symbol, "since": since},
                )
                deleted = result.rowcount
                if deleted:
                    LOGGER.warning(
                        "Removed %d corrupt rows from %s for %s",
                        deleted, table, symbol
                    )
                total += deleted
        return total

    # ------------------------------------------------------------------
    # 15m backfill from 1m
    # ------------------------------------------------------------------

    def _heal_15m(self, symbol: str, since: str) -> tuple:
        df_1m = self._load_table(symbol, "ohlcv_1m", since)
        if df_1m.empty:
            return 0, 0

        df_expected = self._resample(df_1m, "15min")
        if df_expected.empty:
            return 0, 0

        df_existing = self._load_table(symbol, "ohlcv_15m", since)
        existing_ts = (
            set(df_existing["timestamp"].dt.floor("15min").dt.strftime("%Y-%m-%d %H:%M:%S").tolist())
            if not df_existing.empty else set()
        )

        df_expected["ts_key"] = (
            df_expected["timestamp"].dt.floor("15min").dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        df_missing = df_expected[~df_expected["ts_key"].isin(existing_ts)].drop(columns=["ts_key"])
        missing_count = len(df_missing)
        if missing_count == 0:
            return 0, 0

        repaired = self._upsert_rows(symbol, "ohlcv_15m", df_missing)
        LOGGER.info("Healed 15m for %s: found=%d repaired=%d", symbol, missing_count, repaired)
        return missing_count, repaired

    # ------------------------------------------------------------------
    # Daily backfill from 1m
    # ------------------------------------------------------------------

    def _heal_daily(self, symbol: str, since: str) -> tuple:
        df_1m = self._load_table(symbol, "ohlcv_1m", since)
        if df_1m.empty:
            return 0, 0

        df_expected = self._resample_daily(df_1m)
        if df_expected.empty:
            return 0, 0

        df_existing = self._load_table(symbol, "ohlcv_daily", since)
        existing_dates = (
            set(df_existing["timestamp"].dt.date.astype(str).tolist())
            if not df_existing.empty else set()
        )

        df_expected["date_str"] = df_expected["timestamp"].dt.date.astype(str)
        df_missing = df_expected[~df_expected["date_str"].isin(existing_dates)].drop(columns=["date_str"])
        missing_count = len(df_missing)
        if missing_count == 0:
            return 0, 0

        repaired = self._upsert_rows(symbol, "ohlcv_daily", df_missing)
        LOGGER.info("Healed daily for %s: found=%d repaired=%d", symbol, missing_count, repaired)
        return missing_count, repaired

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_table(self, symbol: str, table: str, since: str) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text(f"""
                    SELECT timestamp, open, high, low, close, volume
                    FROM {table}
                    WHERE symbol = :sym AND timestamp >= :since
                    ORDER BY timestamp ASC
                """),
                conn,
                params={"sym": symbol, "since": since},
            )
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=False)
        return df

    def _resample(self, df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
        df = df_1m.copy().set_index("timestamp").sort_index()
        df = df.between_time(_MARKET_START, _MARKET_END)
        if df.empty:
            return pd.DataFrame()
        resampled = (
            df[["open", "high", "low", "close", "volume"]]
            .resample(rule, label="left", closed="left")
            .agg(_RESAMPLE_AGG)
            .dropna(subset=["open", "close"])
        )
        resampled = resampled[(resampled["close"] > 0) & (resampled["high"] >= resampled["low"])]
        return resampled.reset_index()

    def _resample_daily(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        df = df_1m.copy().set_index("timestamp").sort_index()
        df = df.between_time(_MARKET_START, _MARKET_END)
        if df.empty:
            return pd.DataFrame()
        resampled = (
            df[["open", "high", "low", "close", "volume"]]
            .resample("1D", label="left", closed="left")
            .agg(_RESAMPLE_AGG)
            .dropna(subset=["open", "close"])
        )
        resampled = resampled[(resampled["close"] > 0) & (resampled["high"] >= resampled["low"])]
        resampled = resampled.reset_index()
        resampled["timestamp"] = (
            pd.to_datetime(resampled["timestamp"].dt.date)
            + pd.Timedelta(hours=9, minutes=15)
        )
        return resampled

    def _upsert_rows(self, symbol: str, table: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            rows.append({
                "symbol":    symbol,
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open":      float(row.get("open",   0.0) or 0.0),
                "high":      float(row.get("high",   0.0) or 0.0),
                "low":       float(row.get("low",    0.0) or 0.0),
                "close":     float(row.get("close",  0.0) or 0.0),
                "volume":    float(row.get("volume", 0.0) or 0.0),
                "vix":       None,
            })
        sql = text(f"""
            INSERT OR REPLACE INTO {table}
                (symbol, timestamp, open, high, low, close, volume, vix)
            VALUES
                (:symbol, :timestamp, :open, :high, :low, :close, :volume, :vix)
        """)
        with self._engine.begin() as conn:
            conn.execute(sql, rows)
        return len(rows)
