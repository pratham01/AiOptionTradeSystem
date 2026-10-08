"""
BarStore — Centralised persistence service for OHLCV bars.

Responsibilities
----------------
* Async-safe DB writes via DbWriteWorker (non-blocking on the WS callback thread).
* CSV writes via CsvDataCatalog (daily 1-min live file + yearly N-min strategy file).
* Daily 3-min (strategy-timeframe) file maintenance (merge today's bars into the
  year-file so dashboards always have a fresh, deduplicated file to read).

Extracted from: LiveMarketDataService._flush_symbol_minute and
                LiveMarketDataService._update_intraday_yearly_3min_file
"""
from __future__ import annotations

import logging
from datetime import date, time as dt_time
from pathlib import Path

import pandas as pd

from trade_system.domains.market_data.infrastructure.database.db_write_worker import DbWriteWorker
from trade_system.domains.market_data.infrastructure.data.storage import CsvDataCatalog

LOGGER = logging.getLogger(__name__)


def _merge_3min_bars(*, existing: pd.DataFrame, today_bars: pd.DataFrame) -> pd.DataFrame:
    """
    Merge today's completed N-min bars into an existing year-file DataFrame.

    Rules:
    - Keep all rows from *existing* that are NOT from today.
    - Append today_bars (which may be a subset of the full day, e.g. bars up to now).
    - Deduplicate on timestamp, preferring the freshest value (keep='last').
    - Sort ascending by timestamp.
    """
    if existing.empty:
        return today_bars.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    today_date = today_bars["timestamp"].dt.date.iloc[0]
    prev = existing[pd.to_datetime(existing["timestamp"]).dt.date != today_date]
    merged = pd.concat([prev, today_bars], ignore_index=True)
    return merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")


class BarStore:
    """
    Owns all OHLCV bar persistence for a given set of symbols.

    Parameters
    ----------
    db_worker : DbWriteWorker
        Pre-started async DB writer.  BarStore does NOT start/stop it.
    catalog : CsvDataCatalog
        Provides symbol-specific file paths.
    strategy_timeframe_minutes : int
        The N-min strategy timeframe (typically 3).
    market_start : dt_time
        Session open (e.g. time(9, 15)).  Used to filter intraday bars.
    market_end : dt_time
        Session close (e.g. time(15, 30)).
    """

    def __init__(
        self,
        db_worker: DbWriteWorker,
        catalog: CsvDataCatalog,
        strategy_timeframe_minutes: int,
        market_start: dt_time,
        market_end: dt_time,
    ) -> None:
        self._db_worker = db_worker
        self._catalog = catalog
        self._strategy_tf = strategy_timeframe_minutes
        self._market_start = market_start
        self._market_end = market_end

    # ------------------------------------------------------------------
    # 1-minute bar persistence (called on every completed minute)
    # ------------------------------------------------------------------

    def save_1min_bar(
        self,
        symbol: str,
        minute_bar: pd.Series,
        trade_date: date,
    ) -> None:
        """
        Persist a completed 1-min bar.

        1. Appends the bar to the daily live-bars CSV file.
        2. Enqueues the bar for async DB insert (resolution="1").

        Parameters
        ----------
        symbol : str
            Fyers symbol string (e.g. ``NSE:NIFTY50-INDEX``).
        minute_bar : pd.Series
            Bar produced by BarAggregator.  Index name == bar timestamp.
        trade_date : date
            Trading date (used for the daily file path).
        """
        minute_df = (
            minute_bar.to_frame().T
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        trade_date_str = trade_date.isoformat()

        # --- CSV append ---
        try:
            csv_path = self._catalog.live_bars_path(symbol, 1, trade_date_str)
            self._catalog.append_frame(minute_df, csv_path)
        except Exception as exc:
            LOGGER.error("Failed to append 1-min bar CSV for %s: %s", symbol, exc)

        # --- Async DB insert ---
        try:
            self._db_worker.enqueue(symbol, "1", minute_df.to_dict("records"))
        except Exception as exc:
            LOGGER.error("Failed to enqueue 1-min bar to DB for %s: %s", symbol, exc)

    def save_nmin_bar(
        self,
        symbol: str,
        nmin_bar: pd.Series,
        resolution: str,
    ) -> None:
        """
        Persist a completed N-min bar to the database asynchronously.
        Does NOT append to a CSV file (handled by update_strategy_file).
        """
        bar_df = (
            nmin_bar.to_frame().T
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        try:
            self._db_worker.enqueue(symbol, str(resolution), bar_df.to_dict("records"))
        except Exception as exc:
            LOGGER.error("Failed to enqueue %s-min bar to DB for %s: %s", resolution, symbol, exc)

    # ------------------------------------------------------------------
    # Strategy-timeframe bar persistence (called on every completed minute)
    # ------------------------------------------------------------------

    def update_strategy_file(
        self,
        symbol: str,
        minute_data: pd.DataFrame,
        current_date: date,
        *,
        resample_fn,
        completed_bars_fn,
        sanitize_fn,
        latest_session_minute_fn,
        timeframe_minutes: int | None = None,
    ) -> pd.DataFrame:
        """
        Resample today's 1-min data to the strategy timeframe and merge it
        into the year-level CSV file.

        Parameters
        ----------
        symbol : str
            Fyers symbol.
        minute_data : pd.DataFrame
            Full 1-min DataFrame (indexed by timestamp) for this symbol.
        current_date : date
            Today's trading date.
        resample_fn : callable
            ``resample_to_timeframe(df, minutes) -> DataFrame``.
        completed_bars_fn : callable
            ``_completed_timeframe_bars(df, minutes, latest_minute) -> DataFrame``.
        sanitize_fn : callable
            ``_sanitize_intraday_minutes(df, start, end) -> DataFrame``.
        latest_session_minute_fn : callable
            ``_latest_session_minute(df, date) -> Timestamp | None``.

        Returns
        -------
        pd.DataFrame
            Updated strategy DataFrame (indexed by timestamp, most recent 1500 rows).
            Returns empty DataFrame if nothing to save.
        """
        today_minutes = sanitize_fn(
            minute_data[minute_data.index.date == current_date],
            self._market_start,
            self._market_end,
        )
        if today_minutes.empty:
            return pd.DataFrame()

        tf = timeframe_minutes or self._strategy_tf
        path: Path = self._catalog.historical_year_path(
            symbol, str(tf), current_date.year
        )
        existing = pd.DataFrame()
        if path.exists():
            try:
                existing = pd.read_csv(path, parse_dates=["timestamp"])
                existing = sanitize_fn(
                    existing.set_index("timestamp"),
                    self._market_start,
                    self._market_end,
                ).reset_index()
            except Exception as exc:
                LOGGER.warning("Could not read strategy file %s: %s — rebuilding.", path, exc)
                existing = pd.DataFrame()

        today_bars = resample_fn(today_minutes, tf)
        latest_minute = latest_session_minute_fn(today_minutes, current_date)
        today_bars = completed_bars_fn(today_bars, tf, latest_minute).reset_index()
        if today_bars.empty:
            return pd.DataFrame()

        merged = _merge_3min_bars(existing=existing, today_bars=today_bars)
        try:
            merged.to_csv(path, index=False)
        except Exception as exc:
            LOGGER.error("Failed to write strategy file %s: %s", path, exc)

        return merged.set_index("timestamp").sort_index().tail(1500)

    # ------------------------------------------------------------------
    # Volume backfill helpers (delegated from collector)
    # ------------------------------------------------------------------

    def patch_zero_volume_bars(
        self,
        symbol: str,
        minute_data: pd.DataFrame,
        trade_date: date,
        history_fetcher,
        lookback_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Backfill zero-volume bars from the REST API for the most recent
        ``lookback_minutes`` bars of ``trade_date``.

        Parameters
        ----------
        symbol : str
        minute_data : pd.DataFrame  (indexed by timestamp)
        trade_date : date
        history_fetcher : callable  (symbol, date, min_timestamp) -> DataFrame
        lookback_minutes : int

        Returns
        -------
        pd.DataFrame
            Updated minute_data with volumes patched in-place where possible.
        """
        if minute_data.empty or "volume" not in minute_data.columns:
            return minute_data

        today_frame = minute_data[minute_data.index.date == trade_date]
        zero_mask = pd.to_numeric(today_frame["volume"], errors="coerce").fillna(0) <= 0
        recent_zeros = today_frame.loc[zero_mask].tail(lookback_minutes)
        if recent_zeros.empty:
            return minute_data

        latest_needed = pd.Timestamp(recent_zeros.index.max())
        try:
            history = history_fetcher(symbol, trade_date, min_timestamp=latest_needed)
        except Exception:
            LOGGER.exception("Failed recent volume backfill for %s on %s", symbol, trade_date)
            return minute_data

        if history.empty or "volume" not in history.columns:
            return minute_data

        updated = False
        for ts in recent_zeros.index:
            match = history[history.index == ts]
            if match.empty:
                continue
            volume = float(match.iloc[-1]["volume"] or 0.0)
            if volume <= 0:
                continue
            minute_data.loc[ts, "volume"] = volume
            updated = True

        if updated:
            today_path = self._catalog.live_bars_path(symbol, 1, trade_date.isoformat())
            payload = (
                minute_data[minute_data.index.date == trade_date]
                .reset_index()
                .rename(columns={"index": "timestamp"})
                .sort_values("timestamp")
            )
            if "symbol" in payload.columns:
                payload = payload.drop(columns=["symbol"])
            try:
                payload.to_csv(today_path, index=False)
            except Exception as exc:
                LOGGER.warning("Failed to rewrite patched volume CSV for %s: %s", symbol, exc)

        return minute_data
