"""
MarketDataPipeline — Focused module for OHLC collection, gap detection, and Supertrend calculation.

Extracted from the LiveMarketDataService monolith (collector.py).
Single Responsibility: Transform raw websocket ticks into validated OHLC bars
and compute Supertrend indicators with data-continuity protection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from trade_system.domains.strategy.application.indicators import calculate_supertrend
from trade_system.domains.market_data.domain.schemas.market_data import validate_candles
from trade_system.interfaces.live.helpers import (
    _sanitize_intraday_minutes,
    get_gap_adjusted_data,
    resample_to_timeframe,
    valid_supertrend_rows as _valid_supertrend_rows,
    create_minute_bar,
)

LOGGER = logging.getLogger("MarketDataPipeline")


@dataclass
class OHLCBar:
    """Typed OHLC bar produced by the pipeline."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_complete: bool = False


@dataclass
class SupertrendState:
    """Current Supertrend state for a symbol."""
    symbol: str
    direction: int          # 1=bullish, -1=bearish, 0=unknown
    level: float            # Current Supertrend line value
    signal: int             # 1=flip to bullish, -1=flip to bearish, 0=no change
    bar_time: pd.Timestamp | None = None
    close_price: float = 0.0
    data_quality_ok: bool = True


class MarketDataPipeline:
    """
    Owns: tick ingestion → minute bar building → gap validation → Supertrend computation.

    Usage:
        pipeline = MarketDataPipeline(
            symbols=["NSE:NIFTY50-INDEX"],
            supertrend_period=7,
            supertrend_multiplier=3,
            strategy_timeframe_minutes=15,
            market_start=time(9, 15),
            market_end=time(15, 30),
        )

        # Feed ticks from the websocket:
        completed_bar = pipeline.process_tick(symbol, tick_dict)

        # Get current Supertrend:
        st_state = pipeline.compute_supertrend(symbol)
    """

    def __init__(
        self,
        symbols: list[str],
        supertrend_period: int = 7,
        supertrend_multiplier: int = 3,
        strategy_timeframe_minutes: int = 15,
        market_start=None,
        market_end=None,
        catalog=None,
    ) -> None:
        self.symbols = symbols
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.strategy_timeframe_minutes = strategy_timeframe_minutes
        self.market_start = market_start
        self.market_end = market_end
        self.catalog = catalog

        # Per-symbol state
        self.minute_data: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
        self.strategy_data: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
        self.tick_buffer: dict[str, list[dict]] = {s: [] for s in symbols}
        self.last_cumulative_volume: dict[str, float | None] = {s: None for s in symbols}
        self.current_minute: dict[str, datetime | None] = {s: None for s in symbols}

        # Supertrend state tracking
        self.last_trend: dict[str, int | None] = {s: None for s in symbols}
        self.last_supertrend_level: dict[str, float] = {s: 0.0 for s in symbols}

    # ─────────────────────────────────────────────────────────────────────────
    # TICK INGESTION
    # ─────────────────────────────────────────────────────────────────────────

    def process_tick(self, symbol: str, tick: dict[str, Any]) -> OHLCBar | None:
        """
        Process a single websocket tick.
        Returns a completed OHLCBar when a new minute has started, else None.
        """
        if symbol not in self.symbols:
            return None

        tick_time: datetime = tick["timestamp"]
        ltp: float = float(tick["ltp"])
        cumulative_volume: float = float(tick.get("volume", 0.0))

        # Calculate per-tick volume delta from cumulative volume
        last_vol = self.last_cumulative_volume.get(symbol)
        tick_volume = max(0.0, cumulative_volume - (last_vol or cumulative_volume))
        self.last_cumulative_volume[symbol] = cumulative_volume

        # Determine the minute bucket for this tick
        minute_bucket = tick_time.replace(second=0, microsecond=0)

        # Start a new minute
        if self.current_minute.get(symbol) != minute_bucket:
            completed_bar = self._flush_minute(symbol)
            self.current_minute[symbol] = minute_bucket
            self.tick_buffer[symbol] = []
            if completed_bar:
                self._append_bar_to_minute_data(symbol, completed_bar)
                return completed_bar

        self.tick_buffer[symbol].append({
            "timestamp": tick_time,
            "ltp": ltp,
            "volume": tick_volume,
        })
        return None

    def _flush_minute(self, symbol: str) -> OHLCBar | None:
        """Convert the tick buffer into a completed OHLC bar."""
        buf = self.tick_buffer.get(symbol, [])
        if not buf or not self.current_minute.get(symbol):
            return None

        prices = [t["ltp"] for t in buf]
        volumes = [t["volume"] for t in buf]

        return OHLCBar(
            timestamp=self.current_minute[symbol],
            symbol=symbol,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
            is_complete=True,
        )

    def _append_bar_to_minute_data(self, symbol: str, bar: OHLCBar) -> None:
        """Add a completed bar to the in-memory minute_data frame."""
        new_row = pd.DataFrame([{
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }], index=[bar.timestamp])

        existing = self.minute_data.get(symbol, pd.DataFrame())
        self.minute_data[symbol] = pd.concat([existing, new_row]).sort_index()

    # ─────────────────────────────────────────────────────────────────────────
    # SUPERTREND COMPUTATION
    # ─────────────────────────────────────────────────────────────────────────

    def compute_supertrend(self, symbol: str, current_date: date | None = None) -> SupertrendState | None:
        """
        Compute the current Supertrend state for a symbol.
        Returns None if there is insufficient or gapped data.
        """
        df = self.strategy_data.get(symbol, pd.DataFrame())
        if df.empty:
            return SupertrendState(symbol=symbol, direction=0, level=0.0, signal=0, data_quality_ok=False)

        today = current_date or date.today()
        adjusted = get_gap_adjusted_data(df, self.supertrend_period, today)

        if adjusted is None or len(adjusted) < self.supertrend_period + 2:
            LOGGER.warning("[%s] Insufficient data for Supertrend computation.", symbol)
            return SupertrendState(symbol=symbol, direction=0, level=0.0, signal=0, data_quality_ok=False)

        # calculate_supertrend will self-abort if it detects a data gap > 45 minutes
        st_df = calculate_supertrend(adjusted, period=self.supertrend_period, multiplier=self.supertrend_multiplier)

        if st_df is None or st_df.empty:
            LOGGER.error("[%s] ❌ Supertrend aborted (data gap detected). No signal will be generated.", symbol)
            return SupertrendState(symbol=symbol, direction=0, level=0.0, signal=0, data_quality_ok=False)

        valid_rows = _valid_supertrend_rows(st_df)
        if valid_rows.empty:
            return SupertrendState(symbol=symbol, direction=0, level=0.0, signal=0, data_quality_ok=False)

        last = valid_rows.iloc[-1]
        current_direction = int(last["supertrend_direction"])
        current_level = float(last["supertrend"])
        current_signal = int(last.get("supertrend_signal", 0))
        last_recorded = self.last_trend.get(symbol)

        # Detect flip
        signal = 0
        if last_recorded is not None and current_direction != last_recorded:
            signal = current_direction  # 1=flipped bullish, -1=flipped bearish
            LOGGER.info(
                "[%s] ⚡ Supertrend FLIP: %s → %s at %.2f",
                symbol, "BULL" if last_recorded == 1 else "BEAR",
                "BULL" if current_direction == 1 else "BEAR", float(last["close"])
            )

        self.last_trend[symbol] = current_direction
        self.last_supertrend_level[symbol] = current_level

        return SupertrendState(
            symbol=symbol,
            direction=current_direction,
            level=current_level,
            signal=signal,
            bar_time=valid_rows.index[-1] if hasattr(valid_rows.index, "__getitem__") else None,
            close_price=float(last["close"]),
            data_quality_ok=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DATA SEEDING
    # ─────────────────────────────────────────────────────────────────────────

    def seed_strategy_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Load historical strategy-timeframe data for warm-starting the indicator.
        Called once at market open before the websocket starts.
        """
        if df is None or df.empty:
            LOGGER.warning("[%s] No seed data provided. Supertrend will be cold-started.", symbol)
            return

        # Validate and clean
        valid_candles, errors = validate_candles(df.reset_index().to_dict("records"))
        if errors:
            LOGGER.warning("[%s] Seed data validation: %d invalid rows discarded.", symbol, len(errors))

        self.strategy_data[symbol] = df.sort_index().drop_duplicates()
        LOGGER.info("[%s] Seeded %d bars of strategy data.", symbol, len(self.strategy_data[symbol]))

    def reset_daily_state(self, symbols: list[str] | None = None) -> None:
        """Reset all intraday state for a new trading day."""
        targets = symbols or self.symbols
        for s in targets:
            self.minute_data[s] = pd.DataFrame()
            self.tick_buffer[s] = []
            self.last_cumulative_volume[s] = None
            self.current_minute[s] = None
            self.last_trend[s] = None
        LOGGER.info("MarketDataPipeline state reset for: %s", targets)
