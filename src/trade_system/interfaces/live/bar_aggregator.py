"""
BarAggregator — Tick-to-OHLCV minute bar aggregation.

Responsibilities
----------------
* Buffer incoming ticks per symbol per minute slot.
* On minute boundary: build a complete 1-min OHLCV bar and emit it via the
  ``on_bar_complete`` callback.
* Compute per-minute volume using cumulative tick volume deltas.
* Expose ``flush(symbol)`` for end-of-session cleanup.

Design
------
The aggregator is stateful but self-contained.  It holds no references to the
broker, DB, or Telegram.  It only knows about ticks and bars, making it trivially
unit-testable with synthetic tick data.

Callback signature
------------------
    on_bar_complete(symbol: str, bar: pd.Series, minute_data: dict[str, pd.DataFrame]) -> None

The callback receives the *completed* bar plus the caller's minute_data dict
(passed in so the pipeline can immediately update its rolling window).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

import pandas as pd

from trade_system.interfaces.live.helpers import create_minute_bar

LOGGER = logging.getLogger(__name__)


class BarAggregator:
    """
    Stateful per-symbol tick buffer that emits completed 1-min OHLCV bars.

    Parameters
    ----------
    symbols : list[str]
        The symbol universe to track.  State dicts are pre-allocated.
    on_bar_complete : Callable
        Called when a minute bar is finalized.
        Signature: ``on_bar_complete(symbol, bar, minute_data_ref)``
    minute_data_ref : dict[str, pd.DataFrame]
        Shared mutable reference to the rolling minute-data store.  The
        aggregator appends each completed bar here before calling the callback.
    """

    def __init__(
        self,
        symbols: list[str],
        on_bar_complete: Callable[[str, pd.Series, dict[str, pd.DataFrame]], None],
        minute_data_ref: dict[str, pd.DataFrame],
    ) -> None:
        self._on_bar_complete = on_bar_complete
        self._minute_data = minute_data_ref

        # Per-symbol state
        self._tick_buffer: dict[str, list[dict]] = {s: [] for s in symbols}
        self._current_minute: dict[str, datetime | None] = {s: None for s in symbols}
        self._last_cumulative_volume: dict[str, float | None] = {s: None for s in symbols}
        self._last_tick_price: dict[str, float | None] = {s: None for s in symbols}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest_tick(self, symbol: str, tick: dict) -> None:
        """
        Process a single tick for ``symbol``.

        Flushes the previous minute's buffer if the tick has crossed a
        minute boundary, then buffers this tick.

        Parameters
        ----------
        symbol : str
        tick : dict
            Must contain ``timestamp`` (datetime), ``ltp`` (float), ``volume`` (float).
        """
        if symbol not in self._tick_buffer:
            self._tick_buffer[symbol] = []
            self._current_minute[symbol] = None
            self._last_cumulative_volume[symbol] = None
            self._last_tick_price[symbol] = None

        tick_minute = tick["timestamp"].replace(second=0, microsecond=0)
        current_minute = self._current_minute[symbol]

        if current_minute is None:
            self._current_minute[symbol] = tick_minute
        elif tick_minute > current_minute:
            # Minute boundary crossed — complete the previous bar
            self._flush_symbol(symbol)
            self._current_minute[symbol] = tick_minute

        self._tick_buffer[symbol].append(tick)
        self._last_tick_price[symbol] = float(tick["ltp"])

    def flush(self, symbol: str | None = None) -> None:
        """
        Force-flush open minute buffers.  Call at market close or WebSocket stop.

        Parameters
        ----------
        symbol : str | None
            If None, flushes all symbols.
        """
        targets = [symbol] if symbol else list(self._tick_buffer.keys())
        for sym in targets:
            self._flush_symbol(sym)

    @property
    def last_tick_price(self) -> dict[str, float | None]:
        """Read-only access to last known tick prices (for dashboard updates)."""
        return self._last_tick_price

    def reset(self, symbols: list[str]) -> None:
        """Reset all state (called at day boundary / _reset_intraday_state)."""
        for sym in symbols:
            self._tick_buffer[sym] = []
            self._current_minute[sym] = None
            self._last_cumulative_volume[sym] = None
            self._last_tick_price[sym] = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_symbol(self, symbol: str) -> None:
        """Build a completed minute bar from the buffer and emit via callback."""
        buffered_ticks = self._tick_buffer.get(symbol, [])
        if not buffered_ticks:
            return

        bar = create_minute_bar(
            symbol,
            buffered_ticks,
            self._last_cumulative_volume.get(symbol),
        )
        self._tick_buffer[symbol] = []

        if bar is None:
            return

        # Update cumulative volume tracker from last tick
        last_vol = float(buffered_ticks[-1].get("volume", 0.0) or 0.0)
        if last_vol > 0:
            self._last_cumulative_volume[symbol] = last_vol

        # Update the shared minute_data rolling window
        try:
            minute_df = (
                bar.to_frame().T
                .reset_index()
                .rename(columns={"index": "timestamp"})
                .set_index("timestamp")
            )
            combined = pd.concat(
                [self._minute_data.get(symbol, pd.DataFrame()), minute_df]
            ).sort_index()
            self._minute_data[symbol] = combined[
                ~combined.index.duplicated(keep="last")
            ].tail(1500)
        except Exception as exc:
            LOGGER.error("Failed to update minute_data for %s: %s", symbol, exc)

        # Emit to the pipeline
        try:
            self._on_bar_complete(symbol, bar, self._minute_data)
        except Exception as exc:
            LOGGER.error("on_bar_complete callback raised for %s: %s", symbol, exc)
