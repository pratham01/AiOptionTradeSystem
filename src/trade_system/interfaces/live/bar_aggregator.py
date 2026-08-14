"""
BarAggregator & MultiTimeframeBarAggregator — Tick-to-OHLCV multi-timeframe bar aggregation.

Responsibilities:
- Buffer incoming ticks per symbol per minute slot.
- On minute boundary: build a complete 1-min OHLCV bar.
- On higher timeframe boundaries (3m, 5m, 15m): build resampled bars and emit events.
- Compute per-minute volume using cumulative tick volume deltas.
- Thread-safe state access for dashboard and strategy pipelines.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional

import pandas as pd

from trade_system.interfaces.live.helpers import create_minute_bar, resample_to_timeframe

LOGGER = logging.getLogger("BarAggregator")


class BarAggregator:
    """
    Stateful per-symbol tick buffer that emits completed 1-min OHLCV bars.
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
        self._lock = threading.Lock()

    def ingest_tick(self, symbol: str, tick: dict) -> None:
        with self._lock:
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
                self._flush_symbol(symbol)
                self._current_minute[symbol] = tick_minute

            self._tick_buffer[symbol].append(tick)
            self._last_tick_price[symbol] = float(tick["ltp"])

    def flush(self, symbol: str | None = None) -> None:
        with self._lock:
            targets = [symbol] if symbol else list(self._tick_buffer.keys())
            for sym in targets:
                self._flush_symbol(sym)

    @property
    def last_tick_price(self) -> dict[str, float | None]:
        with self._lock:
            return dict(self._last_tick_price)

    def reset(self, symbols: list[str]) -> None:
        with self._lock:
            for sym in symbols:
                self._tick_buffer[sym] = []
                self._current_minute[sym] = None
                self._last_cumulative_volume[sym] = None
                self._last_tick_price[sym] = None

    def _flush_symbol(self, symbol: str) -> None:
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

        last_vol = float(buffered_ticks[-1].get("volume", 0.0) or 0.0)
        if last_vol > 0:
            self._last_cumulative_volume[symbol] = last_vol

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

        try:
            self._on_bar_complete(symbol, bar, self._minute_data)
        except Exception as exc:
            LOGGER.error("on_bar_complete callback raised for %s: %s", symbol, exc)


class MultiTimeframeBarAggregator:
    """
    Advanced Multi-Timeframe Bar Aggregator.
    Builds 1m, 3m, 5m, 15m, and Daily candles simultaneously from live tick stream.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframes: Optional[List[int]] = None,
        on_timeframe_bar: Optional[Callable[[str, int, pd.Series, pd.DataFrame], None]] = None,
    ) -> None:
        self.symbols = symbols
        self.timeframes = timeframes or [1, 3, 5, 15]
        self._on_timeframe_bar = on_timeframe_bar

        self._tf_data: Dict[str, Dict[int, pd.DataFrame]] = {
            s: {tf: pd.DataFrame() for tf in self.timeframes} for s in symbols
        }
        self._minute_data: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
        self._base_aggregator = BarAggregator(
            symbols=symbols,
            on_bar_complete=self._on_1m_bar_complete,
            minute_data_ref=self._minute_data,
        )

    def ingest_tick(self, symbol: str, tick: dict) -> None:
        """Ingest live tick into base 1m aggregator."""
        self._base_aggregator.ingest_tick(symbol, tick)

    def flush(self, symbol: Optional[str] = None) -> None:
        """Force flush all buffered ticks."""
        self._base_aggregator.flush(symbol)

    def get_timeframe_data(self, symbol: str, timeframe_minutes: int) -> pd.DataFrame:
        """Get the cached rolling dataframe for a specific timeframe."""
        sym_dict = self._tf_data.get(symbol, {})
        return sym_dict.get(timeframe_minutes, pd.DataFrame())

    @property
    def last_tick_price(self) -> Dict[str, Optional[float]]:
        return self._base_aggregator.last_tick_price

    def _on_1m_bar_complete(self, symbol: str, bar_1m: pd.Series, minute_data: Dict[str, pd.DataFrame]) -> None:
        """Fires whenever a 1-min bar finishes, computing HTF resamplings."""
        df_1m = minute_data.get(symbol, pd.DataFrame())
        if df_1m.empty:
            return

        # 1. Dispatch 1m bar
        if 1 in self.timeframes:
            self._tf_data.setdefault(symbol, {})[1] = df_1m
            if self._on_timeframe_bar:
                try:
                    self._on_timeframe_bar(symbol, 1, bar_1m, df_1m)
                except Exception as exc:
                    LOGGER.error("Callback error on 1m bar for %s: %s", symbol, exc)

        # 2. Check higher timeframes (3m, 5m, 15m)
        bar_time: datetime = bar_1m.name if isinstance(bar_1m.name, datetime) else pd.to_datetime(bar_1m.name).to_pydatetime()
        minute_val = bar_time.minute

        for tf in [t for t in self.timeframes if t > 1]:
            # Trigger resampling if bar completes the TF boundary (e.g. minute 14 finishes the 0-14 15m candle)
            if (minute_val + 1) % tf == 0 or len(df_1m) % tf == 0:
                try:
                    resampled = resample_to_timeframe(df_1m, tf)
                    if not resampled.empty:
                        self._tf_data.setdefault(symbol, {})[tf] = resampled
                        completed_htf_bar = resampled.iloc[-1]
                        if self._on_timeframe_bar:
                            self._on_timeframe_bar(symbol, tf, completed_htf_bar, resampled)
                except Exception as exc:
                    LOGGER.error("Failed to resample %dm for %s: %s", tf, symbol, exc)
