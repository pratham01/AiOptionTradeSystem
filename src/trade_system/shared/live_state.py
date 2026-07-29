"""
LiveState — Shared live state file for communication between the live bot and the dashboard.

The live bot (collector) writes to this file atomically on every tick/flush.
The dashboard reads from this file instead of making redundant broker API calls.

File format: JSON at `data/live_state.json`
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Default path — can be overridden via TRADE_SYSTEM_DATA_DIR env var
_DATA_DIR = Path(os.getenv("TRADE_SYSTEM_DATA_DIR", "data")).resolve()
LIVE_STATE_PATH = _DATA_DIR / "live_state.json"

# How old (seconds) the state file can be before we consider it stale
STALE_THRESHOLD_SECONDS = 30


@dataclass
class SymbolState:
    """Current live state for a single symbol."""
    symbol: str
    ltp: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    previous_close: float = 0.0
    volume: int = 0
    change: float = 0.0
    change_percent: float = 0.0
    supertrend_direction: int = 0  # 1=bullish, -1=bearish, 0=unknown
    supertrend_level: float = 0.0
    last_tick_time: str = ""


@dataclass
class SystemHealth:
    """Health metrics for the live bot."""
    ws_connected: bool = False
    db_queue_depth: int = 0
    db_total_written: int = 0
    db_errors: int = 0
    db_avg_write_ms: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class LiveState:
    """Full live state snapshot."""
    timestamp: str = ""
    bot_running: bool = False
    symbols: dict[str, dict[str, Any]] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)


class LiveStateWriter:
    """
    Thread-safe writer for the live state file.

    Usage:
        writer = LiveStateWriter()
        writer.update_tick("NSE:NIFTY50-INDEX", ltp=24500.0, volume=1234567)
        writer.update_supertrend("NSE:NIFTY50-INDEX", direction=1, level=24300.0)
        writer.flush()  # Atomically writes to disk
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self._path = state_path or LIVE_STATE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._symbols: dict[str, SymbolState] = {}
        self._health = SystemHealth()
        self._bot_running = False
        self._start_time = time.monotonic()
        self._last_flush_time = 0.0
        # Minimum interval between disk writes (seconds) to avoid I/O thrashing
        self._min_flush_interval = 0.5

    def set_bot_running(self, running: bool) -> None:
        with self._lock:
            self._bot_running = running
            self._health.ws_connected = running

    def update_tick(self, symbol: str, ltp: float, volume: float = 0,
                    tick_time: datetime | None = None) -> None:
        """Update the latest tick price for a symbol. Called on every websocket tick."""
        with self._lock:
            state = self._symbols.get(symbol)
            if state is None:
                state = SymbolState(symbol=symbol)
                self._symbols[symbol] = state

            state.ltp = ltp
            if volume > 0:
                state.volume = int(volume)

            # Track intraday high/low
            if state.high == 0.0 or ltp > state.high:
                state.high = ltp
            if state.low == 0.0 or ltp < state.low:
                state.low = ltp

            state.close = ltp
            if tick_time:
                state.last_tick_time = tick_time.isoformat()
            else:
                state.last_tick_time = datetime.now().isoformat()

            # Compute change if we have previous_close
            if state.previous_close > 0:
                state.change = ltp - state.previous_close
                state.change_percent = (state.change / state.previous_close) * 100.0

    def update_minute_bar(self, symbol: str, open_: float, high: float,
                          low: float, close: float, volume: int) -> None:
        """Update from a completed minute bar."""
        with self._lock:
            state = self._symbols.get(symbol)
            if state is None:
                state = SymbolState(symbol=symbol)
                self._symbols[symbol] = state

            state.ltp = close
            state.close = close
            state.volume = volume
            # Only update open on the first bar of the day
            if state.open == 0.0:
                state.open = open_

            # Update intraday high/low
            if state.high == 0.0 or high > state.high:
                state.high = high
            if state.low == 0.0 or low < state.low:
                state.low = low

    def update_previous_close(self, symbol: str, prev_close: float) -> None:
        """Set the previous close for change calculations."""
        with self._lock:
            state = self._symbols.get(symbol)
            if state is None:
                state = SymbolState(symbol=symbol)
                self._symbols[symbol] = state
            state.previous_close = prev_close

    def update_supertrend(self, symbol: str, direction: int, level: float) -> None:
        """Update supertrend state for a symbol."""
        with self._lock:
            state = self._symbols.get(symbol)
            if state is None:
                state = SymbolState(symbol=symbol)
                self._symbols[symbol] = state
            state.supertrend_direction = direction
            state.supertrend_level = level

    def update_health(self, db_stats: dict[str, Any] | None = None,
                      ws_connected: bool | None = None) -> None:
        """Update system health metrics."""
        with self._lock:
            if db_stats:
                self._health.db_queue_depth = db_stats.get("queue_depth", 0)
                self._health.db_total_written = db_stats.get("total_written", 0)
                self._health.db_errors = db_stats.get("total_errors", 0)
                self._health.db_avg_write_ms = db_stats.get("avg_write_ms", 0.0)
            if ws_connected is not None:
                self._health.ws_connected = ws_connected
            self._health.uptime_seconds = time.monotonic() - self._start_time

    def flush(self, force: bool = False) -> None:
        """
        Atomically write the current state to disk.

        Uses write-to-temp-then-rename for crash safety.
        Throttled to avoid excessive I/O (min 500ms between flushes unless forced).
        """
        now = time.monotonic()
        if not force and (now - self._last_flush_time) < self._min_flush_interval:
            return

        with self._lock:
            state = LiveState(
                timestamp=datetime.now().isoformat(),
                bot_running=self._bot_running,
                symbols={sym: asdict(s) for sym, s in self._symbols.items()},
                health=asdict(self._health),
            )

        try:
            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(asdict(state), f)
                os.replace(tmp_path, str(self._path))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            self._last_flush_time = now
        except Exception as e:
            LOGGER.error("Failed to write live state: %s", e)

    def reset_day(self) -> None:
        """Reset all symbol states for a new trading day."""
        with self._lock:
            for state in self._symbols.values():
                state.ltp = 0.0
                state.open = 0.0
                state.high = 0.0
                state.low = 0.0
                state.close = 0.0
                state.change = 0.0
                state.change_percent = 0.0
                state.volume = 0
                state.supertrend_direction = 0
                state.supertrend_level = 0.0


class LiveStateReader:
    """
    Reader for the live state file.

    Usage:
        reader = LiveStateReader()
        if reader.is_bot_running():
            prices = reader.get_current_prices(["NSE:NIFTY50-INDEX"])
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self._path = state_path or LIVE_STATE_PATH
        self._cache: dict[str, Any] | None = None
        self._cache_mtime: float = 0.0

    def _read(self) -> dict[str, Any] | None:
        """Read the state file, with mtime-based caching to avoid redundant reads."""
        try:
            if not self._path.exists():
                return None

            mtime = self._path.stat().st_mtime
            if self._cache is not None and mtime == self._cache_mtime:
                return self._cache

            with open(self._path, "r") as f:
                data = json.load(f)

            self._cache = data
            self._cache_mtime = mtime
            return data
        except (json.JSONDecodeError, OSError) as e:
            LOGGER.warning("Failed to read live state: %s", e)
            return None

    def is_fresh(self, max_age_seconds: float = STALE_THRESHOLD_SECONDS) -> bool:
        """Check if the state file exists and is fresh (not stale)."""
        data = self._read()
        if not data:
            return False
        try:
            ts = datetime.fromisoformat(data["timestamp"])
            age = (datetime.now() - ts).total_seconds()
            return age < max_age_seconds
        except (KeyError, ValueError):
            return False

    def is_bot_running(self) -> bool:
        """Check if the live bot reports itself as running."""
        data = self._read()
        if not data:
            return False
        if not data.get("bot_running", False):
            return False
        return self.is_fresh()

    def get_current_prices(self, symbols: list[str] | None = None) -> dict[str, SymbolState]:
        """Get current prices for all or specified symbols."""
        data = self._read()
        if not data:
            return {}

        result: dict[str, SymbolState] = {}
        sym_data = data.get("symbols", {})

        for sym, s_dict in sym_data.items():
            if symbols and sym not in symbols:
                continue
            result[sym] = SymbolState(**s_dict)

        return result

    def get_system_health(self) -> dict[str, Any]:
        """Get system health metrics."""
        data = self._read()
        if not data:
            return {}
        return data.get("health", {})

    def get_raw_state(self) -> dict[str, Any] | None:
        """Get the full raw state dictionary."""
        return self._read()
