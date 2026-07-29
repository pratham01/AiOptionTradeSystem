"""
WebSocketAdapter — Fyers WebSocket lifecycle management.

Responsibilities
----------------
* Own the Fyers DataSocket connection: connect, subscribe, keep_running.
* Detect fatal token errors (-99 / -300 codes or "token" keyword) and request
  a refresh via broker._ws_token_cache invalidation + force_refresh flag.
* Expose a clean on_tick(symbol, tick) callback interface so that the upstream
  orchestrator (LiveMarketDataService) is completely decoupled from Fyers SDK.
* Track ws_running state so the main run_forever() loop can restart after error.

Design
------
The adapter is intentionally thin: it owns *only* the connection lifecycle and
routes raw messages to the caller-supplied ``on_tick`` callback.  It does NOT
process ticks, build bars, or know anything about indicators.
"""
from __future__ import annotations

import logging
import time
import threading
from collections.abc import Callable
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger(__name__)

# Fyers WebSocket fatal token error codes (string representation)
_WS_TOKEN_ERROR_CODES = frozenset(["-99", "-300"])


class WebSocketAdapter:
    """
    Wraps the Fyers DataSocket with automatic reconnect and token-error handling.

    Parameters
    ----------
    broker : FyersBroker
        Must expose ``.websocket_access_token()`` and ``._ws_token_cache``.
    settings : Settings
        Used for market_start / market_end clocks.
    on_tick : Callable[[str, dict], None]
        Called for every valid market-hours tick.
        Signature: ``on_tick(symbol, tick_dict)``
    on_connected : Callable[[], None] | None
        Optional hook called once per successful WebSocket connection.
    market_start : dt_time
        Market open time — ticks outside this window are dropped.
    market_end : dt_time
        Market close time.
    """

    def __init__(
        self,
        broker,
        on_tick: Callable[[str, dict], None],
        symbols: list[str],
        market_start: dt_time,
        market_end: dt_time,
        on_connected: Callable[[], None] | None = None,
    ) -> None:
        self._broker = broker
        self._on_tick = on_tick
        self._symbols = symbols
        self._market_start = market_start
        self._market_end = market_end
        self._on_connected = on_connected

        self._ws = None
        self._running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Connect the WebSocket and begin receiving ticks (blocking thread)."""
        from fyers_apiv3.FyersWebsocket import data_ws  # type: ignore[import]

        with self._lock:
            if self._running:
                LOGGER.warning("WebSocketAdapter.start() called while already running.")
                return

            self._ws = data_ws.FyersDataSocket(
                access_token=self._broker.websocket_access_token(),
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=self._handle_open,
                on_close=self._handle_close,
                on_error=self._handle_error,
                on_message=self._handle_message,
            )
            self._running = True

        # Run blocking connect in a daemon thread managed by the orchestrator
        self._ws.connect()

    def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception as exc:
                LOGGER.warning("Error closing WebSocket: %s", exc)
        self._ws = None

    def subscribe(self, symbols: list[str] | None = None) -> None:
        """Subscribe to SymbolUpdate feed for the given (or initial) symbol list."""
        syms = symbols or self._symbols
        if self._ws:
            self._ws.subscribe(symbols=syms, data_type="SymbolUpdate")
            LOGGER.info("WebSocket subscribed to %d symbols.", len(syms))

    # ------------------------------------------------------------------
    # Fyers SDK callbacks
    # ------------------------------------------------------------------

    def _handle_open(self) -> None:
        LOGGER.info("WebSocket connected. Subscribing to %d symbols.", len(self._symbols))
        self.subscribe()
        self._ws.keep_running()
        if self._on_connected:
            try:
                self._on_connected()
            except Exception as exc:
                LOGGER.error("on_connected callback raised: %s", exc)

    def _handle_close(self, message) -> None:
        LOGGER.warning("WebSocket closed: %s", message)
        self._running = False

    def _handle_error(self, message) -> None:
        LOGGER.error("WebSocket error: %s", message)
        msg_str = str(message).lower()
        # Detect fatal token errors
        is_token_error = (
            "token" in msg_str
            or any(code in msg_str for code in _WS_TOKEN_ERROR_CODES)
        )
        if is_token_error:
            LOGGER.warning(
                "WebSocket token error detected. Invalidating cached WS token "
                "and signalling for refresh on next reconnect."
            )
            self._broker._ws_token_cache = None  # noqa: SLF001 — intentional cache bust
            if hasattr(self._broker, "force_refresh"):
                self._broker.force_refresh = True
            # Signal run_forever() loop to restart the connection
            self._running = False

    def _handle_message(self, message) -> None:
        """Route incoming tick messages to the on_tick callback."""
        payload = message if isinstance(message, list) else [message]
        for item in payload:
            symbol = item.get("symbol")
            if symbol not in self._symbols or item.get("ltp") is None:
                continue

            # Parse epoch timestamp → IST datetime
            tick_time = self._from_epoch_ist(item.get("timestamp", time.time()))

            # Drop ticks outside market hours
            if not self._is_market_timestamp(tick_time):
                continue

            cumulative_volume = (
                item.get("volume")
                or item.get("vol_traded_today")
                or item.get("vol_traded")
                or item.get("v")
                or 0.0
            )
            tick = {
                "timestamp": tick_time,
                "ltp": float(item["ltp"]),
                "volume": float(cumulative_volume),
            }
            try:
                self._on_tick(symbol, tick)
            except Exception as exc:
                LOGGER.error("on_tick callback raised for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _from_epoch_ist(epoch_seconds: float) -> datetime:
        return datetime.fromtimestamp(epoch_seconds, tz=IST).replace(microsecond=0, tzinfo=None)

    def _is_market_timestamp(self, ts: datetime) -> bool:
        t = ts.time()
        return self._market_start <= t < self._market_end
