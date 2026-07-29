"""
OrbPipeline — Opening Range Breakout (ORB) analytics.

Triggered
---------
``on_bar(symbol, bar, minute_data)``  — called on every completed 1-min bar.
``on_tick(symbol, price, tick_time)`` — called on every live tick for
    real-time breakout detection.

Responsibilities
----------------
1. Lock in the first 15-min candle (9:15–9:29) as the Opening Range once ≥14
   of 15 bars are available.
2. Broadcast an Opening Range reference card to Telegram (both channels).
3. Detect price crossing above ORB High → BUY CALL alert.
4. Detect price crossing below ORB Low  → BUY PUT alert.
5. Fire each directional alert only once per trading day (deduplication via set).

Index-only
----------
ORB is only relevant for index instruments.  FoPipeline does NOT call this.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dt_time

import pandas as pd

LOGGER = logging.getLogger(__name__)

_ORB_START = dt_time(9, 15)
_ORB_END = dt_time(9, 30)
_MIN_BARS_REQUIRED = 14   # ≥14 of 15 minute bars needed to lock the range


class OrbPipeline:
    """
    Per-symbol Opening Range Breakout tracker.

    Parameters
    ----------
    notifier : TelegramNotifier
        Main alert channel.
    confirmed_notifier : TelegramNotifier
        ST-confirmed (detailed) channel.
    """

    def __init__(self, notifier, confirmed_notifier) -> None:
        self.notifier = notifier
        self.confirmed_notifier = confirmed_notifier

        # Per-symbol state (reset each day via reset())
        self._candle: dict[str, dict | None] = {}       # locked ORB candle
        self._break_sent: dict[str, set[str]] = {}      # {"HIGH", "LOW"} once sent

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_symbol(self, symbol: str) -> None:
        """Pre-allocate state for a symbol."""
        if symbol not in self._candle:
            self._candle[symbol] = None
            self._break_sent[symbol] = set()

    def reset(self, symbols: list[str]) -> None:
        """Reset intraday state (call at each day boundary)."""
        for sym in symbols:
            self._candle[sym] = None
            self._break_sent[sym] = set()

    def get_candle(self, symbol: str) -> dict | None:
        """Return the locked ORB candle for a symbol (None if not yet locked)."""
        return self._candle.get(symbol)

    # ------------------------------------------------------------------
    # Bar-level hook (called once per completed 1-min bar)
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: pd.Series,
        minute_data: dict[str, pd.DataFrame],
        current_date: date,
    ) -> None:
        """Check whether the ORB range can be locked on this bar."""
        self.register_symbol(symbol)
        if self._candle[symbol] is not None:
            return  # Already locked today

        df = minute_data.get(symbol, pd.DataFrame())
        if df.empty:
            return

        today = df[df.index.date == current_date]
        first_15 = today[
            (today.index.time >= _ORB_START) & (today.index.time < _ORB_END)
        ]
        if len(first_15) < _MIN_BARS_REQUIRED:
            return

        candle = {
            "high": float(first_15["high"].max()),
            "low":  float(first_15["low"].min()),
            "open": float(first_15["open"].iloc[0]),
            "close": float(first_15["close"].iloc[-1]),
        }
        self._candle[symbol] = candle
        short_sym = self._short(symbol)
        LOGGER.info(
            "ORB locked for %s — High=%.2f Low=%.2f",
            symbol, candle["high"], candle["low"],
        )

        ref_msg = (
            f"📐 <b>Opening Range Set — {short_sym}</b>\n"
            f"<i>First 15-min candle (9:15 – 9:29)</i>\n"
            f"Open:  ₹{candle['open']:.2f}\n"
            f"🔺 High: ₹{candle['high']:.2f}\n"
            f"🔻 Low:  ₹{candle['low']:.2f}\n"
            f"Range: ₹{candle['high'] - candle['low']:.2f} pts\n"
            f"<i>Alerts will fire on breakout above ₹{candle['high']:.2f} "
            f"or breakdown below ₹{candle['low']:.2f}</i>"
        )
        self.notifier.send(ref_msg)
        self.confirmed_notifier.send(ref_msg)

    # ------------------------------------------------------------------
    # Tick-level hook (called on every live tick)
    # ------------------------------------------------------------------

    def on_tick(self, symbol: str, price: float, tick_time: datetime) -> None:
        """Fire directional breakout alert when price crosses ORB boundaries."""
        self.register_symbol(symbol)
        candle = self._candle.get(symbol)
        if not candle:
            return

        sent = self._break_sent[symbol]
        time_str = tick_time.strftime("%H:%M") if isinstance(tick_time, datetime) else str(tick_time)
        short_sym = self._short(symbol)

        if "HIGH" not in sent and price > candle["high"]:
            sent.add("HIGH")
            gap = price - candle["high"]
            msg = (
                f"🟢 <b>ORB Breakout — {short_sym} {time_str}</b>\n"
                f"Price broke <b>ABOVE</b> first 15-min candle high\n\n"
                f"First 15m High : ₹{candle['high']:.2f}\n"
                f"Current Price  : ₹{price:.2f}  (+{gap:.2f} pts)\n"
                f"First 15m Low  : ₹{candle['low']:.2f}\n\n"
                f"Action: <b>BUY CALL / LONG</b>\n"
                f"SL Ref: Below ₹{candle['high']:.2f} (ORB High)"
            )
            self.notifier.send(msg)
            self.confirmed_notifier.send(msg)
            LOGGER.info("ORB HIGH breakout for %s at %.2f (ORB High=%.2f)", symbol, price, candle["high"])

        if "LOW" not in sent and price < candle["low"]:
            sent.add("LOW")
            gap = candle["low"] - price
            msg = (
                f"🔴 <b>ORB Breakdown — {short_sym} {time_str}</b>\n"
                f"Price broke <b>BELOW</b> first 15-min candle low\n\n"
                f"First 15m Low  : ₹{candle['low']:.2f}\n"
                f"Current Price  : ₹{price:.2f}  (-{gap:.2f} pts)\n"
                f"First 15m High : ₹{candle['high']:.2f}\n\n"
                f"Action: <b>BUY PUT / SHORT</b>\n"
                f"SL Ref: Above ₹{candle['low']:.2f} (ORB Low)"
            )
            self.notifier.send(msg)
            self.confirmed_notifier.send(msg)
            LOGGER.info("ORB LOW breakdown for %s at %.2f (ORB Low=%.2f)", symbol, price, candle["low"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _short(symbol: str) -> str:
        return symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")
