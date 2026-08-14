"""
SrPipeline — Support/Resistance channel and zone proximity analytics.

Triggered
---------
``on_bar(symbol, bar, minute_data)`` — 15-min S/R channel touch detection.
``on_tick(symbol, price, tick_time)`` — Zone proximity alert (daily zones).

Responsibilities
----------------
1. On each completed 1-min bar: resample to 15-min, detect S/R channels
   using SupportResistanceChannelDetector, fire a Telegram alert when price
   touches a channel (60-min cooldown per channel).
2. On each tick: check whether price is within ``zone_buffer_pct`` of a
   pre-loaded daily zone level and fire a zone proximity alert.
3. Gap and previous-day high/low/close level touch detection.

Index-only
----------
S/R channel and daily-zone logic is computed only for index instruments.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from typing import Any

import pandas as pd

from trade_system.interfaces.live.helpers import resample_to_timeframe
from trade_system.domains.strategy.application.indicators.support_resistance_channels import (
    SupportResistanceChannelDetector,
)

LOGGER = logging.getLogger(__name__)

_CHANNEL_COOLDOWN_SECS = 3600      # 60 min between repeat alerts for same channel
_CHANNEL_TOUCH_BUFFER = 0.001      # 0.1% buffer around channel bounds
_ZONE_BUFFER_PCT = 0.002           # 0.2% proximity to daily zone level
_ZONE_COOLDOWN_SECS = 1800         # 30 min cooldown per zone
_MIN_15M_BARS = 25                 # minimum bars for S/R calculation


class SrPipeline:
    """
    Support/Resistance channel and daily-zone proximity pipeline.

    Parameters
    ----------
    notifier : TelegramNotifier
    alert_agent : LiveAlertAgent   — used for PrevDay level retests.
    alert_debounce_seconds : int   — level-touch debounce (default 300).
    """

    def __init__(
        self,
        notifier,
        alert_agent,
        alert_debounce_seconds: int = 300,
    ) -> None:
        self.notifier = notifier
        self.alert_agent = alert_agent
        self.debounce_secs = alert_debounce_seconds

        self.sr_detector = SupportResistanceChannelDetector()

        # Per-symbol state
        self._daily_zones: dict[str, dict[str, float]] = {}
        self._last_sr_alert_time: dict[str, dict[str, datetime]] = {}
        self._last_zone_alert_time: dict[str, dict[str, datetime]] = {}
        self._previous_day_levels: dict[str, dict | None] = {}
        self._gap_alert_sent: dict[str, bool] = {}
        self._first_live_price: dict[str, float | None] = {}
        self._last_tick_price: dict[str, float | None] = {}
        self._last_level_alert_time: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_symbol(self, symbol: str) -> None:
        if symbol not in self._daily_zones:
            self._daily_zones[symbol] = {}
            self._last_sr_alert_time[symbol] = {}
            self._last_zone_alert_time[symbol] = {}
            self._previous_day_levels[symbol] = None
            self._gap_alert_sent[symbol] = False
            self._first_live_price[symbol] = None
            self._last_tick_price[symbol] = None
            self._last_level_alert_time[symbol] = {}

    def reset(self, symbols: list[str]) -> None:
        """Reset intraday state at day boundary."""
        for sym in symbols:
            self._daily_zones[sym] = {}
            self._last_sr_alert_time[sym] = {}
            self._last_zone_alert_time[sym] = {}
            self._gap_alert_sent[sym] = False
            self._first_live_price[sym] = None
            self._last_tick_price[sym] = None
            self._last_level_alert_time[sym] = {}
            # Keep _previous_day_levels — set externally at day-init

    def set_previous_day_levels(self, symbol: str, levels: dict | None) -> None:
        """Inject previous-day OHLC levels from the orchestrator's day-init."""
        self._previous_day_levels[symbol] = levels

    def set_daily_zones(self, symbol: str, zones: dict[str, float]) -> None:
        """Inject pre-calculated daily S/R zones from the orchestrator."""
        self._daily_zones[symbol] = zones

    # ------------------------------------------------------------------
    # Bar-level hook — 15-min S/R channel touch
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: pd.Series,
        minute_data: dict[str, pd.DataFrame],
    ) -> None:
        """Detect 15-min S/R channel touch on each completed 1-min bar."""
        self.register_symbol(symbol)
        df = minute_data.get(symbol)
        if df is None or df.empty:
            return

        try:
            df_15m = resample_to_timeframe(df, 15).reset_index()
            if len(df_15m) < _MIN_15M_BARS:
                return

            snapshots = self.sr_detector.calculate(df_15m)
            if not snapshots:
                return

            latest_snap = snapshots[-1]
            channels = latest_snap.channels
            if not channels:
                return

            current_price = float(bar.get("close", 0.0))
            if current_price == 0.0:
                return

            now = self._now()
            short_sym = self._short(symbol)

            for ch in channels:
                buf = current_price * _CHANNEL_TOUCH_BUFFER
                if (ch.low - buf) <= current_price <= (ch.high + buf):
                    channel_id = f"{ch.channel_type}_{ch.low:.1f}_{ch.high:.1f}"
                    last_time = self._last_sr_alert_time[symbol].get(channel_id)
                    if last_time is None or (now - last_time).total_seconds() > _CHANNEL_COOLDOWN_SECS:
                        self._last_sr_alert_time[symbol][channel_id] = now
                        ch_type_str = ch.channel_type.capitalize()
                        msg = (
                            f"🔔 <b>{short_sym} — 15m SR {ch_type_str} Touch</b>\n\n"
                            f"Price: ₹{current_price:.2f}\n"
                            f"Channel: ₹{ch.low:.2f} – ₹{ch.high:.2f}\n"
                            f"Time: {now.strftime('%H:%M:%S')}"
                        )
                        self.notifier.send(msg)
        except Exception as exc:
            LOGGER.error("SrPipeline.on_bar failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Tick-level hooks — gap alert + daily zone proximity
    # ------------------------------------------------------------------

    def on_first_tick(
        self,
        symbol: str,
        price: float,
        tick_time: pd.Timestamp,
        major_gap_threshold_pct: float,
    ) -> None:
        """
        Called on the very first live tick of the day for this symbol.
        Detects major gap-up / gap-down vs previous close.
        """
        self.register_symbol(symbol)
        if self._first_live_price[symbol] is not None:
            return  # Already fired today

        self._first_live_price[symbol] = price
        prev = self._previous_day_levels.get(symbol)
        if not prev:
            return

        prev_close = float(prev["close"])
        if prev_close <= 0:
            return

        gap_points = price - prev_close
        gap_pct = (gap_points / prev_close) * 100.0
        if abs(gap_pct) < major_gap_threshold_pct:
            return

        if self._gap_alert_sent[symbol]:
            return
        self._gap_alert_sent[symbol] = True

        direction = "Gap Up" if gap_points > 0 else "Gap Down"
        color = "🟢" if gap_points > 0 else "🔴"
        short_sym = self._short(symbol)
        self.notifier.send(
            f"{color} <b>{short_sym} {tick_time.strftime('%H:%M')}</b>\n"
            f"Major {direction}\n"
            f"Open/first live price: ₹{price:.2f}\n"
            f"Previous close: ₹{prev_close:.2f}\n"
            f"Gap: {gap_points:+.2f} pts ({gap_pct:+.2f}%)"
        )

    def on_tick(
        self,
        symbol: str,
        price: float,
        tick_time: datetime,
    ) -> None:
        """
        Called on every live tick.
        1. Previous-day level touch → log via alert_agent (no Telegram).
        2. Daily zone proximity → Telegram alert (with cooldown).
        """
        self.register_symbol(symbol)
        previous_price = self._last_tick_price[symbol]
        self._last_tick_price[symbol] = price

        # Previous-day OHLC level retest (log only, no Telegram)
        prev = self._previous_day_levels.get(symbol)
        if prev:
            level_map = {
                "high":  float(prev["high"]),
                "low":   float(prev["low"]),
                "close": float(prev["close"]),
            }
            for level_name, level_value in level_map.items():
                last_alert = self._last_level_alert_time[symbol].get(level_name)
                if last_alert:
                    ts = tick_time if isinstance(tick_time, datetime) else tick_time.to_pydatetime()
                    if (ts.replace(tzinfo=None) - last_alert.replace(tzinfo=None)).total_seconds() < self.debounce_secs:
                        continue
                if self._price_touched_level(previous_price, price, level_value):
                    self.alert_agent.alert_retest(
                        symbol, f"PrevDay_{level_name.upper()}", level_value, log_only=True
                    )

        # Daily zone proximity alert
        zones = self._daily_zones.get(symbol, {})
        if not zones:
            return

        now_dt = tick_time if isinstance(tick_time, datetime) else tick_time.to_pydatetime()
        short_sym = self._short(symbol)
        for zone_name, zone_price in zones.items():
            if zone_price <= 0:
                continue
            proximity = abs(price - zone_price) / zone_price
            if proximity > _ZONE_BUFFER_PCT:
                continue
            last = self._last_zone_alert_time[symbol].get(zone_name)
            if last and (now_dt.replace(tzinfo=None) - last.replace(tzinfo=None)).total_seconds() < _ZONE_COOLDOWN_SECS:
                continue
            self._last_zone_alert_time[symbol][zone_name] = now_dt
            direction = "above" if price > zone_price else "at/below"
            # Log only — Telegram proximity alerts disabled per user preference
            LOGGER.info(
                "[%s] Zone proximity: %s @ ₹%.2f | Price: ₹%.2f (%s)",
                short_sym, zone_name, zone_price, price, direction
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _price_touched_level(
        previous_price: float | None,
        current_price: float,
        level: float,
    ) -> bool:
        if previous_price is None:
            return abs(current_price - level) < 1e-9
        lower = min(previous_price, current_price)
        upper = max(previous_price, current_price)
        return lower <= level <= upper

    @staticmethod
    def _short(symbol: str) -> str:
        return symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")

    @staticmethod
    def _now() -> datetime:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
