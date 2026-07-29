"""
AlertDispatcher — Single-responsibility module for all Telegram alert formatting and delivery.

Extracted from LiveMarketDataService (collector.py).
Principle: No alert logic should live inside the data or signal pipeline.
All Telegram message construction and sending passes through this class.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger("AlertDispatcher")


@dataclass
class AlertConfig:
    """Debounce and routing configuration."""
    # Minimum seconds between repeated alerts for the same symbol/type
    debounce_seconds: int = 300
    # Whether to also send to the confirmed (secondary) channel
    send_to_confirmed_channel: bool = False


class AlertDispatcher:
    """
    Centralised alert dispatcher. All Telegram messages go through here.

    Features:
    - Per-alert-type debounce to prevent spam
    - Dual-channel routing (main notifier + confirmed notifier)
    - Structured alert types for auditability
    - Graceful error handling (alert failure never crashes the main pipeline)
    """

    def __init__(
        self,
        notifier: TelegramNotifier,
        confirmed_notifier: TelegramNotifier | None = None,
        config: AlertConfig | None = None,
    ) -> None:
        self._notifier = notifier
        self._confirmed_notifier = confirmed_notifier or notifier
        self._config = config or AlertConfig()
        # {alert_key -> last_sent datetime}
        self._last_sent: dict[str, datetime] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC SEND METHODS (typed by alert category)
    # ─────────────────────────────────────────────────────────────────────────

    def send_market_status(self, message: str) -> None:
        """Bot start/stop and market open/close notifications."""
        self._send("market_status", message, channel="main")

    def send_supertrend_flip(self, symbol: str, message: str) -> None:
        """ST Flip alert — sent to the confirmed (STFlip) Telegram channel."""
        self._send(f"st_flip_{symbol}", message, channel="confirmed", debounce=False)

    def send_supertrend_touch(self, symbol: str, message: str) -> None:
        """ST Touch (re-entry) alert."""
        self._send(f"st_touch_{symbol}", message, channel="main")

    def send_premarket_report(self, symbol: str, message: str) -> None:
        """Daily premarket context report."""
        self._send(f"premarket_{symbol}", message, channel="main", debounce=False)

    def send_level_alert(self, symbol: str, level_name: str, message: str) -> None:
        """Price touching key S/R levels."""
        self._send(f"level_{symbol}_{level_name}", message, channel="main")

    def send_option_chain_alert(self, symbol: str, message: str) -> None:
        """Option chain OI-based signals."""
        self._send(f"oc_{symbol}", message, channel="main")

    def send_breakout_alert(self, symbol: str, message: str) -> None:
        """Breakout scanner signals."""
        self._send(f"breakout_{symbol}", message, channel="main")

    def send_sniper_signal(self, symbol: str, message: str) -> None:
        """Sniper reversal alerts."""
        self._send(f"sniper_{symbol}", message, channel="main", debounce=False)

    def send_gamma_blast(self, symbol: str, message: str) -> None:
        """Gamma Blast strategy alerts."""
        self._send(f"gamma_{symbol}", message, channel="confirmed", debounce=False)

    def send_eod_summary(self, message: str) -> None:
        """End of day performance summary."""
        self._send("eod_summary", message, channel="main", debounce=False)

    def send_raw(self, message: str, channel: str = "main") -> None:
        """Low-level send — use only for one-off messages without debounce."""
        self._deliver(message, channel)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL ROUTING
    # ─────────────────────────────────────────────────────────────────────────

    def _send(self, key: str, message: str, channel: str = "main", debounce: bool = True) -> None:
        """
        Core internal method. Applies debounce and routes to the right channel.
        """
        if debounce:
            last = self._last_sent.get(key)
            if last:
                elapsed = (datetime.now() - last).total_seconds()
                if elapsed < self._config.debounce_seconds:
                    LOGGER.debug(
                        "Debounced alert '%s' (sent %.0fs ago, cooldown=%ds)",
                        key, elapsed, self._config.debounce_seconds
                    )
                    return

        self._last_sent[key] = datetime.now()
        self._deliver(message, channel)

    def _deliver(self, message: str, channel: str) -> None:
        """Actually send the message, with error isolation."""
        notifier = self._confirmed_notifier if channel == "confirmed" else self._notifier
        try:
            notifier.send(message)
        except Exception as exc:
            # CRITICAL: Alert failures must never crash the trading pipeline
            LOGGER.error("Failed to deliver Telegram alert: %s", exc)

    def reset_debounce(self, key: str) -> None:
        """Manually clear debounce state for a specific alert key."""
        self._last_sent.pop(key, None)
