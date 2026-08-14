from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from trade_system.shared.notifications.telegram import TelegramNotifier
from trade_system.shared.notifications.templates import TelegramTemplateManager

LOGGER = logging.getLogger("AlertDispatcher")


@dataclass
class AlertConfig:
    """Debounce and routing configuration."""
    debounce_seconds: int = 300
    send_to_confirmed_channel: bool = False


class AlertDispatcher:
    """
    Centralised alert dispatcher. All Telegram messages pass through here.

    Features:
    - Integrated with TelegramTemplateManager for configurable message templates
    - Per-strategy and per-symbol cooldown tracking
    - Multi-channel routing (main channel vs confirmed channel)
    - Exception isolation (alert failures never crash the trading pipeline)
    """

    def __init__(
        self,
        notifier: TelegramNotifier,
        confirmed_notifier: TelegramNotifier | None = None,
        config: AlertConfig | None = None,
        template_manager: TelegramTemplateManager | None = None,
    ) -> None:
        self._notifier = notifier
        self._confirmed_notifier = confirmed_notifier or notifier
        self._config = config or AlertConfig()
        self._template_manager = template_manager or TelegramTemplateManager()
        self._last_sent: dict[str, datetime] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY TEMPLATE-BASED ALERTS
    # ─────────────────────────────────────────────────────────────────────────

    def dispatch_strategy_alert(
        self,
        strategy_name: str,
        symbol: str,
        context: Dict[str, Any],
        is_index: bool = True,
    ) -> bool:
        """
        Format and dispatch a strategy alert using configurable templates.
        Enforces enable/disable toggles, cooldowns, and channel routing.
        """
        if not self._template_manager.is_strategy_enabled(strategy_name, is_index=is_index):
            LOGGER.debug("Alert for %s (%s) is disabled by config.", strategy_name, symbol)
            return False

        if self._template_manager.should_debounce(strategy_name, symbol):
            LOGGER.debug("Alert for %s (%s) debounced.", strategy_name, symbol)
            return False

        try:
            message = self._template_manager.format_message(strategy_name, context)
            channel = self._template_manager.get_channel(strategy_name)
            self._deliver(message, channel)
            return True
        except Exception as exc:
            LOGGER.error("Failed to format/dispatch %s alert for %s: %s", strategy_name, symbol, exc)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC SEND METHODS (typed convenience methods)
    # ─────────────────────────────────────────────────────────────────────────

    def send_market_status(self, message: str) -> None:
        """Bot start/stop and market open/close notifications."""
        self._send("market_status", message, channel="main")

    def send_supertrend_flip(self, symbol: str, message: str) -> None:
        """ST Flip alert — sent to the confirmed Telegram channel."""
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
        notifier = self._confirmed_notifier if channel == "confirmed" else self._notifier
        try:
            notifier.send(message)
        except Exception as exc:
            LOGGER.error("Failed to deliver Telegram alert: %s", exc)

    def reset_debounce(self, key: str) -> None:
        self._last_sent.pop(key, None)

