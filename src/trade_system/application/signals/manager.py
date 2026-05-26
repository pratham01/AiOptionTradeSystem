"""Signal manager for orchestrating signal generation and distribution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from trade_system.core.events.bus import EventBus, EventType, SignalEvent
from trade_system.core.models.domain import Signal
from trade_system.application.signals.generator import SignalGenerator, SignalRegistry

LOGGER = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    generators: list[dict[str, Any]] = field(default_factory=list)
    min_confidence: float = 0.5
    cooldown_minutes: int = 5
    max_signals_per_hour: int = 10


class SignalManager:
    """Manages signal generation and distribution."""

    def __init__(
        self,
        config: SignalConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config or SignalConfig()
        self.event_bus = event_bus
        self._generators: list[tuple[str, SignalGenerator]] = []
        self._signal_history: list[Signal] = []
        self._last_signal_time: dict[str, datetime] = {}

    def add_generator(self, name: str, config: dict[str, Any] | None = None) -> None:
        """Add a signal generator."""
        generator_class = SignalRegistry.get(name)
        if generator_class:
            generator = generator_class(name, config)
            self._generators.append((name, generator))
            LOGGER.info("Added signal generator: %s", name)
        else:
            LOGGER.warning("Unknown signal generator: %s", name)

    def process_data(self, data: pd.DataFrame, symbol: str) -> list[Signal]:
        """Process data through all generators."""
        all_signals: list[Signal] = []

        for name, generator in self._generators:
            try:
                signals = generator.generate(data, symbol)
                for signal in signals:
                    if self._validate_signal(signal):
                        all_signals.append(signal)
                        self._record_signal(signal)
                        self._distribute_signal(signal)
            except Exception as e:
                LOGGER.error("Error in generator %s: %s", name, e)

        return all_signals

    def _validate_signal(self, signal: Signal) -> bool:
        """Validate a signal before distribution."""
        # Confidence check
        if signal.confidence < self.config.min_confidence:
            return False

        # Cooldown check
        now = datetime.utcnow()
        last_time = self._last_signal_time.get(signal.symbol)
        if last_time and (now - last_time) < timedelta(minutes=self.config.cooldown_minutes):
            return False

        # Rate limit check
        hour_ago = now - timedelta(hours=1)
        recent_signals = [s for s in self._signal_history if s.timestamp > hour_ago]
        if len(recent_signals) >= self.config.max_signals_per_hour:
            return False

        return True

    def _record_signal(self, signal: Signal) -> None:
        """Record signal for history and cooldown."""
        self._signal_history.append(signal)
        self._last_signal_time[signal.symbol] = signal.timestamp

        # Trim history (keep last 1000)
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-1000:]

    def _distribute_signal(self, signal: Signal) -> None:
        """Distribute signal via event bus."""
        if self.event_bus:
            event = SignalEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                signal_type=signal.signal_type.value,
                price=signal.price,
                metadata=signal.metadata,
            )
            # Fire and forget
            import asyncio
            try:
                asyncio.create_task(self.event_bus.emit(EventType.SIGNAL, event))
            except RuntimeError:
                # No event loop running
                pass

    def get_signal_history(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> list[Signal]:
        """Get signal history."""
        result = self._signal_history

        if symbol:
            result = [s for s in result if s.symbol == symbol]

        if since:
            result = [s for s in result if s.timestamp >= since]

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get signal manager statistics."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        return {
            "total_signals": len(self._signal_history),
            "signals_last_hour": len([s for s in self._signal_history if s.timestamp > hour_ago]),
            "signals_last_day": len([s for s in self._signal_history if s.timestamp > day_ago]),
            "generators_active": len(self._generators),
            "config": {
                "min_confidence": self.config.min_confidence,
                "cooldown_minutes": self.config.cooldown_minutes,
            },
        }
