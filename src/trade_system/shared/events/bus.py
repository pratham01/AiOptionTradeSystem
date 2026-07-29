"""Event-driven architecture for real-time data flow."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine

LOGGER = logging.getLogger(__name__)


class EventType(Enum):
    MARKET_DATA = auto()
    SIGNAL = auto()
    ORDER = auto()
    POSITION_UPDATE = auto()
    NOTIFICATION = auto()
    
    # Swarm Orchestration Events
    SESSION_STARTED = auto()
    PREMARKET_NEWS_READY = auto()
    MARKET_CONTEXT_READY = auto()
    OPTION_CHAIN_READY = auto()
    TRADE_SUGGESTION_GENERATED = auto()
    SESSION_PLAN_READY = auto()


@dataclass(slots=True)
class MarketEvent:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"


@dataclass(slots=True)
class SignalEvent:
    symbol: str
    timestamp: datetime
    signal_type: str  # "ENTRY_LONG", "EXIT_LONG", "ENTRY_SHORT", "EXIT_SHORT"
    price: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NotificationEvent:
    channel: str  # "telegram", "webhook", etc.
    message: str
    priority: str = "normal"  # "low", "normal", "high", "critical"
    metadata: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Async event bus for decoupled component communication."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[..., Coroutine]]] = {
            et: [] for et in EventType
        }
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._running = False

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[..., Coroutine],
    ) -> None:
        """Subscribe a coroutine handler to an event type."""
        self._handlers[event_type].append(handler)
        LOGGER.debug("Handler subscribed to %s", event_type.name)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[..., Coroutine],
    ) -> None:
        """Unsubscribe a handler."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def emit(self, event_type: EventType, event: Any) -> None:
        """Emit an event to all subscribers."""
        await self._queue.put((event_type, event))

    async def _process_events(self) -> None:
        """Background task to process events."""
        while self._running:
            try:
                event_type, event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                handlers = self._handlers.get(event_type, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        LOGGER.error("Error in event handler: %s", e)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                LOGGER.error("Event processing error: %s", e)

    async def start(self) -> None:
        """Start the event processing loop."""
        self._running = True
        await self._process_events()

    async def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
