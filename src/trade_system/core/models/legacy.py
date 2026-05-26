from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""


@dataclass(slots=True)
class Tick:
    symbol: str
    timestamp: datetime
    ltp: float
    volume: float


@dataclass(slots=True)
class Signal:
    action: str
    reason: str = ""
    size: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Position:
    side: int = 0
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_time: datetime | None = None


@dataclass(slots=True)
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    reason: str
