"""Signal generation and management."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trade_system.core.models.domain import MarketData, Signal, SignalType


@dataclass
class SignalContext:
    """Context passed to signal generators."""

    symbol: str
    timeframe: str
    data: list[MarketData]
    indicators: dict[str, Any]


class SignalGenerator(ABC):
    """Abstract base for all signal generators."""

    def __init__(self, name: str, min_confidence: float = 0.5) -> None:
        self.name = name
        self.min_confidence = min_confidence

    @abstractmethod
    def generate(self, context: SignalContext) -> Signal | None:
        """Generate a signal from market context. Return None if no signal."""
        pass

    def create_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        price: float,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        """Helper to create a signal with proper ID and timestamp."""
        return Signal(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            source=self.name,
            metadata=metadata or {},
        )


class CompositeSignalGenerator(SignalGenerator):
    """Combines multiple signal generators with weights."""

    def __init__(
        self,
        generators: list[tuple[SignalGenerator, float]],
        aggregation_threshold: float = 0.6,
    ) -> None:
        super().__init__("composite")
        self.generators = generators
        self.aggregation_threshold = aggregation_threshold

    def generate(self, context: SignalContext) -> Signal | None:
        signals: list[Signal] = []
        weights: list[float] = []

        for generator, weight in self.generators:
            signal = generator.generate(context)
            if signal:
                signals.append(signal)
                weights.append(weight)

        if not signals:
            return None

        # Weighted confidence aggregation
        total_weight = sum(weights)
        weighted_confidence = sum(
            s.confidence * w for s, w in zip(signals, weights)
        ) / total_weight

        if weighted_confidence < self.aggregation_threshold:
            return None

        # Majority voting for signal type
        long_votes = sum(w for s, w in zip(signals, weights) if s.signal_type == SignalType.ENTRY_LONG)
        short_votes = sum(w for s, w in zip(signals, weights) if s.signal_type == SignalType.ENTRY_SHORT)

        if long_votes > short_votes:
            signal_type = SignalType.ENTRY_LONG
        else:
            signal_type = SignalType.ENTRY_SHORT

        return self.create_signal(
            symbol=context.symbol,
            signal_type=signal_type,
            price=context.data[-1].close if context.data else 0.0,
            confidence=weighted_confidence,
            metadata={"sources": [s.source for s in signals]},
        )
