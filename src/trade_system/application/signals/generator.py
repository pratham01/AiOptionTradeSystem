"""Signal generator registry and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import pandas as pd

from trade_system.core.models.domain import Signal


class SignalGenerator(ABC):
    """Base class for signal generators."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}

    @abstractmethod
    def generate(self, data: pd.DataFrame, symbol: str) -> list[Signal]:
        """Generate signals from data."""
        pass

    def validate_config(self) -> bool:
        """Validate generator configuration."""
        return True


class SignalRegistry:
    """Registry for signal generators."""

    _generators: dict[str, type[SignalGenerator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a signal generator."""
        def decorator(generator_class: type[SignalGenerator]) -> type[SignalGenerator]:
            cls._generators[name] = generator_class
            return generator_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type[SignalGenerator] | None:
        """Get a generator class by name."""
        return cls._generators.get(name)

    @classmethod
    def list(cls) -> list[str]:
        """List all registered generators."""
        return list(cls._generators.keys())

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> SignalGenerator | None:
        """Create a generator instance."""
        generator_class = cls.get(name)
        if generator_class:
            return generator_class(name, config)
        return None


# Convenience decorator
def register_signal_generator(name: str) -> Callable:
    """Decorator to register a signal generator."""
    return SignalRegistry.register(name)
