from __future__ import annotations

from trade_system.domains.strategy.application.strategies.base import BaseStrategy
from trade_system.domains.strategy.application.strategies.sma_cross import SmaCrossStrategy
from trade_system.domains.strategy.application.strategies.rvol_trend import RvolTrendStrategy
from trade_system.domains.strategy.application.strategies.volume_divergence import VolumeDivergenceStrategy
from trade_system.domains.strategy.application.strategies.volume_profile import VolumeProfileStrategy


def registered_strategies() -> dict[str, type[BaseStrategy]]:
    return {
        SmaCrossStrategy.name: SmaCrossStrategy,
        RvolTrendStrategy.name: RvolTrendStrategy,
        VolumeDivergenceStrategy.name: VolumeDivergenceStrategy,
        VolumeProfileStrategy.name: VolumeProfileStrategy,
    }


def build_strategy(name: str, **kwargs) -> BaseStrategy:
    registry = registered_strategies()
    if name not in registry:
        raise KeyError(f"Unknown strategy: {name}. Available: {', '.join(sorted(registry))}")
    return registry[name](**kwargs)
