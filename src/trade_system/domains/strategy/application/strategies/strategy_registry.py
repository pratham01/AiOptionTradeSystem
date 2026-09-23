"""Strategy Registry and Factory implementing the Factory & Strategy Design Patterns."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

from trade_system.domains.strategy.application.strategies.base import BaseStrategy
from trade_system.domains.strategy.application.strategies.sma_cross import SmaCrossStrategy
from trade_system.domains.strategy.application.strategies.rvol_trend import RvolTrendStrategy
from trade_system.domains.strategy.application.strategies.volume_divergence import VolumeDivergenceStrategy
from trade_system.domains.strategy.application.strategies.volume_profile import VolumeProfileStrategy
from trade_system.domains.strategy.application.strategies.bollinger_options import BollingerOptionStrategy
from trade_system.domains.strategy.application.strategies.prajwal_price_action import PrajwalPriceActionStrategy
from trade_system.domains.strategy.application.strategies.supertrend_strategy import SupertrendStrategy
from trade_system.domains.strategy.application.strategies.orb_strategy import OrbStrategy
from trade_system.domains.strategy.application.strategies.gamma_blast_strategy import GammaBlastStrategy
from trade_system.domains.strategy.application.strategies.sniper_reversal_strategy import SniperReversalStrategy
from trade_system.domains.strategy.application.strategies.intraday_edge_strategy import IntradayEdgeStrategy
from trade_system.domains.strategy.application.strategies.smc_strategy import SmartMoneyConceptStrategy
from trade_system.shared.exceptions import StrategyNotFoundError

LOGGER = logging.getLogger(__name__)


def registered_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Returns the dictionary of all registered strategy classes."""
    return {
        SupertrendStrategy.name: SupertrendStrategy,
        OrbStrategy.name: OrbStrategy,
        GammaBlastStrategy.name: GammaBlastStrategy,
        SniperReversalStrategy.name: SniperReversalStrategy,
        IntradayEdgeStrategy.name: IntradayEdgeStrategy,
        SmaCrossStrategy.name: SmaCrossStrategy,
        RvolTrendStrategy.name: RvolTrendStrategy,
        VolumeDivergenceStrategy.name: VolumeDivergenceStrategy,
        VolumeProfileStrategy.name: VolumeProfileStrategy,
        BollingerOptionStrategy.name: BollingerOptionStrategy,
        PrajwalPriceActionStrategy.name: PrajwalPriceActionStrategy,
        SmartMoneyConceptStrategy.name: SmartMoneyConceptStrategy,
        "smart_money_concepts": SmartMoneyConceptStrategy,
    }


class StrategyFactory:
    """Factory Method pattern implementation to create and configure strategy instances."""

    @classmethod
    def create_strategy(cls, name: str, **kwargs: Any) -> BaseStrategy:
        """Instantiate a strategy by its registered name."""
        registry = registered_strategies()
        strategy_cls = registry.get(name.lower())
        if not strategy_cls:
            available = ", ".join(sorted(registry.keys()))
            raise StrategyNotFoundError(f"Unknown strategy '{name}'. Available strategies: {available}")
        LOGGER.debug("Instantiating strategy '%s' with params %s", name, kwargs)
        return strategy_cls(**kwargs)

    @classmethod
    def list_available_strategies(cls) -> List[str]:
        """List all available strategy identifiers."""
        return sorted(list(registered_strategies().keys()))


def build_strategy(name: str, **kwargs: Any) -> BaseStrategy:
    """Convenience functional builder (backward compatible)."""
    return StrategyFactory.create_strategy(name, **kwargs)
