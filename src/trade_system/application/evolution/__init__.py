"""Self-Evolution Engine for the Agentic Trading System."""

from trade_system.application.evolution.trade_logger import TradeLogger
from trade_system.application.evolution.outcome_tracker import OutcomeTracker
from trade_system.application.evolution.feature_analyzer import FeatureAnalyzer, FeatureImportance
from trade_system.application.evolution.weight_evolver import WeightEvolver
from trade_system.application.evolution.evolution_loop import EvolutionLoop

__all__ = [
    "TradeLogger",
    "OutcomeTracker",
    "FeatureAnalyzer",
    "FeatureImportance",
    "WeightEvolver",
    "EvolutionLoop",
]
