"""Core domain layer: events, models, and signals."""

from trade_system.core.events.bus import EventBus, MarketEvent, SignalEvent
from trade_system.core.models.domain import (
    MarketData,
    Signal,
    SignalType,
    Trade,
    TradeStatus,
    TradeHorizon,
    TradeDirection,
    TradeOutcome,
    MarketRegime,
    MarketContext,
    OptionChainSnapshot,
    OptionChainAnalysis,
    GlobalContext,
    PreMarketBrief,
    OptionParams,
    SetupFeatures,
    TradeSuggestion,
    SessionPlan,
    VolumeProfileSummary,
)
from trade_system.core.signals import SignalGenerator
from trade_system.core.ports.broker import Broker, DataBroker, TradingBroker
from trade_system.core.ports.notifier import Notifier
from trade_system.core.ports.weights import WeightsProvider

__all__ = [
    "EventBus",
    "MarketEvent",
    "SignalEvent",
    "MarketData",
    "Signal",
    "SignalType",
    "Trade",
    "TradeStatus",
    "TradeHorizon",
    "TradeDirection",
    "TradeOutcome",
    "MarketRegime",
    "MarketContext",
    "OptionChainSnapshot",
    "OptionChainAnalysis",
    "GlobalContext",
    "PreMarketBrief",
    "OptionParams",
    "SetupFeatures",
    "TradeSuggestion",
    "SessionPlan",
    "SignalGenerator",
    "Broker",
    "DataBroker",
    "TradingBroker",
    "Notifier",
    "WeightsProvider",
]
