"""Core domain models for the trading system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalType(Enum):
    ENTRY_LONG = "ENTRY_LONG"
    EXIT_LONG = "EXIT_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_SHORT = "EXIT_SHORT"


class TradeStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class TradeHorizon(Enum):
    """Whether the trade is intraday or swing (multi-day)."""
    INTRADAY = "INTRADAY"
    SWING = "SWING"


class TradeDirection(Enum):
    """Option trade direction."""
    CALL = "CALL"   # Bullish — buy CE
    PUT = "PUT"     # Bearish — buy PE


class MarketRegime(Enum):
    """Broad market regime classification."""
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class TradeOutcome(Enum):
    """Outcome of a suggested/executed trade."""
    PENDING = "PENDING"
    WIN = "WIN"
    LOSS = "LOSS"
    NEUTRAL = "NEUTRAL"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# Primitive market data
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Signal (raw indicator signal)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Signal:
    id: str
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    price: float
    confidence: float = 0.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "price": self.price,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Trade (execution record)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Trade:
    id: str
    symbol: str
    entry_signal: Signal
    entry_price: Decimal
    quantity: int
    status: TradeStatus = TradeStatus.PENDING
    exit_signal: Signal | None = None
    exit_price: Decimal | None = None
    pnl: Decimal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_pnl(self) -> Decimal | None:
        """Calculate P&L if trade is closed."""
        if self.status != TradeStatus.CLOSED or self.exit_price is None:
            return None
        qty = Decimal(self.quantity)
        if self.entry_signal.signal_type == SignalType.ENTRY_LONG:
            return (self.exit_price - self.entry_price) * qty
        else:
            return (self.entry_price - self.exit_price) * qty


@dataclass(slots=True)
class VolumeProfileSummary:
    """Summarized volume profile levels."""
    point_of_control: float
    value_area_low: float
    value_area_high: float
    total_volume: float = 0.0


# ---------------------------------------------------------------------------
# Market Context & Pre-Market
# ---------------------------------------------------------------------------

@dataclass
class PreMarketBrief:
    """Pre-market macroeconomic and news context."""
    timestamp: datetime
    usd_inr: float | None = None
    crude_oil: float | None = None
    sgx_nifty_trend: str = "NEUTRAL"
    fii_dii_net: float | None = None
    top_news: list[str] = field(default_factory=list)
    macro_events: list[str] = field(default_factory=list)
    overall_sentiment: str = "NEUTRAL"
    agent_summary: str = ""

@dataclass
class OptionChainAnalysis:
    """Detailed option chain analysis for index decisions."""
    timestamp: datetime
    symbol: str
    pcr: float
    max_pain: float
    atm_strike: float
    highest_ce_oi_strike: float
    highest_pe_oi_strike: float
    ce_oi_change_pct: float
    pe_oi_change_pct: float
    iv_percentile: float | None = None
    trend_bias: str = "NEUTRAL"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class OptionChainSnapshot:
    """Summarized option chain data for LLM context."""
    timestamp: datetime
    pcr: float | None = None            # Put-Call Ratio (OI based)
    pcr_volume: float | None = None     # Volume-based PCR
    vix: float | None = None            # India VIX
    atm_strike: float | None = None
    max_pain: float | None = None
    call_oi_change: float | None = None  # Today's call OI change %
    put_oi_change: float | None = None   # Today's put OI change %
    iv_skew: str | None = None          # "call_heavy", "put_heavy", "neutral"


@dataclass
class GlobalContext:
    """Global market indicators."""
    gift_nifty_pct: float | None = None
    sgx_nifty_pct: float | None = None
    sp500_pct: float | None = None
    nasdaq_pct: float | None = None
    dxy_pct: float | None = None        # Dollar index
    crude_oil_pct: float | None = None
    top_headlines: list[str] = field(default_factory=list)


@dataclass
class MarketContext:
    """Structured market context — produced by MarketContextAgent."""
    timestamp: datetime
    regime: MarketRegime = MarketRegime.UNKNOWN
    bias: str = "NEUTRAL"
    risk_level: str = "MEDIUM"
    vix: float | None = None
    pcr: float | None = None
    narrative: str = ""
    option_chain: OptionChainSnapshot | None = None
    global_ctx: GlobalContext | None = None
    score: float = 0.0
    tradeable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "bias": self.bias,
            "risk_level": self.risk_level,
            "vix": self.vix,
            "pcr": self.pcr,
            "narrative": self.narrative,
            "score": self.score,
            "tradeable": self.tradeable,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Option Parameters
# ---------------------------------------------------------------------------

@dataclass
class OptionParams:
    """Suggested option contract parameters."""
    direction: TradeDirection
    suggested_strike: float | None = None    # e.g., 24000
    expiry_type: str = "weekly"              # "weekly" or "monthly"
    atm_offset: int = 0                      # 0 = ATM, 1 = 1 OTM, -1 = 1 ITM
    max_premium: float | None = None         # Maximum acceptable premium
    min_delta: float | None = None           # Minimum delta for strike
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "suggested_strike": self.suggested_strike,
            "expiry_type": self.expiry_type,
            "atm_offset": self.atm_offset,
            "max_premium": self.max_premium,
            "min_delta": self.min_delta,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Setup Features (for evolution tracking)
# ---------------------------------------------------------------------------

@dataclass
class SetupFeatures:
    """
    Feature snapshot at the time of trade suggestion.
    Used by the Self-Evolution Engine to learn what predicts wins.
    """
    rsi_daily: float | None = None
    rsi_hourly: float | None = None
    adx: float | None = None
    volume_surge: float | None = None      # Vol vs 20-day avg
    atr_pct: float | None = None           # ATR as % of price
    vix: float | None = None
    pcr: float | None = None
    near_support: bool = False
    near_resistance: bool = False
    is_compressed: bool = False            # Bollinger/ATR squeeze
    vol_delta_positive: bool = False       # Cumulative volume delta bullish
    above_vwap: bool | None = None
    above_poc: bool | None = None          # Above Volume Profile POC
    breakout_type: str | None = None       # "daily_high", "weekly_high", "none"
    pattern: str | None = None            # Candlestick pattern name
    market_regime: str | None = None
    alignment_score: float | None = None  # Multi-TF alignment 0-1
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Trade Suggestion (output of the full agentic pipeline)
# ---------------------------------------------------------------------------

@dataclass
class TradeSuggestion:
    """
    The primary output of the TradeOrchestrator.
    Represents one actionable trade idea for the session.
    """
    id: str
    symbol: str
    timestamp: datetime
    direction: TradeDirection
    horizon: TradeHorizon

    # Pricing
    entry_zone_low: float
    entry_zone_high: float
    target: float
    stop_loss: float

    # Option details
    option_params: OptionParams

    # Agent outputs
    confidence: float               # 0.0 – 1.0
    narrative: str                  # LLM-generated reasoning
    setup_features: SetupFeatures   # For evolution tracking

    # Lifecycle
    outcome: TradeOutcome = TradeOutcome.PENDING
    actual_entry: float | None = None
    actual_exit: float | None = None
    actual_pnl_pct: float | None = None

    # Metadata
    is_nifty: bool = False
    sector: str = "Unknown"
    tags: list[str] = field(default_factory=list)  # e.g., ["breakout", "swing", "high_vix"]

    def risk_reward(self) -> float:
        """Calculate risk-reward ratio."""
        risk = abs(self.entry_zone_high - self.stop_loss)
        reward = abs(self.target - self.entry_zone_high)
        return round(reward / risk, 2) if risk > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "horizon": self.horizon.value,
            "entry_zone": [self.entry_zone_low, self.entry_zone_high],
            "target": self.target,
            "stop_loss": self.stop_loss,
            "option_params": self.option_params.to_dict(),
            "confidence": round(self.confidence, 3),
            "narrative": self.narrative,
            "setup_features": self.setup_features.to_dict(),
            "outcome": self.outcome.value,
            "risk_reward": self.risk_reward(),
            "is_nifty": self.is_nifty,
            "sector": self.sector,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Session Plan (container for all suggestions in a trading session)
# ---------------------------------------------------------------------------

@dataclass
class SessionPlan:
    """
    The full daily/session trading plan — max 2 Nifty + 3 FO suggestions.
    """
    date: str                                    # "YYYY-MM-DD"
    generated_at: datetime = field(default_factory=datetime.now)
    market_context: MarketContext | None = None

    nifty_suggestions: list[TradeSuggestion] = field(default_factory=list)   # max 2
    fo_suggestions: list[TradeSuggestion] = field(default_factory=list)       # max 3

    # Meta
    total_screened: int = 0
    agent_notes: str = ""
    is_tradeable_day: bool = True

    MAX_NIFTY = 2
    MAX_FO = 3

    def all_suggestions(self) -> list[TradeSuggestion]:
        return self.nifty_suggestions + self.fo_suggestions

    def add_nifty(self, s: TradeSuggestion) -> bool:
        if len(self.nifty_suggestions) < self.MAX_NIFTY:
            self.nifty_suggestions.append(s)
            return True
        return False

    def add_fo(self, s: TradeSuggestion) -> bool:
        if len(self.fo_suggestions) < self.MAX_FO:
            self.fo_suggestions.append(s)
            return True
        return False

    def is_budget_full(self) -> bool:
        return (len(self.nifty_suggestions) >= self.MAX_NIFTY and
                len(self.fo_suggestions) >= self.MAX_FO)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "generated_at": self.generated_at.isoformat(),
            "market_context": self.market_context.to_dict() if self.market_context else None,
            "nifty_suggestions": [s.to_dict() for s in self.nifty_suggestions],
            "fo_suggestions": [s.to_dict() for s in self.fo_suggestions],
            "total_screened": self.total_screened,
            "agent_notes": self.agent_notes,
            "is_tradeable_day": self.is_tradeable_day,
        }
