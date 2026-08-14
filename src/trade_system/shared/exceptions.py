"""Domain exception hierarchy for trade_system_v2."""
from __future__ import annotations


class TradeSystemError(Exception):
    """Base class for all application and domain errors."""
    pass


# ── Broker & Authentication Errors ──────────────────────────────────────────

class BrokerError(TradeSystemError):
    """Base class for broker communication and execution errors."""
    pass


class BrokerAuthError(BrokerError):
    """Raised when broker credentials, TOTP, or PIN authentication fails."""
    pass


class BrokerSessionExpiredError(BrokerAuthError):
    """Raised when access token is expired or invalidated."""
    pass


class BrokerRateLimitError(BrokerError):
    """Raised when broker API returns rate limit status (HTTP 429)."""
    pass


class BrokerOrderError(BrokerError):
    """Raised when order placement, cancellation, or modification fails."""
    pass


# ── Market Data & Persistence Errors ────────────────────────────────────────

class MarketDataError(TradeSystemError):
    """Base class for market data streaming, ingestion, or parsing errors."""
    pass


class DataGapError(MarketDataError):
    """Raised when unresolvable data gaps or discontinuities are detected."""
    pass


class DataSanityError(MarketDataError):
    """Raised when data fails structural sanity checks (e.g., negative price/volume)."""
    pass


class DatabaseLockError(MarketDataError):
    """Raised when SQLite encounters write conflicts or busy locks."""
    pass


# ── Strategy & Signal Errors ────────────────────────────────────────────────

class StrategyError(TradeSystemError):
    """Base class for strategy computation or signal evaluation errors."""
    pass


class StrategyNotFoundError(StrategyError):
    """Raised when an unrecognized strategy name is requested."""
    pass


class SignalValidationError(StrategyError):
    """Raised when a generated signal has invalid entry, stop loss, or target values."""
    pass


# ── Notification & Alerting Errors ──────────────────────────────────────────

class NotificationError(TradeSystemError):
    """Base class for Telegram, WhatsApp, or alert dispatching errors."""
    pass


class TemplateRenderError(NotificationError):
    """Raised when a notification template cannot be formatted."""
    pass


# ── Advisory & Agentic Errors ───────────────────────────────────────────────

class AgentError(TradeSystemError):
    """Base class for autonomous agent reasoning, memory, or action errors."""
    pass


class AgentDecisionError(AgentError):
    """Raised when agent reasoning loop fails to produce a viable decision."""
    pass
