"""
Broker abstraction layer (Port in Hexagonal Architecture).

This defines the interface that all broker implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Protocol


class BrokerError(Exception):
    """Base exception for broker operations."""
    pass


class AuthenticationError(BrokerError):
    """Authentication failed."""
    pass


class DataFetchError(BrokerError):
    """Failed to fetch data."""
    pass


class OrderError(BrokerError):
    """Order placement failed."""
    pass


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_MARKET = "SL_MARKET"


class Exchange(Enum):
    """Supported exchanges."""
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """Standardized market quote."""

    symbol: str
    exchange: str
    last_price: float
    open: float
    high: float
    low: float
    close: float
    previous_close: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    ltp: float = 0.0

    @property
    def day_range_percent(self) -> float:
        """Calculate intraday range percentage."""
        if self.open == 0:
            return 0.0
        return ((self.high - self.low) / self.open) * 100


@dataclass(frozen=True, slots=True)
class HistoricalData:
    """Historical OHLCV data point."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Standardized order request."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: float = 0.0
    trigger_price: float = 0.0
    disclosed_quantity: int = 0
    tag: str = ""


@dataclass(frozen=True, slots=True)
class OrderResponse:
    """Standardized order response."""

    order_id: str
    status: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    average_price: float
    message: str = ""


class Broker(ABC):
    """Abstract base class for broker implementations."""

    def __init__(self, name: str, client_id: str, access_token: str) -> None:
        self.name = name
        self.client_id = client_id
        self.access_token = access_token
        self._authenticated = False

    @property
    def is_authenticated(self) -> bool:
        """Check if broker is authenticated."""
        return self._authenticated

    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the broker.

        Returns:
            True if authentication successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """
        Get real-time quotes for given symbols.

        Args:
            symbols: List of instrument symbols (e.g., ["NSE:RELIANCE-EQ"])

        Returns:
            Dictionary mapping symbol to MarketQuote.

        Raises:
            DataFetchError: If data fetch fails.
            AuthenticationError: If not authenticated.
        """
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "DAY",
    ) -> list[HistoricalData]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Instrument symbol.
            start_date: Start date.
            end_date: End date.
            timeframe: Candle timeframe (DAY, 1H, 30MIN, etc.)

        Returns:
            List of HistoricalData points.

        Raises:
            DataFetchError: If data fetch fails.
        """
        pass

    @abstractmethod
    def get_market_status(self) -> dict[str, Any]:
        """
        Get current market status.

        Returns:
            Dictionary with market status information.
        """
        pass

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an order.

        Args:
            order: OrderRequest with order details.

        Returns:
            OrderResponse with order status.

        Raises:
            OrderError: If order placement fails.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse | None:
        """
        Get status of an order.

        Args:
            order_id: Order ID.

        Returns:
            OrderResponse if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dictionaries.
        """
        pass

    @abstractmethod
    def get_holdings(self) -> list[dict[str, Any]]:
        """
        Get holdings.

        Returns:
            List of holding dictionaries.
        """
        pass

    @abstractmethod
    def get_funds(self) -> dict[str, float]:
        """
        Get available funds.

        Returns:
            Dictionary with fund information.
        """
        pass


class DataBroker(Broker):
    """Broker that only supports data operations (no trading)."""

    def place_order(self, order: OrderRequest) -> OrderResponse:
        raise NotImplementedError(f"{self.name} does not support order placement")

    def get_order_status(self, order_id: str) -> OrderResponse | None:
        raise NotImplementedError(f"{self.name} does not support order operations")

    def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{self.name} does not support position queries")

    def get_holdings(self) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{self.name} does not support holding queries")

    def get_funds(self) -> dict[str, float]:
        raise NotImplementedError(f"{self.name} does not support fund queries")


class TradingBroker(Broker):
    """Full-featured broker supporting both data and trading."""
    pass
