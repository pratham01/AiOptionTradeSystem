import abc
from typing import Any, Optional


class TradeRepository(abc.ABC):
    """Abstract repository interface for Trades."""

    @abc.abstractmethod
    def update_trade_status(self, trade_id: str, status: str, entry_price: float) -> None:
        """Update the status and entry price of a trade."""
        pass


class LogRepository(abc.ABC):
    """Abstract repository interface for Logging system thoughts and actions."""

    @abc.abstractmethod
    def log_thought(self, agent: str, message: str, symbol: Optional[str] = None, action: Optional[str] = None) -> None:
        """Log an agent thought."""
        pass
