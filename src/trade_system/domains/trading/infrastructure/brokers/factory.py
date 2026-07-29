"""
Broker Factory - Creates and manages broker instances with failover support.

This factory implements the Factory pattern and provides:
- Primary/backup broker failover
- Automatic broker switching on failures
- Centralized broker configuration
- Health monitoring
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Generator

from trade_system.shared.config.settings import BrokerConfig, DhanConfig, FyersConfig, Settings
from trade_system.domains.trading.domain.ports.broker import Broker, BrokerError, DataBroker, TradingBroker
from .dhan.client import DhanBroker

# Import Fyers broker (refactored to new interface)
from .fyers.client import FyersBroker

LOGGER = logging.getLogger(__name__)


class BrokerHealthStatus:
    """Tracks health status of brokers."""

    def __init__(self, broker: Broker) -> None:
        self.broker = broker
        self.is_healthy = True
        self.last_failure = None
        self.failure_count = 0
        self.last_success = datetime.now()

    def mark_success(self) -> None:
        """Mark successful operation."""
        self.is_healthy = True
        self.last_success = datetime.now()
        self.failure_count = 0

    def mark_failure(self, error: Exception) -> None:
        """Mark failed operation."""
        self.is_healthy = False
        self.last_failure = datetime.now()
        self.failure_count += 1
        LOGGER.warning(
            f"Broker {self.broker.name} failure #{self.failure_count}: {error}"
        )

    def can_retry(self, cooldown_minutes: int = 5) -> bool:
        """Check if broker can be retried after cooldown."""
        if self.is_healthy:
            return True
        if self.last_failure is None:
            return True
        return datetime.now() - self.last_failure > timedelta(minutes=cooldown_minutes)


class BrokerManager:
    """
    Manages multiple brokers with failover support.

    Usage:
        manager = BrokerManager(settings)
        
        # Get quotes with automatic failover
        quotes = manager.get_quotes(["NSE:RELIANCE-EQ"])
        
        # Use specific broker
        with manager.use_broker("fyers") as broker:
            data = broker.get_historical_data(...)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.brokers: dict[str, BrokerHealthStatus] = {}
        self.primary_broker_name = settings.primary_broker.lower()
        self.backup_broker_name = settings.backup_broker.lower() if settings.backup_broker else None

        # Initialize brokers
        self._initialize_brokers()

    def _initialize_brokers(self) -> None:
        """Initialize all configured brokers."""
        # Initialize Fyers
        if self.settings.fyers.enabled:
            try:
                fyers_broker = self._create_fyers_broker(self.settings.fyers)
                self.brokers["fyers"] = BrokerHealthStatus(fyers_broker)
                LOGGER.info("Fyers broker initialized")
            except Exception as e:
                LOGGER.error(f"Failed to initialize Fyers broker: {e}")

        # Initialize Dhan
        if self.settings.dhan.enabled:
            try:
                dhan_broker = self._create_dhan_broker(self.settings.dhan)
                self.brokers["dhan"] = BrokerHealthStatus(dhan_broker)
                LOGGER.info("Dhan broker initialized")
            except Exception as e:
                LOGGER.error(f"Failed to initialize Dhan broker: {e}")

        if not self.brokers:
            raise RuntimeError("No brokers configured. Please enable at least one broker.")

    def _create_fyers_broker(self, config: FyersConfig) -> FyersBroker:
        """Create Fyers broker instance."""
        # Pass the authenticator to enable automated token refresh
        from .legacy.fyers_auth import FyersAuthService
        auth_service = FyersAuthService(self.settings)
        return FyersBroker(
            client_id=config.client_id,
            access_token=config.access_token,
            user_id=config.user_id,
            authenticator=auth_service.authenticator,
        )

    @staticmethod
    def _create_dhan_broker(config: DhanConfig) -> DhanBroker:
        """Create Dhan broker instance."""
        return DhanBroker(
            client_id=config.client_id,
            access_token=config.access_token,
            api_key=config.api_key,
        )

    def get_broker(self, name: str | None = None) -> Broker:
        """
        Get a broker instance.

        Args:
            name: Specific broker name (fyers/dhan). If None, returns primary.

        Returns:
            Broker instance.

        Raises:
            RuntimeError: If no healthy broker available.
        """
        if name:
            broker_name = name.lower()
            if broker_name not in self.brokers:
                raise ValueError(f"Unknown broker: {name}")
            status = self.brokers[broker_name]
            if status.is_healthy or status.can_retry():
                return status.broker
            raise RuntimeError(f"Broker {name} is unhealthy and in cooldown")

        # Try primary broker first
        if self.primary_broker_name in self.brokers:
            primary_status = self.brokers[self.primary_broker_name]
            if primary_status.is_healthy or primary_status.can_retry():
                return primary_status.broker

        # Try backup broker
        if self.backup_broker_name and self.backup_broker_name in self.brokers:
            backup_status = self.brokers[self.backup_broker_name]
            if backup_status.is_healthy or backup_status.can_retry():
                LOGGER.warning(f"Falling back to backup broker: {self.backup_broker_name}")
                return backup_status.broker

        # Try any healthy broker
        for name, status in self.brokers.items():
            if status.is_healthy or status.can_retry():
                LOGGER.warning(f"Using broker {name} (not primary/backup)")
                return status.broker

        raise RuntimeError("No healthy brokers available")

    @contextmanager
    def use_broker(self, name: str) -> Generator[Broker, None, None]:
        """
        Context manager to use a specific broker.

        Args:
            name: Broker name (fyers/dhan)
        """
        broker = self.get_broker(name)
        try:
            yield broker
        except Exception as e:
            # Mark broker as unhealthy on failure
            if name in self.brokers:
                self.brokers[name].mark_failure(e)
            raise

    def get_quotes(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get quotes with automatic failover.

        Args:
            symbols: List of symbols to fetch quotes for

        Returns:
            Dictionary of quotes
        """
        last_error = None

        # Try primary broker first
        if self.primary_broker_name in self.brokers:
            try:
                return self._try_broker_quotes(self.primary_broker_name, symbols)
            except Exception as e:
                last_error = e
                self.brokers[self.primary_broker_name].mark_failure(e)

        # Try backup broker
        if self.backup_broker_name and self.backup_broker_name in self.brokers:
            try:
                return self._try_broker_quotes(self.backup_broker_name, symbols)
            except Exception as e:
                last_error = e
                self.brokers[self.backup_broker_name].mark_failure(e)

        # Try any other healthy broker
        for name in self.brokers:
            if name not in [self.primary_broker_name, self.backup_broker_name]:
                try:
                    return self._try_broker_quotes(name, symbols)
                except Exception as e:
                    last_error = e
                    self.brokers[name].mark_failure(e)

        raise BrokerError(f"All brokers failed to fetch quotes. Last error: {last_error}")

    def _try_broker_quotes(self, broker_name: str, symbols: list[str]) -> dict[str, Any]:
        """Try to get quotes from a specific broker."""
        status = self.brokers[broker_name]
        broker = status.broker

        if not broker.is_authenticated:
            if not broker.authenticate():
                raise BrokerError(f"Failed to authenticate {broker_name}")

        quotes = broker.get_quotes(symbols)
        status.mark_success()
        return quotes

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "DAY",
    ) -> list[Any]:
        """
        Get historical data with automatic failover.
        """
        last_error = None

        # Try brokers in order of preference
        broker_order = [self.primary_broker_name]
        if self.backup_broker_name:
            broker_order.append(self.backup_broker_name)
        broker_order.extend([name for name in self.brokers if name not in broker_order])

        for broker_name in broker_order:
            if broker_name not in self.brokers:
                continue

            try:
                status = self.brokers[broker_name]
                broker = status.broker

                if not broker.is_authenticated:
                    if not broker.authenticate():
                        raise BrokerError(f"Failed to authenticate {broker_name}")

                data = broker.get_historical_data(
                    symbol=symbol,
                    start_date=start_date.date() if hasattr(start_date, "date") else start_date,
                    end_date=end_date.date() if hasattr(end_date, "date") else end_date,
                    timeframe=timeframe,
                )
                status.mark_success()
                return data

            except Exception as e:
                last_error = e
                self.brokers[broker_name].mark_failure(e)
                LOGGER.warning(f"Historical data fetch failed for {broker_name}: {e}")

        raise BrokerError(f"All brokers failed to fetch historical data. Last error: {last_error}")

    def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status of all brokers."""
        status = {}
        for name, health in self.brokers.items():
            status[name] = {
                "is_healthy": health.is_healthy,
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_failure": health.last_failure.isoformat() if health.last_failure else None,
                "failure_count": health.failure_count,
                "authenticated": health.broker.is_authenticated,
            }
        return status

    def authenticate_all(self) -> dict[str, bool]:
        """Authenticate all brokers."""
        results = {}
        for name, health in self.brokers.items():
            try:
                success = health.broker.authenticate()
                results[name] = success
                if success:
                    health.mark_success()
                else:
                    health.mark_failure(Exception("Authentication failed"))
            except Exception as e:
                results[name] = False
                health.mark_failure(e)
        return results

    def __repr__(self) -> str:
        return f"BrokerManager(brokers={list(self.brokers.keys())}, primary={self.primary_broker_name})"


# Singleton instance for easy access
_broker_manager: BrokerManager | None = None


def get_broker_manager(settings: Settings | None = None) -> BrokerManager:
    """Get singleton broker manager instance."""
    global _broker_manager
    if _broker_manager is None:
        if settings is None:
            settings = Settings.load()
        _broker_manager = BrokerManager(settings)
    return _broker_manager


def reset_broker_manager() -> None:
    """Reset singleton broker manager (for testing)."""
    global _broker_manager
    _broker_manager = None
