from __future__ import annotations

import logging

from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.config import Settings
from trade_system.infrastructure.data.storage import CsvDataCatalog
from trade_system.interfaces.live.collector import LiveMarketDataService
from trade_system.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def create_live_market_service(settings: Settings | None = None) -> LiveMarketDataService:
    settings = settings or Settings.load()
    settings.ensure_directories()

    auth_service = FyersAuthService(settings)
    token = auth_service.get_valid_token()
    
    if not token:
        LOGGER.error("Failed to retrieve a valid Fyers token for Live Bot.")

    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth_service.authenticator,
    )
    catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
    return LiveMarketDataService(
        broker=broker,
        catalog=catalog,
        symbols=settings.live_symbols,
        timeframe_minutes=settings.live_timeframe_minutes,
        strategy_timeframe_minutes=settings.indicator_config.trend_timeframe_minutes,
        indicator_config=settings.indicator_config,
        settings=settings,
    )


def run_live_trading_bot() -> int:
    settings = Settings.load()
    configure_logging(settings.log_level)
    service = create_live_market_service(settings)

    LOGGER.info(
        "Starting live trading bot for %s | trend timeframe=%s min | market window=%s-%s",
        settings.live_symbols,
        settings.indicator_config.trend_timeframe_minutes,
        settings.market_start,
        settings.market_end,
    )
    try:
        service.run_forever()
    except KeyboardInterrupt:
        LOGGER.info("Keyboard interrupt received. Stopping live trading bot.")
        service.stop()
        service.shutdown = True
        return 0
    except Exception:
        LOGGER.exception("Live trading bot stopped due to an unexpected error.")
        service.stop()
        service.shutdown = True
        return 1
    return 0
