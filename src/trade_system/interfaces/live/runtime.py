from __future__ import annotations

import logging

from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.data.storage import CsvDataCatalog
from trade_system.interfaces.live.collector import LiveMarketDataService
from trade_system.shared.utils.logging_utils import configure_logging

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
    broker_manager = get_broker_manager(settings)
    
    return LiveMarketDataService(
        broker=broker,
        broker_manager=broker_manager,
        catalog=catalog,
        symbols=settings.live_symbols,
        timeframe_minutes=settings.live_timeframe_minutes,
        strategy_timeframe_minutes=settings.indicator_config.trend_timeframe_minutes,
        indicator_config=settings.indicator_config,
        settings=settings,
    )


def create_live_orchestrator(settings: Settings | None = None):
    """Creates the modern modular LiveMarketDataOrchestrator with separate Index and FO collectors."""
    from trade_system.interfaces.live.collectors.orchestrator import LiveMarketDataOrchestrator

    settings = settings or Settings.load()
    settings.ensure_directories()
    auth_service = FyersAuthService(settings)
    token = auth_service.get_valid_token()

    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth_service.authenticator,
    )
    return LiveMarketDataOrchestrator(broker=broker, settings=settings)


def run_live_trading_bot() -> int:
    settings = Settings.load()
    configure_logging(settings.log_level)
    service = create_live_market_service(settings)

    # ── Startup Data Backfill ──────────────────────────────────────────────────
    # Backfill any missing historical data before the live stream starts.
    # This ensures the dashboard always has recent data even if the bot was offline.
    try:
        from trade_system.interfaces.live.data_sync_service import DataSyncService
        LOGGER.info("Running startup data backfill (filling gaps in DB)...")
        sync_svc = DataSyncService(broker=service.broker, settings=settings)
        health = sync_svc.get_data_health_report()
        needs_sync = any(not v["is_current"] for v in health.values())
        if needs_sync:
            totals = sync_svc.backfill_all(resolutions=["15", "D"])
            LOGGER.info("Startup backfill complete: %s", totals)
        else:
            LOGGER.info("DB is current — skipping backfill")
    except Exception as _sync_err:
        LOGGER.warning("Startup backfill failed (non-fatal): %s", _sync_err)
    # ─────────────────────────────────────────────────────────────────────────

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


