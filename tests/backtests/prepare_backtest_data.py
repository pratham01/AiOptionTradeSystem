import asyncio
import logging
from datetime import date, timedelta
from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient as FyersBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator
from trade_system.domains.analysis.application.analysis.fo_historical_service import FOHistoricalService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_backtest_data")

async def main():
    settings = Settings.load()
    authenticator = FyersAuthenticator(settings)
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=settings.fyers.access_token,
        user_id=settings.fyers.user_id,
        authenticator=authenticator,
    )
    
    if not broker.authenticate():
        logger.error("Failed to authenticate.")
        return

    service = FOHistoricalService(broker=broker, settings=settings)
    
    symbol = "NSE:NIFTY50-INDEX"
    from_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching 1 year of data for {symbol} starting from {from_date}...")
    
    # We call history_service directly to bypass FO_UNIVERSE checks
    service.history_service.collect(
        symbol=symbol,
        resolution="D",
        from_date=date.today() - timedelta(days=365),
        to_date=date.today(),
        sleep_seconds=0.1
    )
    
    logger.info("Data fetch complete.")

if __name__ == "__main__":
    asyncio.run(main())
