import asyncio
import logging
from datetime import date, timedelta
from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient as FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator
from trade_system.application.analysis.fo_historical_service import FOHistoricalService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_3m_data")

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
        return

    service = FOHistoricalService(broker=broker, settings=settings)
    
    symbol = "NSE:NIFTY50-INDEX"
    # Fetch last 30 days of 3-minute data
    # Note: Fyers 3m data limit is usually 60-100 days
    from_date = date.today() - timedelta(days=30)
    
    logger.info(f"Fetching 3-minute data for {symbol} from {from_date}...")
    
    service.history_service.collect(
        symbol=symbol,
        resolution="3",
        from_date=from_date,
        to_date=date.today(),
        sleep_seconds=0.2
    )
    
    logger.info("3m Data fetch complete.")

if __name__ == "__main__":
    asyncio.run(main())
