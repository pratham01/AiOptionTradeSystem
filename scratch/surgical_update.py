import asyncio
import logging
from trade_system.application.analysis.fo_historical_service import FOHistoricalService
from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO)

async def surgical_update():
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator
    )
    
    service = FOHistoricalService(broker=broker, settings=settings)
    
    symbols = ["NSE:VEDL-EQ", "NSE:CHOLAFIN-EQ"]
    end = date.today()
    start = end - timedelta(days=5)
    
    for sym in symbols:
        print(f"Surgical update for {sym} (5m)...")
        service.collect(sym, "5", start, end)
        print(f"Done.")

if __name__ == "__main__":
    asyncio.run(surgical_update())
