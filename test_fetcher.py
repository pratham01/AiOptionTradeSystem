import asyncio
import logging
from trade_system.application.analysis.top_gainers import NSETop100GainersFetcher
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.config import Settings

logging.basicConfig(level=logging.INFO)

async def test_fetcher():
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator
    )
    
    fetcher = NSETop100GainersFetcher(broker=broker)
    all_quotes = fetcher.fetch_all_quotes(batch_size=200)
    print(f"Total quotes fetched: {len(all_quotes)}")
    
    filtered = fetcher.apply_filters(all_quotes)
    print(f"Passed filters: {len(filtered)}")
    
    result = fetcher.get_top_gainers(filtered, top_n=10)
    for i, g in enumerate(result.gainers, 1):
        print(f"{i}. {g.name}: {g.change_pct}%")

if __name__ == "__main__":
    asyncio.run(test_fetcher())
