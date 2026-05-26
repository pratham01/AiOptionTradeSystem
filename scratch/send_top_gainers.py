import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path("src").resolve()))

from trade_system.application.agent.postmarket_improver_agent import PostMarketImproverAgent
from trade_system.config.settings import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient

async def main():
    print("Testing PostMarket Improver Agent Top Gainers...")
    settings = Settings.load()  # Use load()!
    
    token = settings.fyers.access_token
    
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=None,
    )
    
    agent = PostMarketImproverAgent(broker, settings)
    
    from datetime import date
    today_str = date.today().strftime('%Y%m%d')
    lock_file = Path(settings.data_dir) / "top_gainers" / f"sent_lock_{today_str}.txt"
    if lock_file.exists():
        lock_file.unlink()
        
    from trade_system.application.analysis.top_gainers import NSETop100GainersFetcher
    from trade_system.infrastructure.data.nse_universe import NSE_UNIVERSE
    
    fetcher = NSETop100GainersFetcher(broker=agent.broker)
    
    print(f"Fetching quotes for Nifty 500 in batches of 50...")
    all_quotes = fetcher.fetch_all_quotes(symbols=NSE_UNIVERSE, batch_size=50)
    
    if not all_quotes:
        print("Failed to fetch any quotes! Check Fyers Authentication!")
        return
        
    filtered = fetcher.apply_filters(all_quotes)
    result = fetcher.get_top_gainers(filtered, top_n=10)
    
    fetcher.save_results(result)
    fetcher.send_telegram_report(result, agent.gainer_notifier, max_rows=10)
    
    if lock_file.parent.exists():
        lock_file.write_text("Sent manually.")
    
    print(f"Successfully pushed Top 10 Gainers from Nifty 500 to Telegram!")

if __name__ == "__main__":
    asyncio.run(main())
