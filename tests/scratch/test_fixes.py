import sys
import asyncio
from pathlib import Path

# Add src to python path
sys.path.append(str(Path("src").resolve()))

from trade_system.domains.advisory.application.agent.fo_option_buyer_agent import load_latest_sector_performance
from trade_system.domains.advisory.application.agent.postmarket_improver_agent import PostMarketImproverAgent
from trade_system.shared.config.settings import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker

async def main():
    print("Testing FO Option Buyer Agent Sector Logic...")
    sector_perf = load_latest_sector_performance()
    print(f"Sector Performance Result (Top 5): {list(sector_perf.items())[:5]}")
    print(f"Total Sectors Loaded: {len(sector_perf)}")
    
    print("\n----------------------------------\n")
    
    print("Testing PostMarket Improver Agent Top Gainers (batch_size=50)...")
    settings = Settings()
    broker = FyersBroker(
        client_id=settings.fyers_client_id,
        access_token=settings.fyers_access_token,
        user_id=settings.fyers_user_id
    )
    
    # Just initialize the agent with broker as the first positional argument
    agent = PostMarketImproverAgent(broker, settings)
    
    from trade_system.domains.analysis.application.analysis.top_gainers import NSETop100GainersFetcher
    from trade_system.domains.market_data.infrastructure.data.nse_universe import NSE_UNIVERSE
    
    fetcher = NSETop100GainersFetcher(broker=agent.broker)
    
    # Test a small slice (first 100)
    test_universe = NSE_UNIVERSE[:100]
    
    print(f"Fetching quotes for {len(test_universe)} stocks in batches of 50...")
    all_quotes = fetcher.fetch_all_quotes(symbols=test_universe, batch_size=50)
    filtered = fetcher.apply_filters(all_quotes)
    result = fetcher.get_top_gainers(filtered, top_n=10)
    
    print(f"Successfully fetched {len(all_quotes)} quotes.")
    print(f"Top Gainers from slice:")
    for g in result.gainers[:3]:
        print(f"  {g.symbol}: {g.change_pct}%")

if __name__ == "__main__":
    asyncio.run(main())
