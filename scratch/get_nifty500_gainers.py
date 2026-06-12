import sys
import asyncio
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.agent.top_gainers_agent import BrokerTopGainersAgent, create_fyers_broker, format_top_gainers_table
from trade_system.infrastructure.data.fo_universe import get_fo_universe

async def main():
    try:
        broker = create_fyers_broker()
        print("Connected to Fyers successfully.")
        
        # 1. Fetch Nifty 500
        agent = BrokerTopGainersAgent(broker=broker)
        print("Fetching top 10 gainers from Nifty 500 universe...")
        top_10_gainers = agent.top_gainers(top_n=10)
        
        # 2. Fetch F&O
        fo_agent = BrokerTopGainersAgent(broker=broker, symbols=get_fo_universe())
        print("Fetching top 5 and worst 5 F&O stocks...")
        fo_top_5, fo_worst_5 = fo_agent.top_and_worst(n=5)
        
        print("\n=== TOP 10 GAINERS (NIFTY 500) ===")
        print(format_top_gainers_table(top_10_gainers, title_prefix="Top (Nifty 500)"))
        
        print("\n=== TOP 5 GAINERS (F&O) ===")
        print(format_top_gainers_table(fo_top_5, title_prefix="Top F&O"))
        
        print("\n=== WORST 5 LOSERS (F&O) ===")
        print(format_top_gainers_table(fo_worst_5, title_prefix="Worst F&O"))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
