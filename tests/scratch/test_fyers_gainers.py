import sys
from pathlib import Path

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.advisory.application.agent.top_gainers_agent import create_fyers_broker

def main():
    try:
        broker = create_fyers_broker()
        print("Successfully authenticated with Fyers.")
        
        # Test a small batch
        test_symbols = ["NSE:RELIANCE-EQ", "NSE:SBIN-EQ", "NSE:TCS-EQ"]
        print(f"Fetching quotes for: {test_symbols}")
        
        quotes = broker.get_quotes(test_symbols)
        print(f"Quotes received: {len(quotes)} symbols.")
        for symbol, quote in quotes.items():
            print(f"  {symbol}: Last Price = {quote.last_price}, Change % = {quote.change_percent}, Volume = {quote.volume}")
            
    except Exception as e:
        print(f"Failed to fetch quotes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
