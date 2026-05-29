import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging
logging.basicConfig(level=logging.INFO)

from trade_system.config import Settings
from trade_system.infrastructure.brokers.factory import get_broker_manager
from trade_system.infrastructure.data.fo_universe import get_sector_mapping

def test_quotes():
    print("Loading settings...")
    settings = Settings.load()
    print("Initializing BrokerManager...")
    manager = get_broker_manager(settings)
    
    fo_metadata = get_sector_mapping()
    symbols = ["NSE:SBIN-EQ"]  # Just test one symbol
    print(f"Fetching quotes for {len(symbols)} symbols...")
    try:
        quotes = manager.get_quotes(symbols)
        print(f"Quotes fetched successfully: {len(quotes)} items.")
        for sym, q in list(quotes.items())[:3]:
            print(f" - {sym}: LTP={q.last_price}, PrevClose={q.previous_close}")
    except Exception as e:
        print(f"ERROR: Failed to fetch quotes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quotes()
