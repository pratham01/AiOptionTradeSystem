import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging
logging.basicConfig(level=logging.INFO)

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping

def test_dhan():
    print("Loading settings...")
    settings = Settings.load()
    
    # Dhan is enabled in .env
    
    print("Initializing BrokerManager...")
    manager = get_broker_manager(settings)
    
    # Verify Dhan was initialized
    if "dhan" not in manager.brokers:
        print("ERROR: Dhan broker not initialized. Check credentials or config.")
        return
        
    print("Dhan broker successfully initialized.")
    fo_metadata = get_sector_mapping()
    symbols = list(fo_metadata.keys())[:10]  # Just test first 10
    
    print(f"Fetching quotes from Dhan for {len(symbols)} symbols...")
    try:
        quotes = manager._try_broker_quotes("dhan", symbols)
        print(f"Quotes fetched successfully: {len(quotes)} items.")
        for sym, q in list(quotes.items())[:3]:
            print(f" - {sym}: LTP={q.last_price}, PrevClose={q.previous_close}")
    except Exception as e:
        print(f"ERROR: Failed to fetch quotes from Dhan: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dhan()
