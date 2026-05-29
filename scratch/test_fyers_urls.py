import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.factory import get_broker_manager

def test_urls():
    settings = Settings.load()
    manager = get_broker_manager(settings)
    broker = manager.get_broker("fyers")
    broker.authenticate()
    
    print(f"Fyers instance: {broker.fyers}")
    
    # Try calling quotes directly using SDK
    print("Calling broker.fyers.quotes directly...")
    try:
        res = broker.fyers.quotes(data={"symbols": "NSE:SBIN-EQ"})
        print(f"Raw SDK response: {res}")
    except Exception as e:
        print(f"Exception from SDK call: {e}")

if __name__ == "__main__":
    test_urls()
