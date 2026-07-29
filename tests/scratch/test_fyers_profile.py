import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager

def test_profile():
    print("Loading settings...")
    settings = Settings.load()
    print("Initializing BrokerManager...")
    manager = get_broker_manager(settings)
    broker = manager.get_broker("fyers")
    broker.authenticate()
    
    print("Fetching profile...")
    try:
        profile = broker.fyers.get_profile()
        print(f"Profile fetched: {profile}")
    except Exception as e:
        print(f"Failed to fetch profile: {e}")

if __name__ == "__main__":
    test_profile()
