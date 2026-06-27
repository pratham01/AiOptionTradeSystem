import sys
from pathlib import Path
import logging

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

logging.basicConfig(level=logging.INFO)

def test_fyers():
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator
    )
    
    broker.verify_session()
    
    # Try fetching for NSE:ABB-EQ which failed
    symbol = "NSE:ABB-EQ"
    payload = {
        "symbol": symbol,
        "resolution": "D",
        "date_format": "1",
        "range_from": "2025-05-27",
        "range_to": "2026-05-26",
        "cont_flag": "1",
    }
    
    print("Testing history request for NSE:ABB-EQ...")
    response = broker.fyers.history(data=payload)
    print(f"Response: {response}")
    
    # Try testing NSE:CUMMINSIND-EQ which succeeded
    symbol_ok = "NSE:CUMMINSIND-EQ"
    payload_ok = {
        "symbol": symbol_ok,
        "resolution": "D",
        "date_format": "1",
        "range_from": "2025-05-27",
        "range_to": "2026-05-26",
        "cont_flag": "1",
    }
    print(f"\nTesting history request for {symbol_ok}...")
    response_ok = broker.fyers.history(data=payload_ok)
    print(f"Response (should be ok): {response_ok.get('s')}")

if __name__ == "__main__":
    test_fyers()
