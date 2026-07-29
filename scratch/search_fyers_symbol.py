import sys
from pathlib import Path
import logging

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

logging.basicConfig(level=logging.INFO)

def search_symbols():
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
    
    test_symbols = [
        "NSE:L&TFH-EQ",
        "NSE:LTF-EQ",
        "NSE:LTFH-EQ",
        "NSE:MCDOWELL-N-EQ",
        "NSE:MCDOWELL-EQ",
        "NSE:MCDOWELL_N-EQ",
        "NSE:GMRINFRA-EQ",
        "NSE:GMRI-EQ",
        "NSE:IDFC-EQ",
        "NSE:IDFCLTD-EQ",
        "NSE:PEL-EQ",
        "NSE:PIRAMAL-EQ",
        "NSE:TATAMOTORS-EQ",
        "NSE:SBIN-EQ" # Controls symbol to make sure quotes work
    ]
    
    print("Fetching quotes individually...")
    for sym in test_symbols:
        try:
            quotes = broker.get_quotes([sym])
            if sym in quotes:
                print(f"  ✅ {sym} is VALID. Close: {quotes[sym].get('lp') or quotes[sym].get('close')}")
            else:
                print(f"  ❌ {sym} is INVALID.")
        except Exception as e:
            print(f"  💥 Error querying {sym}: {e}")

if __name__ == "__main__":
    search_symbols()
