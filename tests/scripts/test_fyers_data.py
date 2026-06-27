import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import os
from datetime import datetime, timedelta

from trade_system.config import Settings
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.brokers.factory import get_broker_manager
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator

def test_data():
    settings = Settings.load()
    authenticator = FyersAuthenticator(settings)
    broker = FyersBroker(
        client_id=settings.fyers_client_id,
        access_token=settings.fyers_access_token,
        user_id=settings.fyers_user_id,
        authenticator=authenticator
    )
    
    if not broker.authenticate():
        print("Failed to authenticate.")
        return

    symbol = "NSE:NIFTY50-INDEX"
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Fetching data for {symbol} from {from_date} to {to_date}...")
    df = broker.get_historical_data(symbol, "1", from_date, to_date)
    
    if df is not None and not df.empty:
        print(f"Successfully fetched {len(df)} rows.")
        print(df.tail())
    else:
        print("Failed to fetch data or data is empty.")

if __name__ == "__main__":
    test_data()
