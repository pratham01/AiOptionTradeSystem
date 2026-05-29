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

def test_ltfh():
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
    
    symbol = "NSE:L&TFH-EQ"
    
    # 1. Test using date format (string)
    payload_str = {
        "symbol": symbol,
        "resolution": "D",
        "date_format": "1",
        "range_from": "2025-05-27",
        "range_to": "2026-05-26",
        "cont_flag": "1",
    }
    print("Testing string format for NSE:L&TFH-EQ...")
    res_str = broker.fyers.history(data=payload_str)
    print(f"String format result: success={res_str.get('s')}, candles_count={len(res_str.get('candles', []))}")
    
    # 2. Test using epoch format
    import datetime
    start_epoch = int(datetime.datetime(2025, 5, 27).timestamp())
    end_epoch = int(datetime.datetime(2026, 5, 26).timestamp())
    payload_epoch = {
        "symbol": symbol,
        "resolution": "D",
        "date_format": "0",
        "range_from": str(start_epoch),
        "range_to": str(end_epoch),
        "cont_flag": "1",
    }
    print("\nTesting epoch format for NSE:L&TFH-EQ...")
    res_epoch = broker.fyers.history(data=payload_epoch)
    print(f"Epoch format result: {res_epoch}")

if __name__ == "__main__":
    test_ltfh()
