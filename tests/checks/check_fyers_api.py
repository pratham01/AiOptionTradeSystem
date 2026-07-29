import json
import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from fyers_apiv3 import fyersModel
from trade_system.shared.config import Settings

def main():
    settings = Settings.load()
    with open(".secrets/fyers_token.json") as f:
        token = json.load(f)["access_token"]
        if ":" in token:
            token = token.split(":", 1)[1]
        
    fyers = fyersModel.FyersModel(
        client_id=settings.fyers_client_id,
        is_async=False,
        token=token,
        log_path=None
    )
    
    # Try with a weekend range (Saturday to Sunday)
    payload_1 = {
        "symbol": "NSE:NIFTY50-INDEX",
        "resolution": "D",
        "date_format": "1",
        "range_from": "2026-06-20",
        "range_to": "2026-06-21",
        "cont_flag": "1"
    }
    print("Testing weekend range with YYYY-MM-DD:")
    response_1 = fyers.history(data=payload_1)
    print(response_1)
    
    # Try with date_format = "0" (epoch)
    from datetime import datetime
    start_epoch = int(datetime(2026, 6, 15).timestamp())
    end_epoch = int(datetime(2026, 6, 22).timestamp())
    payload_2 = {
        "symbol": "NSE:NIFTY50-INDEX",
        "resolution": "D",
        "date_format": "0",
        "range_from": start_epoch,
        "range_to": end_epoch,
        "cont_flag": "1"
    }
    print("\nTesting with Epoch:")
    response_2 = fyers.history(data=payload_2)
    print(response_2)

if __name__ == "__main__":
    main()
