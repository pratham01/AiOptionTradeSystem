import sys
from pathlib import Path

# Add src to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker

settings = Settings.load()
auth = FyersAuthService(settings)
token = auth.get_valid_token()
broker = FyersBroker(client_id=settings.fyers.client_id, access_token=token, user_id=settings.fyers.user_id, authenticator=auth.authenticator)
broker.verify_session()

# Test quotes call first
res_quotes = broker.fyers.quotes(data={"symbols": "NSE:NIFTY50-INDEX"})
print("Quotes response:", res_quotes)

# Test optionschain call
data = {"symbol": "NSE:NIFTY50-INDEX", "strikecount": 10, "greeks": "1"}
res_oc = broker.fyers.optionchain(data=data)
print("Optionschain response:", res_oc)
