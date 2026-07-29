import sys
from pathlib import Path
import inspect

root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker as LegacyFyersBroker
from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker as ClientFyersBroker
from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

print("Legacy FyersBroker fetch_history signature:")
print(inspect.signature(LegacyFyersBroker.fetch_history))

print("\nClient FyersBroker fetch_history signature:")
print(inspect.signature(ClientFyersBroker.fetch_history))

try:
    settings = Settings.load()
    auth_service = FyersAuthService(settings)
    token = auth_service.get_valid_token()
    print(f"\nToken retrieved successfully. Length: {len(token) if token else 0}")
    
    broker = LegacyFyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth_service.authenticator
    )
    
    print("\nAttempting to call LegacyFyersBroker.fetch_history positionally or keyword-based:")
    import datetime
    end = datetime.date.today()
    start = end - datetime.timedelta(days=7)
    
    # Try calling it exactly like supertrend_touch_agent.py
    df = broker.fetch_history(
        symbol="NSE:TCS-EQ",
        resolution="15",
        range_from=start.isoformat(),
        range_to=end.isoformat()
    )
    print(f"Success! Result shape: {df.shape}")
except Exception as e:
    print(f"Error during legacy call: {e}")
    import traceback
    traceback.print_exc()

try:
    broker_client = ClientFyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth_service.authenticator
    )
    
    print("\nAttempting to call ClientFyersBroker.fetch_history exactly like supertrend_touch_agent.py:")
    df_client = broker_client.fetch_history(
        symbol="NSE:TCS-EQ",
        resolution="15",
        range_from=start.isoformat(),
        range_to=end.isoformat()
    )
    print(f"Success! Result shape: {df_client.shape}")
except Exception as e:
    print(f"Error during client call: {e}")
    import traceback
    traceback.print_exc()
