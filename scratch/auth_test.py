import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))

from trade_system.config.settings import Settings
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient

settings = Settings()
auth_service = FyersAuthService(settings)
token = auth_service.get_valid_token()

broker = FyersBrokerClient(
    client_id=settings.fyers.client_id,
    access_token=token,
    user_id=settings.fyers.user_id,
    authenticator=auth_service.authenticator,
)

print(f"Authenticating Broker...")
success = broker.authenticate()
print(f"Authentication success: {success}")

if success:
    res = broker.get_quotes(["NSE:RELIANCE-EQ"])
    print("Quotes:", res)
