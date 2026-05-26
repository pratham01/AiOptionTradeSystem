import logging
from trade_system.config import Settings
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

logging.basicConfig(level=logging.INFO)

def test_connection():
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    
    print(f"Testing with token: {token[:10]}...")
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator
    )
    
    # Test 1: Authentication
    if broker.authenticate():
        print("✅ Authentication Successful")
    else:
        print("❌ Authentication Failed")
        return

    # Test 2: Quotes
    symbols = ["NSE:NIFTY50-INDEX", "NSE:SBIN-EQ"]
    try:
        quotes = broker.get_quotes(symbols)
        print(f"✅ Quotes Received: {list(quotes.keys())}")
    except Exception as e:
        print(f"❌ Quotes Failed: {e}")

if __name__ == "__main__":
    test_connection()
