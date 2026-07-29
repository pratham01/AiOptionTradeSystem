import logging
import os
from trade_system.shared.config import Settings
from fyers_apiv3 import fyersModel

logging.basicConfig(level=logging.DEBUG)

def debug_raw_fyers():
    settings = Settings.load()
    app_id = settings.fyers.client_id
    access_token = settings.fyers.access_token
    
    # Extract JWT part if it's formatted as app_id:token
    if ":" in access_token:
        token = access_token.split(":")[1]
    else:
        token = access_token

    print(f"DEBUG: App ID: {app_id}")
    print(f"DEBUG: Token Prefix: {token[:10]}...")

    fyers = fyersModel.FyersModel(client_id=app_id, token=token, is_async=False, log_path="logs/")
    
    # 1. Profile
    print("\n--- Profile Test ---")
    profile = fyers.get_profile()
    print(f"Profile Response: {profile}")

    # 2. Quotes
    print("\n--- Quotes Test ---")
    # Try a single symbol first
    symbols = "NSE:SBIN-EQ"
    quotes = fyers.quotes(data={"symbols": symbols})
    print(f"Quotes Response: {quotes}")

if __name__ == "__main__":
    debug_raw_fyers()
