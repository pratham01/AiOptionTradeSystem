import sys
import json
from pathlib import Path

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from fyers_apiv3 import fyersModel

def main():
    settings = Settings.load()
    token_path = Path(".secrets/fyers_token.json")
    if not token_path.exists():
        print("Token file not found.")
        return
        
    token_data = json.loads(token_path.read_text())
    access_token = token_data.get("access_token", "")
    print(f"Token (first 10 chars): {access_token[:10] if access_token else 'None'}")
    
    # Extract JWT
    jwt_token = access_token.split(":", 1)[1] if ":" in access_token else access_token
    
    fyers = fyersModel.FyersModel(
        client_id=settings.fyers.client_id,
        is_async=False,
        token=jwt_token,
    )
    
    print("Calling get_profile()...")
    profile = fyers.get_profile()
    print(f"Profile response: {profile}")
    
    print("\nCalling quotes for RELIANCE...")
    quotes = fyers.quotes(data={"symbols": "NSE:RELIANCE-EQ"})
    print(f"Quotes response: {quotes}")

if __name__ == "__main__":
    main()
