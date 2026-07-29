import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests
import json
from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager

def test_raw_quotes():
    settings = Settings.load()
    token = settings.fyers.access_token
    client_id = settings.fyers.client_id
    
    # Strip client_id prefix if present
    jwt_token = token.split(":", 1)[1] if ":" in token else token
    
    url = "https://api-t1.fyers.in/data/quotes"
    headers = {
        "Authorization": f"{client_id}:{jwt_token}",
        "Content-Type": "application/json"
    }
    params = {
        "symbols": "NSE:SBIN-EQ"
    }
    
    print(f"Making direct GET request to Fyers quotes endpoint...")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Body: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_raw_quotes()
