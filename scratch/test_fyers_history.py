import os
from datetime import datetime
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()

token = os.getenv("FYERS_ACCESS_TOKEN")
client_id = os.getenv("FYERS_CLIENT_ID")

# Strip client_id: if present, matching the codebase logic
if ":" in token:
    sdk_token = token.split(":", 1)[1]
else:
    sdk_token = token

print(f"Client ID: {client_id}")
print(f"SDK Token starts with: {sdk_token[:15]}...")

fyers = fyersModel.FyersModel(
    client_id=client_id,
    is_async=False,
    token=sdk_token,
    log_path="logs/"
)

# Fetch Nifty 50 history for today
today_str = datetime.now().strftime("%Y-%m-%d")
payload = {
    "symbol": "NSE:NIFTY50-INDEX",
    "resolution": "15",
    "date_format": "1",
    "range_from": today_str,
    "range_to": today_str,
    "cont_flag": "1",
}

print("Sending history request payload:", payload)
response = fyers.history(data=payload)
print("Raw Response:", response)
