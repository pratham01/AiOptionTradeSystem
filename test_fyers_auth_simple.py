import os
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()

client_id = os.getenv("FYERS_CLIENT_ID")
env_token = os.getenv("FYERS_ACCESS_TOKEN")

# For REST SDK, we MUST pass only the JWT part. 
# The SDK uses the client_id parameter to construct the appid:token header.
if env_token and ":" in env_token:
    access_token = env_token.split(":")[1]
else:
    access_token = env_token

print(f"Client ID: {client_id}")
print(f"Token (start): {access_token[:20]}...")

fyers = fyersModel.FyersModel(
    client_id=client_id,
    is_async=False,
    token=access_token,
    log_path="."
)

profile = fyers.get_profile()
print(f"Profile Response: {profile}")
