import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

print(f"Telegram Bot Token: {token[:15]}...{token[-5:] if token else ''}")
print(f"Telegram Chat ID: {chat_id}")

url = f"https://api.telegram.org/bot{token}/sendMessage"
payload = {
    "chat_id": chat_id,
    "text": "🤖 Antigravity Test Message: Checking system-wide Telegram notifications.",
    "parse_mode": "HTML"
}

try:
    print("Sending post request to Telegram...")
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Exception raised: {e}")
