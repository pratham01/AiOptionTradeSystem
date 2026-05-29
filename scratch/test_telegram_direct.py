import os
import requests
from dotenv import load_dotenv

def send_telegram_msg(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    print(f"Sending to chat {chat_id} using token {token[:10]}...")
    try:
        r = requests.post(url, json=payload, timeout=10)
        print("Status Code:", r.status_code)
        print("Response:", r.text)
        return r.status_code == 200
    except Exception as e:
        print("Error sending message:", e)
        return False

def main():
    load_dotenv(".env", override=True)
    
    # 1. Main Telegram Bot
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    print("--- Test Main Bot ---")
    if bot_token and chat_id:
        send_telegram_msg(bot_token, chat_id, "🤖 <b>Test message from Trade System V2</b>: Main Bot check.")
    else:
        print("Main Bot credentials missing in .env")
        
    # 2. Confirmed ST Telegram Bot
    st_token = os.getenv("ST_CONFIRMED_TELEGRAM_BOT_TOKEN")
    st_chat_id = os.getenv("ST_CONFIRMED_TELEGRAM_CHAT_ID")
    print("\n--- Test Confirmed Bot ---")
    if st_token and st_chat_id:
        send_telegram_msg(st_token, st_chat_id, "🤖 <b>Test message from Trade System V2</b>: Confirmed Option Buy Bot check.")
    else:
        print("Confirmed Bot credentials missing in .env")

if __name__ == "__main__":
    main()
