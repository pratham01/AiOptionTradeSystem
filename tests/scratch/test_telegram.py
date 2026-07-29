import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import logging
logging.basicConfig(level=logging.INFO)

from trade_system.shared.config import Settings
from trade_system.shared.notifications.telegram import TelegramNotifier

def test_telegram():
    settings = Settings.load()
    print("TELEGRAM_BOT_TOKEN:", settings.telegram.bot_token)
    print("TELEGRAM_CHAT_ID:", settings.telegram.chat_id)
    
    notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
    print("Sending test message to default notifier...")
    res = notifier.send("🔔 <b>Telegram Notifier Test</b>\nThis is a test from the diagnostic script.")
    print("Result:", res)
    
    print("\nST_CONFIRMED_TELEGRAM_BOT_TOKEN:", settings.st_confirmed_telegram.bot_token)
    print("ST_CONFIRMED_TELEGRAM_CHAT_ID:", settings.st_confirmed_telegram.chat_id)
    
    confirmed_notifier = TelegramNotifier(
        settings.st_confirmed_telegram.bot_token or settings.telegram.bot_token,
        settings.st_confirmed_telegram.chat_id or settings.telegram.chat_id
    )
    print("Sending test message to confirmed notifier...")
    res_conf = confirmed_notifier.send("🟢 <b>Confirmed Notifier Test</b>\nThis is a test from the diagnostic script.")
    print("Result:", res_conf)

if __name__ == "__main__":
    test_telegram()
