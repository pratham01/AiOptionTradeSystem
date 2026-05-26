import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured. Skipping notification.")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        return self.send_message(text, parse_mode=parse_mode)
