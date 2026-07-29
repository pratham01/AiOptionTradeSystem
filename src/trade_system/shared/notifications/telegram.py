import os
import re
import logging
import requests
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def _convert_markdown_to_safe_html(self, text: str) -> str:
        """Converts basic markdown (code blocks, bold) to safe HTML for Telegram."""
        if not text:
            return ""
        
        import html
        # 1. Escape entire text to make HTML special characters safe (e.g. & to &amp;, < to &lt;, > to &gt;)
        escaped = html.escape(text)
        
        # 2. Split by triple backticks to identify preformatted code blocks
        parts = escaped.split("```")
        result = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                # Inside code block: wrap in <pre>...</pre>
                code_content = part
                # Strip out any markdown language tags at the start of the code block
                if code_content.startswith("\n"):
                    code_content = code_content[1:]
                else:
                    lines = code_content.split("\n")
                    if lines and (lines[0].strip().lower() in ["python", "bash", "json", "html", "csv", "markdown"]):
                        code_content = "\n".join(lines[1:])
                
                result.append(f"<pre>{code_content}</pre>")
            else:
                # Outside code block: process bold (**text**), code (`text`)
                part_formatted = part
                
                # Convert double asterisks **bold** to <b>bold</b>
                bold_parts = part_formatted.split("**")
                bold_result = []
                for b_idx, b_part in enumerate(bold_parts):
                    if b_idx % 2 == 1:
                        bold_result.append(f"<b>{b_part}</b>")
                    else:
                        bold_result.append(b_part)
                part_formatted = "".join(bold_result)
                
                # Convert single backticks `code` to <code>code</code>
                code_parts = part_formatted.split("`")
                code_result = []
                for c_idx, c_part in enumerate(code_parts):
                    if c_idx % 2 == 1:
                        code_result.append(f"<code>{c_part}</code>")
                    else:
                        code_result.append(c_part)
                part_formatted = "".join(code_result)
                
                result.append(part_formatted)
                
        return "".join(result)

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        # Load settings dynamically to check for fallback config
        from trade_system.shared.config import Settings
        try:
            settings = Settings.load()
            whatsapp_enabled = settings.whatsapp.enabled
            whatsapp_token = settings.whatsapp.access_token
            whatsapp_phone_id = settings.whatsapp.phone_number_id
            whatsapp_to = settings.whatsapp.to_number
        except Exception as e:
            logger.warning(f"Could not load settings for WhatsApp fallback check: {e}")
            whatsapp_enabled = False
            whatsapp_token = ""
            whatsapp_phone_id = ""
            whatsapp_to = ""

        # Safe Markdown to HTML conversion to prevent 400 Bad Request
        if parse_mode == "Markdown":
            text = self._convert_markdown_to_safe_html(text)
            parse_mode = "HTML"

        telegram_ok = False
        telegram_err = None

        if self.token and self.chat_id:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            try:
                response = requests.post(self.base_url, json=payload, timeout=10)
                response.raise_for_status()
                telegram_ok = True
            except Exception as e:
                telegram_err = str(e)
                logger.error(f"Failed to send Telegram message: {e}")
        else:
            telegram_err = "Telegram not configured (missing token/chat_id)"
            logger.warning(telegram_err)

        if telegram_ok:
            return True

        # --- FALLBACK PATHWAY ---
        # 1. Log the message to a backup file
        fallback_log_path = Path("logs/notifications_fallback.log")
        try:
            fallback_log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"========================================================================\n"
                f"TIMESTAMP: {timestamp}\n"
                f"TELEGRAM ERROR: {telegram_err}\n"
                f"MESSAGE:\n{text}\n"
                f"========================================================================\n\n"
            )
            with open(fallback_log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
            logger.info(f"Notification fallback logged to {fallback_log_path}")
        except Exception as log_ex:
            logger.error(f"Failed to write fallback log: {log_ex}")

        # 2. Send via WhatsApp if configured and enabled
        if whatsapp_enabled and whatsapp_token and whatsapp_phone_id and whatsapp_to:
            logger.info("Attempting fallback message to WhatsApp (Meta Cloud API)...")
            whatsapp_text = self._convert_html_to_whatsapp_markdown(text)
            
            url = f"https://graph.facebook.com/v20.0/{whatsapp_phone_id}/messages"
            headers = {
                "Authorization": f"Bearer {whatsapp_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": whatsapp_to,
                "type": "text",
                "text": {
                    "preview_url": False,
                    "body": whatsapp_text
                }
            }
            
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=15)
                res.raise_for_status()
                logger.info("Successfully sent fallback WhatsApp notification.")
                return True
            except Exception as wa_ex:
                logger.error(f"Failed to send WhatsApp fallback notification: {wa_ex}")
                # Log the WhatsApp failure as well
                try:
                    with open(fallback_log_path, "a", encoding="utf-8") as f:
                        f.write(f"WHATSAPP FALLBACK ERROR: {wa_ex}\n\n")
                except:
                    pass
        else:
            logger.debug("WhatsApp fallback is not enabled or not fully configured.")

        return False

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        return self.send_message(text, parse_mode=parse_mode)

    def _convert_html_to_whatsapp_markdown(self, html: str) -> str:
        """Converts basic HTML formatting into WhatsApp markdown."""
        if not html:
            return ""
        
        # Replace bold tags
        text = re.sub(r'<(?:b|strong)>(.*?)</(?:b|strong)>', r'*\1*', html, flags=re.IGNORECASE)
        # Replace italic tags
        text = re.sub(r'<(?:i|em)>(.*?)</(?:i|em)>', r'_\1_', text, flags=re.IGNORECASE)
        # Replace code tags
        text = re.sub(r'<(?:code)>(.*?)</(?:code)>', r'`\1`', text, flags=re.IGNORECASE)
        # Replace line breaks
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        # Strip all other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Unescape common HTML entities
        text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        return text

