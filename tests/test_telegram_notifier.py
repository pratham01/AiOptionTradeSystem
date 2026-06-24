from pathlib import Path
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.config import Settings, WhatsappConfig

class _DummyResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")

def test_telegram_notifier_sends_message_successfully(monkeypatch):
    notifier = TelegramNotifier(token="test-token", chat_id="test-chat")
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append(json)
        return _DummyResponse(200, "ok")

    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.requests.post", fake_post)

    assert notifier.send("test <b>alert</b>") is True
    assert calls[0]["parse_mode"] == "HTML"
    assert calls[0]["text"] == "test <b>alert</b>"
    assert calls[0]["chat_id"] == "test-chat"

def test_telegram_notifier_handles_missing_credentials():
    notifier = TelegramNotifier(token="", chat_id="")
    assert notifier.send("test alert") is False

def test_telegram_notifier_handles_network_error(monkeypatch):
    notifier = TelegramNotifier(token="test-token", chat_id="test-chat")

    def fake_post(url, json=None, headers=None, timeout=10):
        return _DummyResponse(400, "Bad Request")

    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.requests.post", fake_post)
    assert notifier.send("test alert") is False

def test_telegram_notifier_fallback_logging_and_whatsapp_success(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    fallback_log_file = log_dir / "notifications_fallback.log"
    
    # Mock Settings to enable WhatsApp
    class MockSettings:
        whatsapp = WhatsappConfig(
            phone_number_id="12345",
            access_token="wa-token",
            to_number="919876543210",
            enabled=True
        )
    
    monkeypatch.setattr("trade_system.config.Settings.load", lambda *args: MockSettings())
    # Override Path in telegram.py to point to our temp log file
    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.Path", lambda p: fallback_log_file if "fallback" in str(p) else Path(p))
    
    notifier = TelegramNotifier(token="test-token", chat_id="test-chat")
    
    # Mock requests.post to fail on Telegram and succeed on WhatsApp
    post_calls = []
    def fake_post(url, json=None, headers=None, timeout=10):
        post_calls.append((url, json, headers))
        if "api.telegram.org" in url:
            return _DummyResponse(500, "Error")
        elif "graph.facebook.com" in url:
            return _DummyResponse(200, "ok")
        return _DummyResponse(404, "Not Found")
        
    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.requests.post", fake_post)
    
    # Send a message with HTML formatting
    assert notifier.send("Hello <b>world</b>! <br> Code: <code>print(123)</code>") is True
    
    # Check that it logged to file
    assert fallback_log_file.exists()
    log_content = fallback_log_file.read_text()
    assert "Hello <b>world</b>!" in log_content
    assert "TELEGRAM ERROR" in log_content
    
    # Check WhatsApp API call
    assert len(post_calls) == 2
    wa_url, wa_json, wa_headers = post_calls[1]
    assert "graph.facebook.com/v20.0/12345/messages" in wa_url
    assert wa_headers["Authorization"] == "Bearer wa-token"
    # Verify HTML to WhatsApp markdown conversion
    assert wa_json["text"]["body"] == "Hello *world*! \n Code: `print(123)`"
    assert wa_json["to"] == "919876543210"

def test_telegram_notifier_fallback_when_whatsapp_disabled(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    fallback_log_file = log_dir / "notifications_fallback.log"
    
    class MockSettings:
        whatsapp = WhatsappConfig(enabled=False)
        
    monkeypatch.setattr("trade_system.config.Settings.load", lambda *args: MockSettings())
    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.Path", lambda p: fallback_log_file if "fallback" in str(p) else Path(p))
    
    notifier = TelegramNotifier(token="test-token", chat_id="test-chat")
    
    def fake_post(url, json=None, headers=None, timeout=10):
        if "api.telegram.org" in url:
            return _DummyResponse(500, "Error")
        return _DummyResponse(404, "Not Found")
        
    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.requests.post", fake_post)
    
    assert notifier.send("Telegram down message") is False
    
    # Should log to file
    assert fallback_log_file.exists()
    log_content = fallback_log_file.read_text()
    assert "Telegram down message" in log_content


def test_telegram_notifier_converts_markdown_to_html(monkeypatch):
    notifier = TelegramNotifier(token="test-token", chat_id="test-chat")
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append(json)
        return _DummyResponse(200, "ok")

    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.requests.post", fake_post)

    # Message with bold, inline code, and pre-formatted block
    markdown_text = (
        "Hello **world**!\n"
        "Here is a table:\n"
        "```python\n"
        "import sys\n"
        "```\n"
        "And `code`."
    )
    assert notifier.send(markdown_text, parse_mode="Markdown") is True
    assert len(calls) == 1
    assert calls[0]["parse_mode"] == "HTML"
    
    expected_html = (
        "Hello <b>world</b>!\n"
        "Here is a table:\n"
        "<pre>import sys\n</pre>\n"
        "And <code>code</code>."
    )
    assert calls[0]["text"] == expected_html


