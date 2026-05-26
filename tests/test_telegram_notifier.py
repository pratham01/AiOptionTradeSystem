from trade_system.infrastructure.notifications.telegram import TelegramNotifier

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

    def fake_post(_url, json, timeout):
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

    def fake_post(_url, json, timeout):
        return _DummyResponse(400, "Bad Request")

    monkeypatch.setattr("trade_system.infrastructure.notifications.telegram.requests.post", fake_post)
    assert notifier.send("test alert") is False
