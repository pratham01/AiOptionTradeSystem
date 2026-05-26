from types import SimpleNamespace

import pandas as pd

from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers import _parse_nse_market_status
from trade_system.config import Settings
from trade_system.config.settings import TelegramConfig, FyersConfig
from trade_system.interfaces.live.collector import LiveMarketDataService


def test_parse_nse_market_status_accepts_string_exchange_and_segment_codes():
    is_open, detail = _parse_nse_market_status(
        [
            {"exchange": "10", "segment": "10", "status": "OPEN"},
        ]
    )

    assert is_open is True
    assert detail == "OPEN"


def test_parse_nse_market_status_returns_closed_status_for_cash_segment():
    is_open, detail = _parse_nse_market_status(
        [
            {"exchange": 10, "segment": "10", "status": "CLOSED"},
        ]
    )

    assert is_open is False
    assert detail == "CLOSED"


class _DummyBroker:
    def __init__(self, settings: Settings):
        self.settings = settings


class _DummyCatalog:
    pass


def test_collector_converts_epoch_ticks_to_ist_session_time():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )

    ts = service._from_epoch_ist(pd.Timestamp("2026-04-13 09:15:00", tz="Asia/Kolkata").timestamp())

    assert ts == pd.Timestamp("2026-04-13 09:15:00")


def test_refresh_token_via_totp_rebuilds_broker_with_named_credentials(monkeypatch, tmp_path):
    settings = Settings(
        fyers=FyersConfig(client_id="CID-100", user_id="USER1", access_token="old-token"),
        telegram=TelegramConfig(bot_token="token", chat_id="chat"),
    )
    service = LiveMarketDataService(
        broker=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_path = scripts_dir / "authenticate_fyers_totp.py"
    script_path.write_text("print('ok')\n")
    new_settings = Settings(
        fyers=FyersConfig(client_id="CID-100", user_id="USER1", access_token="new-token"),
        telegram=TelegramConfig(bot_token="token", chat_id="chat"),
    )
    captured: dict[str, str] = {}

    monkeypatch.setattr("trade_system.interfaces.live.collector.Path.resolve", lambda self: tmp_path / "src" / "trade_system" / "interfaces" / "live" / "collector.py")
    monkeypatch.setattr("trade_system.interfaces.live.collector.subprocess.run", lambda *args, **kwargs: None)
    monkeypatch.setattr("trade_system.interfaces.live.collector.Settings.load", lambda: new_settings)

    class _ReplacementBroker:
        def __init__(self, *, client_id: str, access_token: str, user_id: str | None = None, **_kwargs):
            captured["client_id"] = client_id
            captured["access_token"] = access_token
            captured["user_id"] = user_id or ""

    monkeypatch.setattr("trade_system.interfaces.live.collector.FyersBrokerClient", _ReplacementBroker)

    assert service._refresh_token_via_totp() is True
    assert captured == {
        "client_id": "CID-100",
        "access_token": "new-token",
        "user_id": "USER1",
    }
