"""
Unit tests to verify that Intraday Option Telegram alerts are suppressed
when enable_intraday_option_alerts is False, and configurable via settings.
"""
from unittest.mock import MagicMock
from trade_system.shared.config import Settings, TelegramConfig
from trade_system.domains.advisory.application.agent.chief_trading_agent import (
    ChiefTradingAgent,
    ChiefTradeSignal,
)


def test_settings_default_intraday_option_alerts_is_false():
    s = Settings()
    assert s.enable_intraday_option_alerts is False


def test_chief_agent_suppresses_option_alert_when_disabled():
    settings = Settings(
        enable_intraday_option_alerts=False,
        telegram=TelegramConfig(bot_token="test_token", chat_id="test_chat", enabled=True),
        st_confirmed_telegram=TelegramConfig(bot_token="test_token", chat_id="test_chat", enabled=True),
    )
    agent = ChiefTradingAgent(settings=settings)
    mock_notifier = MagicMock()
    agent.notifier = mock_notifier

    sig = ChiefTradeSignal(
        signal_id="SIG_001",
        timestamp="2026-10-01 10:00:00",
        symbol="NSE:NIFTY50-INDEX",
        asset_type="INDEX",
        direction="BUY_CALL",
        strike_name="NIFTY 22000 CE",
        option_type="CE",
        spot_entry_trigger=22000.0,
        spot_stop_loss=21950.0,
        target_1=22050.0,
        target_2=22100.0,
        risk_reward="1:2.0",
        confluence_score=85,
        pillar_scores={},
        primary_thesis="Test thesis",
        veto_passed=True,
    )

    sent = agent.dispatch_telegram_alert(sig)
    assert sent is False
    mock_notifier.send.assert_not_called()


def test_option_edge_suppresses_alert_when_disabled():
    from trade_system.interfaces.live.collector import LiveMarketDataService
    
    settings = Settings(
        enable_intraday_option_alerts=False,
        telegram=TelegramConfig(bot_token="test_token", chat_id="test_chat", enabled=True),
    )
    mock_broker = MagicMock()
    mock_catalog = MagicMock()
    service = LiveMarketDataService(
        broker=mock_broker,
        broker_manager=mock_broker,
        catalog=mock_catalog,
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.notifier = MagicMock()
    
    mock_alert = MagicMock()
    mock_alert.symbol = "NSE:NIFTY50-INDEX"
    mock_alert.format_telegram.return_value = "Test Option Alert"

    service._send_option_edge_alerts([mock_alert])
    service.notifier.send.assert_not_called()
