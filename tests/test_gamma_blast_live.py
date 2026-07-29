from datetime import datetime, date, time as dt_time, timedelta
from unittest.mock import MagicMock
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from trade_system.shared.config import Settings, TelegramConfig, FyersConfig
from trade_system.interfaces.live.collector import LiveMarketDataService
from trade_system.domains.advisory.application.agent.gamma_blast_agent import GammaContext
from trade_system.shared import TradeSuggestion, TradeDirection, TradeHorizon, OptionParams

IST = ZoneInfo("Asia/Kolkata")

class _DummyBroker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.quote_payload: dict[str, dict] = {}

    def get_quotes(self, symbols: list[str]):
        return {symbol: self.quote_payload.get(symbol, {}) for symbol in symbols}

    def fetch_history(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def get_historical_data(self, **kwargs) -> list:
        return []

class _DummyCatalog:
    def live_bars_path(self, *_args, **_kwargs):
        pass
    def append_frame(self, *_args, **_kwargs):
        pass

def test_evaluate_gamma_blast_triggers_and_opens_position():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    broker = _DummyBroker(settings)
    service = LiveMarketDataService(
        broker=broker, broker_manager=broker,
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    
    # Mock time to be inside trade window (14:45 PM)
    now = datetime(2026, 4, 15, 14, 45, tzinfo=IST) # Wed
    service._now_ist = MagicMock(return_value=now)
    
    timestamps = pd.date_range(end=now.replace(tzinfo=None), periods=50, freq="1min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 22000.0,
        "high": 22010.0,
        "low": 21990.0,
        "close": 22005.0,
        "volume": 1000.0
    }).set_index("timestamp")
    
    service.minute_data["NSE:NIFTY50-INDEX"] = df
    
    mock_sugg = TradeSuggestion(
        id="test",
        symbol="NSE:NIFTY50-INDEX",
        timestamp=now,
        direction=TradeDirection.CALL,
        horizon=TradeHorizon.INTRADAY,
        entry_zone_low=22000.0,
        entry_zone_high=22010.0,
        target=22050.0,
        stop_loss=21950.0,
        option_params=OptionParams(direction=TradeDirection.CALL, expiry_type="weekly"),
        confidence=0.9,
        setup_features=None,
        narrative="Test breakout"
    )
    service.gamma_agent.analyze_for_gamma_blast = MagicMock(return_value=mock_sugg)
    
    service._evaluate_gamma_blast("NSE:NIFTY50-INDEX")
    
    assert service.gamma_open_position["NSE:NIFTY50-INDEX"] is not None
    trade = service.gamma_open_position["NSE:NIFTY50-INDEX"]
    assert trade["direction"] == TradeDirection.CALL
    assert trade["target_spot"] == 22050.0
    assert trade["stop_spot"] == 21950.0

def test_manage_gamma_blast_exits_on_target_spot():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    broker = _DummyBroker(settings)
    service = LiveMarketDataService(
        broker=broker, broker_manager=broker,
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    
    now = datetime(2026, 4, 15, 14, 50, tzinfo=IST)
    service._now_ist = MagicMock(return_value=now)
    
    trade = {
        "symbol": "NSE:NIFTY50-INDEX",
        "direction": TradeDirection.CALL,
        "entry_time": now - timedelta(minutes=5),
        "entry_spot": 22000.0,
        "option_symbol": "N/A",
        "entry_premium": 0.0,
        "target_spot": 22050.0,
        "stop_spot": 21950.0,
        "reason": "Test"
    }
    service.gamma_open_position["NSE:NIFTY50-INDEX"] = trade
    
    # Mock price hitting target
    timestamps = pd.date_range(end=now.replace(tzinfo=None), periods=5, freq="1min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 22045.0,
        "high": 22055.0,
        "low": 22040.0,
        "close": 22055.0,  # Hitting target!
        "volume": 1000.0
    }).set_index("timestamp")
    service.minute_data["NSE:NIFTY50-INDEX"] = df
    
    service._evaluate_gamma_blast("NSE:NIFTY50-INDEX")
    
    assert service.gamma_open_position["NSE:NIFTY50-INDEX"] is None
    assert len(service.gamma_trades["NSE:NIFTY50-INDEX"]) == 1
    assert service.gamma_trades["NSE:NIFTY50-INDEX"][0]["exit_reason"] == "TARGET_HIT"
