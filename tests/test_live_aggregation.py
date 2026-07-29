from trade_system.shared.config.settings import TelegramConfig, FyersConfig
import pandas as pd

from trade_system.shared.config import Settings
from trade_system.domains.trading.domain.ports.broker import HistoricalData
from trade_system.domains.market_data.infrastructure.data.storage import CsvDataCatalog
from trade_system.interfaces.live.collector import LiveMarketDataService, _merge_intraday_3min_bars
from trade_system.interfaces.live.collector import IctLiveTrade
from trade_system.interfaces.live.confirmed_strategy import (
    confirmed_entry_payload,
    confirmed_exit_payload,
    prepare_confirmed_strategy_frame,
)
from trade_system.interfaces.live.helpers import (
    _completed_timeframe_bars,
    _is_market_timestamp,
    _sanitize_intraday_minutes,
    aggregate_ticks_to_bars,
    create_minute_bar,
    detect_smc_signal,
)


def test_aggregate_ticks_to_bars():
    df = pd.DataFrame(
        [
            {"symbol": "NSE:NIFTY50-INDEX", "timestamp": "2026-03-20 09:15:01", "ltp": 100.0, "volume": 10},
            {"symbol": "NSE:NIFTY50-INDEX", "timestamp": "2026-03-20 09:15:20", "ltp": 105.0, "volume": 20},
            {"symbol": "NSE:NIFTY50-INDEX", "timestamp": "2026-03-20 09:15:45", "ltp": 102.0, "volume": 30},
        ]
    )

    bars = aggregate_ticks_to_bars(df, timeframe_minutes=1)

    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 105.0
    assert row["low"] == 100.0
    assert row["close"] == 102.0
    assert row["volume"] == 60


def test_create_minute_bar_uses_cumulative_volume_delta():
    minute_bar = create_minute_bar(
        "NSE:NIFTY50-INDEX",
        [
            {"timestamp": pd.Timestamp("2026-03-20 09:15:01"), "ltp": 100.0, "volume": 1010},
            {"timestamp": pd.Timestamp("2026-03-20 09:15:20"), "ltp": 105.0, "volume": 1022},
            {"timestamp": pd.Timestamp("2026-03-20 09:15:45"), "ltp": 102.0, "volume": 1030},
        ],
        previous_cumulative_volume=1000,
    )

    assert minute_bar is not None
    assert minute_bar["volume"] == 30


def test_market_timestamp_excludes_1530_and_later():
    assert _is_market_timestamp(pd.Timestamp("2026-03-25 09:15:00"), pd.Timestamp("2026-03-25 09:15:00").time(), pd.Timestamp("2026-03-25 15:30:00").time())
    assert not _is_market_timestamp(pd.Timestamp("2026-03-25 15:30:00"), pd.Timestamp("2026-03-25 09:15:00").time(), pd.Timestamp("2026-03-25 15:30:00").time())
    assert not _is_market_timestamp(pd.Timestamp("2026-03-25 16:00:00"), pd.Timestamp("2026-03-25 09:15:00").time(), pd.Timestamp("2026-03-25 15:30:00").time())


def test_sanitize_intraday_minutes_drops_duplicates_and_after_hours():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-25 09:15:00",
                    "2026-03-25 09:15:00",
                    "2026-03-25 15:29:00",
                    "2026-03-25 15:30:00",
                    "2026-03-25 16:03:00",
                ]
            ),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 20, 30, 40, 50],
        }
    ).set_index("timestamp")

    cleaned = _sanitize_intraday_minutes(
        frame,
        pd.Timestamp("2026-03-25 09:15:00").time(),
        pd.Timestamp("2026-03-25 15:30:00").time(),
    )

    assert list(cleaned.index) == [
        pd.Timestamp("2026-03-25 09:15:00"),
        pd.Timestamp("2026-03-25 15:29:00"),
    ]
    assert cleaned.iloc[0]["open"] == 2


def test_completed_timeframe_bars_excludes_open_bar():
    timeframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-25 14:21:00",
                    "2026-03-25 14:24:00",
                    "2026-03-25 14:27:00",
                ]
            ),
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [10, 20, 30],
        }
    ).set_index("timestamp")

    completed = _completed_timeframe_bars(timeframe, 3, pd.Timestamp("2026-03-25 14:28:00"))

    assert list(completed.index) == [
        pd.Timestamp("2026-03-25 14:21:00"),
        pd.Timestamp("2026-03-25 14:24:00"),
    ]


def test_detect_smc_signal_bullish_sweep_reclaim():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-20 09:15:00", periods=8, freq="3min"),
            "open": [100, 101, 102, 103, 104, 105, 105, 101],
            "high": [102, 103, 104, 105, 106, 107, 106, 109],
            "low": [99, 100, 101, 102, 103, 104, 105, 98],
            "close": [101, 102, 103, 104, 105, 106, 106, 108],
            "volume": [10] * 8,
        }
    ).set_index("timestamp")

    signal = detect_smc_signal(frame)

    assert signal is not None
    assert signal["direction"] == 1
    assert signal["signal_name"] == "BULLISH_SWEEP_RECLAIM"


class _DummyBroker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.quote_payload: dict[str, dict] = {}

    def get_quotes(self, symbols: list[str]):
        return {symbol: self.quote_payload.get(symbol, {}) for symbol in symbols}

    def fetch_history(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def get_historical_data(self, **kwargs) -> list[HistoricalData]:
        df = self.fetch_history(**kwargs)
        if df.empty:
            return []
        
        records = []
        for _, row in df.iterrows():
            records.append(
                HistoricalData(
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0),
                )
            )
        return records


class _DummyCatalog:
    def live_bars_path(self, *_args, **_kwargs):
        raise AssertionError("live_bars_path should not be called in this test")


def test_premarket_console_summary_reports_gap_level_profile_and_mood():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    broker = _DummyBroker(settings)
    broker.quote_payload["NSE:NIFTY50-INDEX"] = {"lp": 22535.0}
    service = LiveMarketDataService(
        broker=broker, broker_manager=broker,
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.previous_day_levels["NSE:NIFTY50-INDEX"] = {
        "open": 22440.0,
        "high": 22530.0,
        "low": 22380.0,
        "close": 22420.0,
    }
    service.minute_data["NSE:NIFTY50-INDEX"] = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-04-10 09:15:00",
                    "2026-04-10 09:16:00",
                    "2026-04-10 09:17:00",
                    "2026-04-10 09:18:00",
                ]
            ),
            "open": [22470.0, 22480.0, 22490.0, 22500.0],
            "high": [22480.0, 22490.0, 22500.0, 22510.0],
            "low": [22460.0, 22470.0, 22480.0, 22490.0],
            "close": [22480.0, 22490.0, 22500.0, 22510.0],
            "volume": [100.0, 300.0, 700.0, 200.0],
        }
    ).set_index("timestamp")

    summary = service._build_premarket_console_summary(
        "NSE:NIFTY50-INDEX",
        {"direction": 1, "supertrend": 22460.0},
    )

    assert "Premarket context | NIFTY50 | 09:09" in summary
    assert "Formation: Gap Up" in summary
    assert "Near previous day HIGH" in summary
    assert "Volume profile: POC" in summary
    assert "Market mood: Bullish auction bias" in summary


def test_fetch_recent_daily_context_keeps_last_completed_day_when_today_not_present():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.broker_manager.fetch_history = lambda **_kwargs: pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-22", "2026-04-23"]),
            "open": [24000.0, 24100.0],
            "high": [24150.0, 24200.0],
            "low": [23950.0, 24050.0],
            "close": [24100.0, 24180.0],
            "volume": [1.0, 1.0],
        }
    )

    recent = service._fetch_recent_daily_context("NSE:NIFTY50-INDEX")

    assert recent[-1]["date"] == "2026-04-23"
    assert recent[-1]["close"] == 24180.0


def test_fetch_recent_daily_context_excludes_current_day_partial_bar():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.broker_manager.fetch_history = lambda **_kwargs: pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-23", "2026-04-24"]),
            "open": [24100.0, 24200.0],
            "high": [24200.0, 24250.0],
            "low": [24050.0, 24150.0],
            "close": [24180.0, 24210.0],
            "volume": [1.0, 1.0],
        }
    )
    service._today_ist = lambda: pd.to_datetime("2026-04-24").date()

    recent = service._fetch_recent_daily_context("NSE:NIFTY50-INDEX")

    assert len(recent) == 1
    assert recent[0]["date"] == "2026-04-23"
    assert recent[0]["close"] == 24180.0


def test_previous_day_supertrend_message_includes_opening_reference_break_status():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    broker = _DummyBroker(settings)
    broker.quote_payload["NSE:NIFTY50-INDEX"] = {"lp": 22440.0}
    service = LiveMarketDataService(
        broker=broker, broker_manager=broker,
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.previous_day_levels["NSE:NIFTY50-INDEX"] = {
        "open": 22410.0,
        "high": 22530.0,
        "low": 22380.0,
        "close": 22490.0,
    }

    message = service._build_previous_day_supertrend_message(
        "NSE:NIFTY50-INDEX",
        {"direction": 1, "supertrend": 22460.0},
    )

    assert "Opening reference price (09:09): ₹22440.00" in message
    assert "Previous day OHLC: O ₹22410.00 | H ₹22530.00 | L ₹22380.00 | C ₹22490.00" in message
    assert "Opening break below previous-day Supertrend" in message


def test_previous_day_supertrend_message_reports_hold_when_opening_price_respects_trend():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    broker = _DummyBroker(settings)
    broker.quote_payload["NSE:NIFTY50-INDEX"] = {"lp": 22480.0}
    service = LiveMarketDataService(
        broker=broker, broker_manager=broker,
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.previous_day_levels["NSE:NIFTY50-INDEX"] = {
        "open": 22410.0,
        "high": 22530.0,
        "low": 22380.0,
        "close": 22490.0,
    }

    message = service._build_previous_day_supertrend_message(
        "NSE:NIFTY50-INDEX",
        {"direction": 1, "supertrend": 22460.0},
    )

    assert "Opening price holding above previous-day Supertrend" in message


def test_supertrend_alert_uses_current_session_bar_transition():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    sent_messages: list[str] = []
    service.notifier.send = sent_messages.append
    service.last_trend["NSE:NIFTY50-INDEX"] = -1
    service._today_ist = lambda: pd.to_datetime("2026-03-24").date()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-24 09:33:00", "2026-03-24 09:36:00"]),
            "close": [22818.95, 22783.55],
            "supertrend": [22790.00, 22983.95],
            "supertrend_direction": [1, -1],
        }
    ).set_index("timestamp")

    service._check_trend_change("NSE:NIFTY50-INDEX", df)

    assert len(sent_messages) == 1
    assert "Direction changed to <b>DOWN</b>" in sent_messages[0]
    assert "09:36" in sent_messages[0]


def test_supertrend_touch_alert_fires_when_candle_hits_line():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    sent_messages: list[str] = []
    service.notifier.send = sent_messages.append

    last_bar = pd.Series(
        {
            "high": 22910.0,
            "low": 22880.0,
            "close": 22895.0,
            "supertrend": 22900.0,
            "supertrend_direction": 1,
        }
    )

    service._check_supertrend_touch("NSE:NIFTY50-INDEX", last_bar, pd.Timestamp("2026-03-24 09:39:00"))

    assert len(sent_messages) == 1
    assert "Level Retest" in sent_messages[0]
    assert "Supertrend" in sent_messages[0]
    


def test_confirmed_entry_payload_detects_15m_confirmed_flip():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-24 09:30:00", "2026-03-24 09:45:00"]),
            "open": [22820.0, 22810.0],
            "high": [22840.0, 22815.0],
            "low": [22800.0, 22710.0],
            "close": [22830.0, 22720.0],
            "supertrend": [22750.0, 22920.0],
            "supertrend_direction": [1, -1],
        }
    ).set_index("timestamp")
    prepared = prepare_confirmed_strategy_frame(frame)

    payload = confirmed_entry_payload(prepared)

    assert payload is not None
    assert payload["direction"] == -1
    assert payload["bar_time"] == pd.Timestamp("2026-03-24 09:45:00")


def test_ict_eod_lines_summarize_experimental_trade_stats():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"),
        enable_experimental_ict_stream=True,
    )
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.ict_trades["NSE:NIFTY50-INDEX"] = [
        IctLiveTrade(
            direction=1,
            entry_time=pd.Timestamp("2026-04-15 10:00:00"),
            entry_price=22500.0,
            stop_loss=22480.0,
            take_profit=22540.0,
            exit_time=pd.Timestamp("2026-04-15 10:18:00"),
            exit_price=22540.0,
            exit_reason="TARGET_HIT",
            trigger="SSL_SWEEP",
            confluence_score=3,
            points_captured=40.0,
            risk_points=20.0,
            r_multiple=2.0,
        ),
        IctLiveTrade(
            direction=-1,
            entry_time=pd.Timestamp("2026-04-15 11:00:00"),
            entry_price=22490.0,
            stop_loss=22510.0,
            take_profit=22450.0,
            exit_time=pd.Timestamp("2026-04-15 11:24:00"),
            exit_price=22510.0,
            exit_reason="SL_HIT",
            trigger="BSL_SWEEP",
            confluence_score=2,
            points_captured=-20.0,
            risk_points=20.0,
            r_multiple=-1.0,
        ),
    ]

    lines = service._build_ict_eod_lines("NSE:NIFTY50-INDEX")
    text = "\n".join(lines)

    assert "ICT experimental" in text
    assert "Trades: 2 | Wins: 1 | Losses: 1" in text
    assert "Net points: +20.00 | Avg R: +0.50" in text
    assert "Target hits: 1 | SL hits: 1" in text


def test_confirmed_exit_payload_detects_sl_hit():
    row = pd.Series(
        {
            "high": 22980.0,
            "low": 22880.0,
            "close": 22910.0,
            "supertrend": 22950.0,
            "supertrend_direction": -1,
        }
    )

    payload = confirmed_exit_payload(
        {"direction": -1, "entry_price": 22800.0},
        row,
        pd.Timestamp("2026-03-24 10:15:00"),
    )

    assert payload is not None
    assert payload["exit_reason"] == "SL_HIT"
    assert payload["exit_price"] == 22950.0


def test_merge_intraday_3min_bars_keeps_pure_ohlcv():
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-24 15:21:00",
                    "2026-03-24 15:24:00",
                    "2026-03-24 15:27:00",
                ]
            ),
            "open": [22492.75, 22486.30, 22489.95],
            "high": [22493.25, 22497.35, 22505.35],
            "low": [22479.55, 22480.05, 22478.10],
            "close": [22486.75, 22491.25, 22492.65],
            "volume": [8175771, 8585576, 6778577],
            "supertrend": [1.0, 2.0, 3.0],
            "supertrend_direction": [1.0, 1.0, -1.0],
        }
    )
    today_bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-25 09:15:00",
                    "2026-03-25 09:18:00",
                    "2026-03-25 09:21:00",
                ]
            ),
            "open": [22900.0, 22910.0, 22890.0],
            "high": [22920.0, 22915.0, 22905.0],
            "low": [22895.0, 22880.0, 22870.0],
            "close": [22910.0, 22890.0, 22875.0],
            "volume": [1000.0, 1200.0, 900.0],
        }
    )

    merged = _merge_intraday_3min_bars(existing=existing, today_bars=today_bars)

    today_rows = merged[merged["timestamp"].dt.date == pd.Timestamp("2026-03-25").date()]
    assert not today_rows.empty
    assert list(merged.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_patch_minute_volume_from_history_uses_fyers_history_when_live_volume_zero():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.broker_manager.fetch_history = lambda **_kwargs: pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-25 09:16:00"]),
            "open": [23149.9],
            "high": [23155.9],
            "low": [23149.2],
            "close": [23155.45],
            "volume": [1250000.0],
        }
    )
    minute_bar = pd.Series(
        {
            "open": 23149.9,
            "high": 23155.9,
            "low": 23149.2,
            "close": 23155.45,
            "volume": 0.0,
            "symbol": "NSE:NIFTY50-INDEX",
        },
        name=pd.Timestamp("2026-03-25 09:16:00"),
    )

    patched = service._patch_minute_volume_from_history("NSE:NIFTY50-INDEX", minute_bar)

    assert patched["volume"] == 1250000.0


def test_fetch_intraday_history_cache_refreshes_when_later_minute_is_requested():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    calls: list[int] = []

    def _fetch_history(**_kwargs):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-03-25 09:16:00"]),
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [100.0],
                }
            )
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-25 09:16:00", "2026-03-25 09:17:00"]),
                "open": [1.0, 1.0],
                "high": [1.0, 1.0],
                "low": [1.0, 1.0],
                "close": [1.0, 1.0],
                "volume": [100.0, 200.0],
            }
        )

    service.broker_manager.fetch_history = _fetch_history

    early = service._fetch_intraday_history_with_cache(
        "NSE:NIFTY50-INDEX",
        pd.Timestamp("2026-03-25").date(),
        min_timestamp=pd.Timestamp("2026-03-25 09:16:00"),
    )
    later = service._fetch_intraday_history_with_cache(
        "NSE:NIFTY50-INDEX",
        pd.Timestamp("2026-03-25").date(),
        min_timestamp=pd.Timestamp("2026-03-25 09:17:00"),
    )

    assert len(calls) == 2
    assert early.index.max() == pd.Timestamp("2026-03-25 09:16:00")
    assert later.index.max() == pd.Timestamp("2026-03-25 09:17:00")


def test_backfill_recent_live_minute_volumes_updates_zero_rows(tmp_path):
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"),
        option_chain_dir=tmp_path / "option_chain_data",
    )
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=CsvDataCatalog(tmp_path / "data"),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    trade_date = pd.Timestamp("2026-03-25").date()
    service.minute_data["NSE:NIFTY50-INDEX"] = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [0.0, 10.0],
        },
        index=pd.to_datetime(["2026-03-25 09:16:00", "2026-03-25 09:17:00"]),
    )
    service.broker_manager.fetch_history = lambda **_kwargs: pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-25 09:16:00", "2026-03-25 09:17:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1250000.0, 10.0],
        }
    )

    service._backfill_recent_live_minute_volumes("NSE:NIFTY50-INDEX", trade_date)

    assert service.minute_data["NSE:NIFTY50-INDEX"].loc[pd.Timestamp("2026-03-25 09:16:00"), "volume"] == 1250000.0


def test_save_option_chain_snapshot_writes_expected_schema(tmp_path):
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"),
        option_chain_dir=tmp_path / "option_chain_data",
    )
    settings.ensure_directories()
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    oc_df = pd.DataFrame(
        {
            "strike": [23000],
            "expiry": ["2026-03-25"],
            "option_type": ["CE"],
            "symbol": ["NSE:NIFTY2632523000CE"],
            "ltp": [120.5],
            "bid": [120.4],
            "ask": [120.6],
            "volume": [100000],
            "oi": [150000],
            "change": [10.0],
            "change_percent": [9.0],
            "iv": [0.0],
            "delta": [0.52],
            "gamma": [0.0012],
            "theta": [-8.4],
            "vega": [11.3],
            "rho": [0.7],
        }
    )

    service._save_option_chain_snapshot("NIFTY50", pd.Timestamp("2026-03-25 10:30:00"), oc_df, 22950.0, 14.2)

    snap_path = settings.option_chain_data_dir / "NIFTY50_strikes_20260325.csv"
    vix_path = settings.option_chain_data_dir / "VIX_20260325.csv"
    snap = pd.read_csv(snap_path)
    vix = pd.read_csv(vix_path)
    assert snap.iloc[0]["spot_price"] == 22950.0
    assert snap.iloc[0]["open_interest"] == 150000
    assert snap.iloc[0]["delta"] == 0.52
    assert snap.iloc[0]["gamma"] == 0.0012
    assert snap.iloc[0]["theta"] == -8.4
    assert snap.iloc[0]["vega"] == 11.3
    assert snap.iloc[0]["rho"] == 0.7
    assert "oi" not in snap.columns
    assert vix.iloc[0]["vix"] == 14.2


def test_save_option_chain_snapshot_repairs_existing_old_schema_file(tmp_path):
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"),
        option_chain_dir=tmp_path / "option_chain_data",
    )
    settings.ensure_directories()
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    snap_path = settings.option_chain_data_dir / "NIFTY50_strikes_20260325.csv"
    snap_path.write_text(
        "timestamp,spot_price,strike,expiry,option_type,symbol,ltp,bid,ask,volume,open_interest,change,change_percent,iv\n"
        "2026-03-25 10:27:00,22950.0,23000.0,,CE,NSE:NIFTY2632523000CE,120.5,120.4,120.6,100000,150000,10.0,9.0,0.0\n"
    )

    oc_df = pd.DataFrame(
        {
            "strike": [23000],
            "expiry": [None],
            "option_type": ["CE"],
            "symbol": ["NSE:NIFTY2632523000CE"],
            "ltp": [121.5],
            "bid": [121.4],
            "ask": [121.6],
            "volume": [100500],
            "oi": [150500],
            "change": [11.0],
            "change_percent": [10.0],
            "iv": [12.0],
            "delta": [0.52],
            "gamma": [0.0012],
            "theta": [-8.4],
            "vega": [11.3],
            "rho": [0.7],
        }
    )

    service._save_option_chain_snapshot("NIFTY50", pd.Timestamp("2026-03-25 10:28:00"), oc_df, 22955.0, 14.2)

    repaired = pd.read_csv(snap_path)
    assert "delta" in repaired.columns
    assert "gamma" in repaired.columns
    assert "theta" in repaired.columns
    assert "vega" in repaired.columns
    assert "rho" in repaired.columns
    assert len(repaired) == 2


def test_select_adjacent_option_chain_strikes_uses_seven_nearest_levels():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"), option_chain_adjacent_strikes=7)
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    current_df = pd.DataFrame(
        {
            "option_type": ["CE"] * 10,
            "strike": [22800, 22850, 22900, 22950, 23000, 23050, 23100, 23150, 23200, 23250],
        }
    )

    strikes = service._select_adjacent_option_chain_strikes(
        current_df,
        23000,
        "CE",
        count=settings.option_chain_adjacent_strikes,
    )

    assert strikes == [23000, 22950, 23050, 22900, 23100, 22850, 23150]


def test_compute_previous_day_supertrend_seed_uses_strategy_timeframe_history():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.strategy_data["NSE:NIFTY50-INDEX"] = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-20 15:21:00",
                    "2026-03-20 15:24:00",
                    "2026-03-20 15:27:00",
                    "2026-03-23 09:15:00",
                    "2026-03-23 09:18:00",
                    "2026-03-23 09:21:00",
                    "2026-03-23 09:24:00",
                    "2026-03-23 09:27:00",
                    "2026-03-23 09:30:00",
                    "2026-03-23 09:33:00",
                    "2026-03-23 09:36:00",
                    "2026-03-23 09:39:00",
                ]
            ),
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5],
            "volume": [10] * 12,
        }
    ).set_index("timestamp")

    seed = service._compute_previous_day_supertrend_seed("NSE:NIFTY50-INDEX")

    assert seed["direction"] in {1, -1}
    assert seed["supertrend"] > 0


def test_build_option_chain_strike_summary_includes_both_ce_and_pe(tmp_path):
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"),
        option_chain_dir=tmp_path / "option_chain_data",
        option_chain_adjacent_strikes=3,
    )
    settings.ensure_directories()
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    sent_messages: list[str] = []
    service.notifier.send = sent_messages.append

    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-25 09:15:00", periods=24, freq="3min").tolist() * 2,
            "option_type": ["CE"] * 24 + ["PE"] * 24,
            "strike": [23000] * 12 + [23050] * 12 + [23000] * 12 + [22950] * 12,
            "ltp": [100 + i for i in range(12)] + [110 + i for i in range(12)] + [120 - i for i in range(12)] + [130 - i for i in range(12)],
        }
    )
    history_path = settings.option_chain_data_dir / "NIFTY50_strikes_20260325.csv"
    history.to_csv(history_path, index=False)

    current_df = pd.DataFrame(
        {
            "option_type": ["CE", "CE", "PE", "PE"],
            "strike": [23000, 23050, 22950, 23000],
            "ltp": [120, 120, 120, 120],
            "oi": [150000, 150000, 150000, 150000],
            "volume": [100000, 100000, 100000, 100000],
            "change": [10.0, 10.0, 10.0, 10.0],
        }
    )
    analysis = {"market_nature": {"label": "BULLISH"}}

    sent = service._send_option_chain_strike_summary(
        "NIFTY50",
        1,
        23010.0,
        pd.Timestamp("2026-03-25 10:30:00"),
        current_df,
        analysis,
    )

    assert sent is True
    assert len(sent_messages) == 1
    assert "<b>CE Supertrend</b>" in sent_messages[0]
    assert "<b>PE Supertrend</b>" in sent_messages[0]


def test_load_previous_day_option_chain_snapshot_returns_latest_reference(tmp_path):
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"),
        option_chain_dir=tmp_path / "option_chain_data",
    )
    settings.ensure_directories()
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    previous_day = settings.option_chain_data_dir / "NIFTY50_strikes_20260324.csv"
    pd.DataFrame(
        [
            {"timestamp": "2026-03-24 15:24:00", "spot_price": 22900, "strike": 22900, "option_type": "PE", "open_interest": 100000, "ltp": 100.0},
            {"timestamp": "2026-03-24 15:27:00", "spot_price": 22920, "strike": 22950, "option_type": "PE", "open_interest": 120000, "ltp": 110.0},
            {"timestamp": "2026-03-24 15:27:00", "spot_price": 22920, "strike": 23050, "option_type": "CE", "open_interest": 130000, "ltp": 90.0},
        ]
    ).to_csv(previous_day, index=False)

    snapshot = service._load_previous_day_option_chain_snapshot("NIFTY50", pd.Timestamp("2026-03-25 10:30:00"))

    assert len(snapshot) == 2
    assert set(snapshot["strike"].tolist()) == {22950, 23050}
    assert "oi" in snapshot.columns
    assert "open_interest" not in snapshot.columns


def test_limit_to_recent_trading_sessions_keeps_last_three_sessions():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-20 15:24:00",
                    "2026-03-21 15:24:00",
                    "2026-03-23 09:15:00",
                    "2026-03-24 09:15:00",
                    "2026-03-25 09:15:00",
                ]
            ),
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 20, 30, 40, 50],
        }
    )

    trimmed = LiveMarketDataService._limit_to_recent_trading_sessions(frame, 3)

    kept_dates = list(pd.to_datetime(trimmed["timestamp"]).dt.date.drop_duplicates())
    assert kept_dates == [
        pd.Timestamp("2026-03-23").date(),
        pd.Timestamp("2026-03-24").date(),
        pd.Timestamp("2026-03-25").date(),
    ]


def test_send_eod_summary_reports_flip_and_touch_counts():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    sent_messages: list[str] = []
    service.notifier.send = sent_messages.append
    service.supertrend_flip_events["NSE:NIFTY50-INDEX"] = [
        {
            "bar_time": pd.Timestamp("2026-03-24 09:18:00"),
            "direction": 1,
            "close": 22850.0,
            "points_before_flip": None,
        },
        {
            "bar_time": pd.Timestamp("2026-03-24 10:03:00"),
            "direction": -1,
            "close": 22910.0,
            "points_before_flip": 60.0,
        },
    ]
    service.supertrend_touch_events["NSE:NIFTY50-INDEX"] = [
        {"bar_time": pd.Timestamp("2026-03-24 09:24:00")},
        {"bar_time": pd.Timestamp("2026-03-24 09:51:00")},
    ]

    service._send_eod_summary(pd.Timestamp("2026-03-24").date())

    assert len(sent_messages) == 1
    assert "Flip signals: 2" in sent_messages[0]
    assert "Touch signals: 2" in sent_messages[0]
    assert "Avg points before direction change: 60.00" in sent_messages[0]


def test_major_gap_alert_fires_on_first_live_price():
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"),
        major_gap_threshold_pct=0.5,
    )
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    sent_messages: list[str] = []
    service.notifier.send = sent_messages.append
    service.previous_day_levels["NSE:NIFTY50-INDEX"] = {
        "open": 22800.0,
        "high": 22900.0,
        "low": 22750.0,
        "close": 22800.0,
    }

    service._check_gap_and_previous_day_levels(
        "NSE:NIFTY50-INDEX",
        {"ltp": 22950.0, "timestamp": pd.Timestamp("2026-03-25 09:15:02")},
    )

    assert len(sent_messages) == 1
    assert "Major Gap Up" in sent_messages[0]


def test_previous_day_level_touch_alert_fires_when_price_crosses_level(monkeypatch):
    settings = Settings(telegram=TelegramConfig(bot_token="token", chat_id="chat"))
    service = LiveMarketDataService(
        broker=_DummyBroker(settings), broker_manager=_DummyBroker(settings),
        catalog=_DummyCatalog(),
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings,
    )
    service.previous_day_levels["NSE:NIFTY50-INDEX"] = {
        "open": 22800.0,
        "high": 22900.0,
        "low": 22750.0,
        "close": 22850.0,
    }
    service.last_tick_price["NSE:NIFTY50-INDEX"] = 22890.0

    called = []
    monkeypatch.setattr(service.alert_agent, "alert_retest", lambda *a, **kw: called.append((a, kw)))

    service._maybe_alert_previous_day_level_touch(
        "NSE:NIFTY50-INDEX",
        22910.0,
        pd.Timestamp("2026-03-25 10:00:00"),
    )

    assert len(called) == 1
