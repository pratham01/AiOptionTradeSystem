"""
LiveMarketDataOrchestrator — Orchestrates live WebSocket streaming, symbol routing, and collector lifecycles.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, date, time as dt_time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from trade_system.shared.config import Settings
from trade_system.shared.live_state import LiveState
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.interfaces.live.alert_dispatcher import AlertDispatcher
from trade_system.interfaces.live.collectors.index_collector import IndexMarketDataCollector
from trade_system.interfaces.live.collectors.fo_collector import FoMarketDataCollector
from trade_system.domains.market_data.application.data_sanity_manager import DataSanityManager

LOGGER = logging.getLogger("LiveOrchestrator")
IST = ZoneInfo("Asia/Kolkata")


class LiveMarketDataOrchestrator:
    """
    Coordinates live data streams, routes ticks to Index and F&O collectors, and manages lifecycle.
    """

    def __init__(self, broker: Any, settings: Optional[Settings] = None) -> None:
        self.broker = broker
        self.settings = settings or Settings.load()
        self.auth_service = FyersAuthService(self.settings)

        # Alert Dispatcher
        from trade_system.shared.notifications.telegram import TelegramNotifier
        main_notifier = TelegramNotifier(
            token=self.settings.telegram.bot_token,
            chat_id=self.settings.telegram.chat_id,
        )
        confirmed_notifier = TelegramNotifier(
            token=self.settings.telegram.bot_token,
            chat_id=getattr(self.settings.telegram, "confirmed_chat_id", self.settings.telegram.chat_id),
        )
        self.alert_dispatcher = AlertDispatcher(
            notifier=main_notifier,
            confirmed_notifier=confirmed_notifier,
        )

        # Specialized Collectors
        self.index_collector = IndexMarketDataCollector(
            symbols=self.settings.live_symbols,
            alert_dispatcher=self.alert_dispatcher,
            strategy_timeframe_minutes=self.settings.indicator_config.trend_timeframe_minutes,
            settings=self.settings,
        )
        self.fo_collector = FoMarketDataCollector(
            alert_dispatcher=self.alert_dispatcher,
            settings=self.settings,
        )

        self.sanity_manager = DataSanityManager(broker=broker, settings=self.settings)
        self.live_state = LiveState.get_instance()

        self.ws = None
        self.ws_running = False
        self.shutdown = False
        self.last_initialized_date: Optional[date] = None

    def run_startup_sync(self) -> None:
        """Verify DB sanity and backfill missing candles before market open."""
        LOGGER.info("Running DataSanityManager startup verification...")
        try:
            report = self.sanity_manager.run_full_sanity_check()
            LOGGER.info("Startup Sanity Check: %s", report.summary())
        except Exception as exc:
            LOGGER.warning("Startup data sanity check failed (non-fatal): %s", exc)

    def route_tick(self, symbol: str, tick: dict) -> None:
        """Route incoming WebSocket tick to the appropriate collector."""
        if "INDEX" in symbol:
            self.index_collector.ingest_tick(symbol, tick)
        else:
            self.fo_collector.ingest_tick(symbol, tick)

    def start_websocket(self) -> None:
        """Connect to Fyers DataSocket WebSocket."""
        try:
            from fyers_apiv3.FyersWebsocket import data_ws
            token = self.auth_service.get_valid_token()
            all_symbols = list(set(self.index_collector.symbols + self.fo_collector.symbols[:50])) # Subscribe indices + top FO

            self.ws = data_ws.FyersDataSocket(
                access_token=token,
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=lambda: self.ws.subscribe(symbols=all_symbols, data_type="SymbolUpdate"),
                on_close=lambda msg: LOGGER.warning("WebSocket closed: %s", msg),
                on_error=lambda msg: LOGGER.error("WebSocket error: %s", msg),
                on_message=self._on_ws_message,
            )
            self.ws_running = True
            self.live_state.set_bot_running(True)
            self.alert_dispatcher.send_market_status("🟢 <b>Trading Bot Live</b>\n\nLive multi-timeframe streaming started.")
            threading.Thread(target=self.ws.connect, daemon=True).start()
        except Exception as exc:
            LOGGER.error("Failed to start WebSocket: %s", exc)

    def _on_ws_message(self, message: Any) -> None:
        payload = message if isinstance(message, list) else [message]
        for item in payload:
            sym = item.get("symbol")
            if not sym or item.get("ltp") is None:
                continue
            tick = {
                "timestamp": datetime.now(),
                "ltp": float(item["ltp"]),
                "volume": float(item.get("volume", 0) or item.get("vol_traded_today", 0) or 0.0),
            }
            self.route_tick(sym, tick)

    def stop(self) -> None:
        self.shutdown = True
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.live_state.set_bot_running(False)
        self.alert_dispatcher.send_market_status("🔴 <b>Trading Bot Stopped</b>\n\nMarket monitoring closed.")
