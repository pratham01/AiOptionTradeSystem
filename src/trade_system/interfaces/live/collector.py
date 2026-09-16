from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import csv
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trade_system.domains.analysis.application.backtesting.ict_fvg_liquidity_research import (
    FVGDetector,
    IctFvgLiquidityResearch,
    LiquidityTracker,
    MarketStructureTracker,
    OrderBlockDetector,
)
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.shared import TradeSuggestion, TradeDirection
from trade_system.shared.config import IndicatorConfig, Settings
from trade_system.domains.market_data.infrastructure.data.storage import CsvDataCatalog
from trade_system.domains.strategy.application.indicators import calculate_supertrend
from trade_system.domains.strategy.application.indicators.rsi_divergence import RsiDivergence
from trade_system.domains.advisory.application.agent.live_alert_agent import LiveAlertAgent
from trade_system.domains.advisory.application.agent.option_chain_monitor_agent import OptionChainMonitorAgent
from trade_system.domains.analysis.application.analysis.mwpl_analyzer import MwplAnalyzer
from trade_system.domains.advisory.application.agent.early_morning_agent import EarlyMorningAgent
from trade_system.domains.advisory.application.agent.sniper_reversal_agent import SniperReversalAgent
from trade_system.domains.advisory.application.agent.gamma_blast_agent import GammaBlastAgent
from trade_system.interfaces.live.confirmed_strategy import (
    build_confirmed_entry_message as _build_confirmed_entry_message,
    build_confirmed_exit_message as _build_confirmed_exit_message,
    confirmed_entry_payload as _confirmed_entry_payload,
    confirmed_exit_payload as _confirmed_exit_payload,
    prepare_confirmed_strategy_frame as _prepare_confirmed_strategy_frame,
)
from trade_system.domains.market_data.infrastructure.database.db_write_worker import DbWriteWorker
from trade_system.shared.live_state import LiveStateWriter
from trade_system.interfaces.live.helpers import (
    _completed_timeframe_bars,
    _is_market_timestamp,
    _latest_session_minute,
    _sanitize_intraday_minutes,
    create_minute_bar,
    detect_smc_signal,
    get_gap_adjusted_data,
    resample_to_timeframe,
    valid_supertrend_rows as _valid_supertrend_rows,
)
from trade_system.shared.notifications.telegram import TelegramNotifier
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import save_market_data_batch, save_option_chain_batch
from trade_system.domains.analysis.application.analysis.breakout_screener import BreakoutScreener
from trade_system.domains.analysis.application.analysis.intraday_option_edge import IntradayOptionEdgePipeline
from sqlalchemy.orm import Session

from trade_system.domains.strategy.application.indicators.volume_profile import VolumeProfileIndicator
from trade_system.shared import VolumeProfileSummary
from trade_system.domains.strategy.application.indicators.support_resistance_channels import SupportResistanceChannelDetector
from trade_system.domains.trading.application.execution.execution_engine import ExecutionEngine
from trade_system.domains.trading.application.execution.position_manager import PositionManager
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# NEW: Modular DDD pipeline imports (Phase 6+7 wiring)
# ---------------------------------------------------------------------------
from trade_system.domains.market_data.application.bar_store import BarStore
from trade_system.interfaces.live.bar_aggregator import BarAggregator
from trade_system.interfaces.live.pipelines.index_pipeline import IndexPipeline
from trade_system.interfaces.live.pipelines.fo_pipeline import FoPipeline
from trade_system.interfaces.live.pipelines.sr_pipeline import SrPipeline
from trade_system.interfaces.live.pipelines.gamma_pipeline import GammaPipeline

# ---------------------------------------------------------------------------
# Observability: metrics, health checks, health HTTP server
# ---------------------------------------------------------------------------
from trade_system.shared.observability.metrics import METRICS
from trade_system.shared.observability.health import HEALTH_CHECKER
from trade_system.interfaces.live.health_server import start_health_server


LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


@dataclass(slots=True)
class TrendRun:
    direction: int
    start_bar_time: pd.Timestamp
    start_price: float


@dataclass(slots=True)
class VolumeProfileSummary:
    point_of_control: float
    value_area_low: float
    value_area_high: float
    total_volume: float


@dataclass(slots=True)
class IctLiveTrade:
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    trigger: str
    confluence_score: int
    points_captured: float
    risk_points: float
    r_multiple: float


class LiveMarketDataService:
    def __init__(
        self,
        broker: FyersBrokerClient,
        catalog: CsvDataCatalog,
        symbols: list[str],
        broker_manager=None,
        timeframe_minutes: int = 1,
        strategy_timeframe_minutes: int | None = None,
        indicator_config: IndicatorConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.broker = broker
        self.settings = settings or broker.settings
        if broker_manager is None:
            from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
            self.broker_manager = get_broker_manager(self.settings)
        else:
            self.broker_manager = broker_manager
        
        self.catalog = catalog
        self.symbols = symbols
        self.timeframe_minutes = timeframe_minutes
        self.indicator_config = indicator_config or self.settings.indicator_config
        self.engine = get_engine()
        self.db_worker = DbWriteWorker(self.engine)
        self.db_worker.start()
        self.last_initialization_attempt = 0.0
        self.strategy_timeframe_minutes = strategy_timeframe_minutes or self.indicator_config.trend_timeframe_minutes
        self.supertrend_period = self.indicator_config.supertrend_period
        self.supertrend_multiplier = self.indicator_config.supertrend_multiplier
        self.rsi_divergence = RsiDivergence(len_fast=5, len_slow=14)
        
        self.notifier = TelegramNotifier(self.settings.telegram.bot_token, self.settings.telegram.chat_id)
        self.confirmed_notifier = TelegramNotifier(
            self.settings.st_confirmed_telegram.bot_token or self.settings.telegram.bot_token,
            self.settings.st_confirmed_telegram.chat_id or self.settings.telegram.chat_id,
        )
        
        self.alert_agent = LiveAlertAgent(settings=self.settings, notifier=self.notifier)
        self.oc_monitor_agent = OptionChainMonitorAgent(settings=self.settings, notifier=self.notifier)
        self.mwpl_analyzer = MwplAnalyzer(settings=self.settings)
        self.early_morning_agent = EarlyMorningAgent(broker=self.broker)
        self.sniper_agent = SniperReversalAgent()
        self.gamma_agent = GammaBlastAgent()
        self.mwpl_setups = self.mwpl_analyzer.identify_setups()
        
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine(broker=self.broker, risk_manager=self.risk_manager, settings=self.settings)
        self.position_manager = PositionManager(broker=self.broker, risk_manager=self.risk_manager, settings=self.settings)
        
        # Inject execution components into agents that support autonomous trading
        self.alert_agent.set_execution_engine(self.execution_engine, self.position_manager)

        self.confirmed_timeframe_minutes = self.indicator_config.confirmed_timeframe_minutes
        self.confirmed_entry_cutoff = self.indicator_config.confirmed_entry_cutoff

        self.minute_data: dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in symbols}
        self.strategy_data: dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in symbols}
        self.tick_buffer: dict[str, list[dict]] = {symbol: [] for symbol in symbols}
        self.last_cumulative_volume: dict[str, float | None] = {symbol: None for symbol in symbols}
        self.last_tick_price: dict[str, float | None] = {symbol: None for symbol in symbols}
        self.history_volume_cache: dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in symbols}
        self.current_minute: dict[str, datetime | None] = {symbol: None for symbol in symbols}
        self.last_trend: dict[str, int | None] = {symbol: None for symbol in symbols}
        self.last_processed_trend_bar_time: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}
        self.last_signal_bar_time: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}
        self.last_touch_bar_time: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}
        self.last_smc_signal_bar_time: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}
        self.last_confirmed_bar_time: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}
        self.confirmed_position: dict[str, dict | None] = {symbol: None for symbol in symbols}
        self.current_trend_run: dict[str, TrendRun | None] = {symbol: None for symbol in symbols}
        self.supertrend_flip_events: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in symbols}
        self.supertrend_touch_events: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in symbols}
        self.previous_day_levels: dict[str, dict[str, float | str] | None] = {symbol: None for symbol in symbols}
        self.previous_day_level_alerts_sent: dict[str, set[str]] = {symbol: set() for symbol in symbols}
        self.gap_alert_sent: dict[str, bool] = {symbol: False for symbol in symbols}
        self.first_live_price: dict[str, float | None] = {symbol: None for symbol in symbols}
        self.premarket_reports_sent: dict[str, date | None] = {symbol: None for symbol in symbols}
        self.last_level_alert_time: dict[str, dict[str, pd.Timestamp]] = {symbol: {} for symbol in symbols}
        self.sr_detector = SupportResistanceChannelDetector()
        self.last_sr_alert_time: dict[str, dict[str, pd.Timestamp]] = {symbol: {} for symbol in symbols}
        self.daily_zones: dict[str, dict[str, float]] = {symbol: {} for symbol in symbols}
        self.zone_buffer_pct: float = 0.002

        self.market_open_today = False
        self.last_initialized_date: date | None = None
        self.ws = None
        self.ws_running = False
        self.oc_thread: threading.Thread | None = None
        self.oc_running = False
        self.oc_analyzers: dict[str, Any] = {}
        self.prev_oc_df: dict[str, pd.DataFrame | None] = {
            self._short_symbol(sym): None for sym in settings.index_symbols
        }
        self.latest_oc_analysis: dict[str, Any | None] = {
            self._short_symbol(sym): None for sym in settings.index_symbols
        }
        self.last_option_chain_direction: dict[str, int | None] = {
            self._short_symbol(sym): None for sym in settings.index_symbols
        }
        self.expiry_max_pain_sent: dict[str, date | None] = {
            self._short_symbol(sym): None for sym in settings.index_symbols
        }

        self.eod_summary_sent_for: date | None = None
        self.shutdown = False
        self.ict_rules = IctFvgLiquidityResearch()
        self.ict_fvg = {symbol: FVGDetector() for symbol in symbols}
        self.ict_ob = {symbol: OrderBlockDetector() for symbol in symbols}
        self.ict_liq = {symbol: LiquidityTracker() for symbol in symbols}
        self.ict_ms = {symbol: MarketStructureTracker() for symbol in symbols}
        self.ict_last_processed_bar_time: dict[str, pd.Timestamp | None] = {symbol: None for symbol in symbols}
        self.ict_open_position: dict[str, dict[str, object] | None] = {symbol: None for symbol in symbols}
        self.ict_trades: dict[str, list[IctLiveTrade]] = {symbol: [] for symbol in symbols}
        self.ict_signal_events: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in symbols}

        self.gamma_open_position: dict[str, dict[str, object] | None] = {symbol: None for symbol in symbols}
        self.gamma_trades: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in symbols}

        # RSI divergence tracking — must mirror _reset_intraday_state
        self.last_rsi_div_signal_time: dict[str, datetime | None] = {symbol: None for symbol in symbols}

        self.breakout_screener = BreakoutScreener()
        self.breakout_thread: threading.Thread | None = None
        self.last_breakout_alerts: dict[str, pd.Timestamp] = {}

        # Intraday Option Edge pipeline
        self.option_edge_pipeline = IntradayOptionEdgePipeline()
        self.option_edge_thread: threading.Thread | None = None
        self.last_option_edge_alerts: dict[str, pd.Timestamp] = {}
        self.alert_debounce_seconds: int = 300  # 5-minute cooldown on level-touch alerts

        # Live state writer for dashboard communication
        self._live_state = LiveStateWriter()

        # ----------------------------------------------------------------
        # Modular DDD pipeline wiring (after all state dicts are initialised)
        # ----------------------------------------------------------------
        _market_start_clock = self._parse_clock(self.settings.market_start)
        _market_end_clock = self._parse_clock(self.settings.market_end)

        # BarStore: centralised persistence (DB async + CSV)
        self.bar_store = BarStore(
            db_worker=self.db_worker,
            catalog=self.catalog,
            strategy_timeframe_minutes=self.strategy_timeframe_minutes,
            market_start=_market_start_clock,
            market_end=_market_end_clock,
        )

        # BarAggregator: tick → 1-min bar emitter
        # minute_data is the shared reference — aggregator writes into it;
        # pipelines read from it.
        self.bar_aggregator = BarAggregator(
            symbols=symbols,
            on_bar_complete=self._on_bar_complete,
            minute_data_ref=self.minute_data,
        )

        # IndexPipeline: full analytics for index symbols
        self.index_pipeline = IndexPipeline(
            settings=self.settings,
            alert_agent=self.alert_agent,
            notifier=self.notifier,
            confirmed_notifier=self.confirmed_notifier,
            bar_store=self.bar_store,
            strategy_timeframe_minutes=self.strategy_timeframe_minutes,
            supertrend_period=self.supertrend_period,
            supertrend_multiplier=self.supertrend_multiplier,
            rsi_divergence=self.rsi_divergence,
            early_morning_agent=self.early_morning_agent,
            sniper_agent=self.sniper_agent,
            gamma_agent=self.gamma_agent,
            mwpl_analyzer=self.mwpl_analyzer,
            mwpl_setups=self.mwpl_setups,
            ict_fvg=self.ict_fvg,
            ict_ob=self.ict_ob,
            ict_liq=self.ict_liq,
            ict_ms=self.ict_ms,
            oc_snapshot_getter=lambda clean_sym: (
                self.oc_analyzers[clean_sym].get_option_chain_df() if clean_sym in self.oc_analyzers else self.prev_oc_df.get(clean_sym),
                self.latest_oc_analysis.get(clean_sym)
            ),
        )

        # FoPipeline: lightweight analytics for F&O equity symbols
        self.fo_pipeline = FoPipeline(
            settings=self.settings,
            bar_store=self.bar_store,
            strategy_timeframe_minutes=self.strategy_timeframe_minutes,
        )

        # Pre-register all symbols in both pipelines
        for sym in symbols:
            if self._is_index(sym):
                self.index_pipeline.register_symbol(sym)
            else:
                self.fo_pipeline.register_symbol(sym)

        # ---- Phase 7: S/R / Gamma pipelines (index-only) ----
        self.sr_pipeline = SrPipeline(
            notifier=self.notifier,
            alert_agent=self.alert_agent,
            alert_debounce_seconds=self.alert_debounce_seconds,
        )
        self.gamma_pipeline = GammaPipeline(
            gamma_agent=self.gamma_agent,
            notifier=self.notifier,
            latest_oc_analysis=self.latest_oc_analysis,
        )
        # Register index symbols in these pipelines
        for sym in symbols:
            if self._is_index(sym):
                self.sr_pipeline.register_symbol(sym)
                self.gamma_pipeline.register_symbol(sym)

    # ------------------------------------------------------------------
    # Symbol classification helpers (single source of truth)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_index(symbol: str) -> bool:
        """Return True for NSE/BSE index symbols (e.g. NSE:NIFTY50-INDEX)."""
        return symbol.endswith("-INDEX")

    @staticmethod
    def _is_fo_stock(symbol: str) -> bool:
        """Return True for F&O equity symbols (e.g. NSE:RELIANCE-EQ)."""
        return not symbol.endswith("-INDEX")

    @staticmethod
    def _now_ist() -> datetime:
        return datetime.now(IST)

    @classmethod
    def _today_ist(cls) -> date:
        return cls._now_ist().date()

    @staticmethod
    def _from_epoch_ist(epoch_seconds: float) -> datetime:
        return datetime.fromtimestamp(epoch_seconds, tz=IST).replace(microsecond=0, tzinfo=None)

    def run_forever(self) -> None:
        LOGGER.info("Starting intraday live bot monitor.")
        import time
        while not self.shutdown:
            now = self._now_ist().replace(tzinfo=None)
            today = now.date()

            if self.last_initialized_date != today and now.time() >= self._parse_clock(self.settings.market_premarket_check):
                current_time_sec = time.time()
                if current_time_sec - self.last_initialization_attempt >= 300: # 5 minutes backoff
                    self.last_initialization_attempt = current_time_sec
                    self._initialize_trading_day(today)

            if self.market_open_today:
                if self._parse_clock(self.settings.market_start) <= now.time() < self._parse_clock(self.settings.market_end):
                    if not self.ws_running:
                        self.start()
                elif now.time() >= self._parse_clock(self.settings.market_end):
                    if self.ws_running:
                        self.stop()
                        self._send_eod_summary(today)
                        self.notifier.send(
                            "🔴 <b>Trading Bot Stopped</b>\n\nMarket closed at 15:30. Monitoring stopped for the day."
                        )
                    LOGGER.info("Market is closed for the day. Exiting live bot process to prevent memory leaks.")
                    break
            time.sleep(5)

    def _ensure_broker_session(self) -> None:
        """Verify the broker session and handle rate-limit or auth failures gracefully."""
        try:
            self.broker.verify_session()
        except Exception as exc:
            LOGGER.warning("Fyers session verification failed: %s.", exc)
            err_msg = str(exc)
            is_rate_limit = any(
                term in err_msg
                for term in ["API Limit exceeded", "Limit exceeded", "429", "Too Many Requests", "Bad request"]
            )
            if is_rate_limit:
                LOGGER.warning(
                    "Rate limits encountered on REST API. Proceeding since access token is present."
                )
                return
            LOGGER.info("Attempting TOTP token refresh...")
            if not self._refresh_token_via_totp():
                if self.broker.access_token:
                    LOGGER.warning("TOTP refresh failed, but proceeding with existing access token.")
                    return
                raise exc
            try:
                self.broker.verify_session()
            except Exception as exc2:
                LOGGER.warning(
                    "Fyers session verification failed after TOTP refresh: %s. Proceeding anyway.", exc2
                )

    def start(self) -> None:
        self._ensure_broker_session()

        # Start health probe HTTP server (port 9090) for Docker/K8s
        start_health_server(port=9090)
        HEALTH_CHECKER.register_check("websocket", lambda: {
            "status": "healthy" if self.ws_running else "unhealthy",
            "connected": self.ws_running,
        })
        METRICS.set_gauge("active_symbols_count", len(self.symbols))

        from fyers_apiv3.FyersWebsocket import data_ws

        self.ws = data_ws.FyersDataSocket(
            access_token=self.broker.websocket_access_token(),
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        self.ws_running = True
        self._live_state.set_bot_running(True)
        self._live_state.flush(force=True)
        self.notifier.send("🟢 <b>Trading Bot Started</b>\n\nMarket is open. Live monitoring started.")
        threading.Thread(target=self.ws.connect, daemon=True).start()
        self._start_option_chain_thread()
        self._start_breakout_thread()
        self._start_option_edge_thread()

    def stop(self) -> None:
        self._flush_all_open_minutes()
        if self.ws and self.ws_running:
            try:
                self.ws.close()
            except Exception:
                LOGGER.exception("Failed to close websocket cleanly.")
        self.oc_running = False
        self.ws_running = False
        self.ws = None
        self.db_worker.stop()
        if hasattr(self, 'scheduler') and self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        self._live_state.set_bot_running(False)
        self._live_state.flush(force=True)

    def on_open(self) -> None:
        LOGGER.info("Websocket connected. Subscribing to %s", self.symbols)
        if self.ws is not None:
            try:
                self.ws.subscribe(symbols=self.symbols, data_type="SymbolUpdate")
                self.ws.keep_running()
            except Exception as e:
                LOGGER.error("Failed to subscribe in on_open: %s", e)

    def on_message(self, message) -> None:
        payload = message if isinstance(message, list) else [message]
        for item in payload:
            symbol = item.get("symbol")
            if symbol not in self.symbols or item.get("ltp") is None:
                continue
            tick_time = self._from_epoch_ist(item.get("timestamp", time.time()))
            if not _is_market_timestamp(
                tick_time,
                self._parse_clock(self.settings.market_start),
                self._parse_clock(self.settings.market_end),
            ):
                continue
            cumulative_volume = (
                item.get("volume")
                or item.get("vol_traded_today")
                or item.get("vol_traded")
                or item.get("v")
                or 0.0
            )
            tick = {
                "timestamp": tick_time,
                "ltp": float(item["ltp"]),
                "volume": float(cumulative_volume),
            }
            self._process_tick(symbol, tick)
            METRICS.increment("ticks_processed_total")

    def on_close(self, message) -> None:
        LOGGER.warning("Websocket closed: %s", message)
        self._flush_all_open_minutes()
        self.ws_running = False

    def on_error(self, message) -> None:
        LOGGER.error("Websocket error: %s", message)
        METRICS.increment("ws_errors_total")
        msg_str = str(message).lower()
        # Detect fatal token errors and request a refresh on the next cycle
        if "token" in msg_str or "-99" in msg_str or "-300" in msg_str:
            LOGGER.warning("Websocket reported token error. Forcing token refresh on next cycle.")
            if hasattr(self, "broker"):
                # Invalidate the cached websocket token so a fresh one is used on reconnect
                self.broker._ws_token_cache = None
                self.broker.force_refresh = True
            self.ws_running = False  # Signal run_forever() to restart the websocket

    def _initialize_trading_day(self, current_date: date) -> None:
        self.market_open_today = False
        self._reset_intraday_state()

        if current_date.weekday() >= 5:
            LOGGER.info("Weekend detected. Bot inactive on %s.", current_date)
            self.last_initialized_date = current_date
            
            try:
                LOGGER.info("Running post-market weekend analysis...")
                import asyncio
                from trade_system.domains.advisory.application.agent.postmarket_improver_agent import PostMarketImproverAgent
                agent = PostMarketImproverAgent(broker=self.broker)

                
                # Check if there is already a running event loop in this thread
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                
                if loop and loop.is_running():
                    # We are already inside an async context (e.g. from agent_engine.py)
                    loop.create_task(agent.run_post_market_analysis())
                else:
                    # We are in a sync context (e.g. from collector.py run_forever)
                    asyncio.run(agent.run_post_market_analysis())
            except Exception as e:
                LOGGER.error(f"Failed to run weekend post-market analysis: {e}")
                
            return

        try:
            self._ensure_broker_session()
            market_open, detail = self.broker.is_market_open_today()
        except Exception as exc:
            LOGGER.error("Failed to check FYERS market status: %s", exc, exc_info=True)
            if "API Limit exceeded per day" in str(exc) or "Limit exceeded" in str(exc) or "Unable to authenticate" in str(exc):
                LOGGER.warning("API rate limit exceeded during startup. Falling back to weekday check.")
                market_open = (current_date.weekday() < 5)
                detail = "API Rate Limit Fallback"
            else:
                return

        if not market_open:
            LOGGER.info("Market not yet open (Current status: %s). Retrying in next loop...", detail)
            # Do NOT set last_initialized_date here, so we retry.
            return

        self.market_open_today = True
        self.last_initialized_date = current_date

        try:
            LOGGER.info("Running DataSanityManager to preserve data sanity and auto-heal missing historical data for FO & Indices...")
            from trade_system.domains.market_data.application.data_sanity_manager import DataSanityManager
            sanity_mgr = DataSanityManager(broker=self.broker, settings=self.settings)
            sanity_mgr.ensure_data_sanity_and_heal(resolutions=["15", "D"], lookback_days=5)
        except Exception as e:
            LOGGER.error(f"Failed to run DataSanityManager: {e}", exc_info=True)

        for symbol in self.symbols:
            try:
                self._refresh_premarket_reference_data(symbol)
            except Exception as e:
                LOGGER.warning(f"Failed to refresh premarket reference data for {symbol} (using local copy if available): {e}")

            try:
                self.minute_data[symbol] = self._fetch_seed_minute_data(
                    symbol,
                    days=self.settings.premarket_intraday_history_days,
                )
            except Exception as e:
                LOGGER.warning(f"Failed to fetch seed minute data for {symbol}: {e}")
                if symbol not in self.minute_data or self.minute_data[symbol] is None:
                    self.minute_data[symbol] = pd.DataFrame()

            try:
                self._load_existing_today_minute_file(symbol, current_date)
            except Exception as e:
                LOGGER.warning(f"Failed to load existing today minute file for {symbol}: {e}")

            # --- Zone Calculation ---
            try:
                self._calculate_daily_zones(symbol)
            except Exception as e:
                LOGGER.warning(f"Failed to calculate daily zones for {symbol}: {e}")

            initial = {"direction": 0, "supertrend": 0.0}
            try:
                initial = self._compute_previous_day_supertrend_seed(symbol)
            except Exception as e:
                LOGGER.warning(f"Failed to compute previous day supertrend seed for {symbol}: {e}")

            try:
                self._prime_intraday_supertrend_state(symbol, current_date)
            except Exception as e:
                LOGGER.warning(f"Failed to prime intraday supertrend state for {symbol}: {e}")

            try:
                recent_daily = self._fetch_recent_daily_context(symbol)
                self.previous_day_levels[symbol] = recent_daily[-1] if recent_daily else None
            except Exception as e:
                LOGGER.warning(f"Failed to fetch recent daily context for {symbol}: {e}")

            
            # Build and send comprehensive pre-market report
            if self.premarket_reports_sent.get(symbol) != current_date:
                report = self._build_premarket_console_summary(symbol, initial)
                zones = self.daily_zones.get(symbol)
                
                zone_info = ""
                if zones:
                    zone_info = (
                        f"\n\n🏰 <b>Support/Resistance Zones</b>\n"
                        f"🔴 Supply: ₹{zones['PDH']:.2f} (PDH) | ₹{zones['VAH']:.2f} (VAH)\n"
                        f"🔵 Demand: ₹{zones['PDL']:.2f} (PDL) | ₹{zones['VAL']:.2f} (VAL)\n"
                        f"⚖️ Pivot: ₹{zones['POC']:.2f} (POC)"
                    )

                # --- Current Trend Heartbeat ---
                curr_trend = self.last_trend.get(symbol)
                trend_icon = "🟢 UP" if curr_trend == 1 else "🔴 DOWN" if curr_trend == -1 else "⚪ NEUTRAL"
                
                # Only send premarket Telegram report for index symbols.
                # F&O stock reports are suppressed unless explicitly enabled to avoid
                # flooding the alert channel with 180+ messages on startup.
                is_index = self._is_index(symbol)
                if is_index or getattr(self.settings, "enable_fo_telegram_alerts", False):
                    self.notifier.send(
                        f"📊 <b>Premarket Report — {self._short_symbol(symbol)}</b>\n\n"
                        f"{report}"
                        f"{zone_info}\n\n"
                        f"🧭 <b>Current Trend:</b> {trend_icon}"
                    )
                self.premarket_reports_sent[symbol] = current_date
                LOGGER.info("Generated premarket report for %s (telegram=%s)", symbol, is_index)
            else:
                LOGGER.info("Premarket report already sent for %s today. Skipping.", symbol)

    def _reset_intraday_state(self) -> None:
        self.minute_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.strategy_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.tick_buffer = {symbol: [] for symbol in self.symbols}
        self.last_cumulative_volume = {symbol: None for symbol in self.symbols}
        self.last_tick_price = {symbol: None for symbol in self.symbols}
        self.history_volume_cache = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.current_minute = {symbol: None for symbol in self.symbols}
        self.last_trend = {symbol: None for symbol in self.symbols}
        self.last_processed_trend_bar_time = {symbol: None for symbol in self.symbols}
        self.last_signal_bar_time = {symbol: None for symbol in self.symbols}
        self.last_touch_bar_time = {symbol: None for symbol in self.symbols}
        self.last_smc_signal_bar_time = {symbol: None for symbol in self.symbols}
        self.last_confirmed_bar_time = {symbol: None for symbol in self.symbols}
        self.confirmed_position = {symbol: None for symbol in self.symbols}
        self.current_trend_run = {symbol: None for symbol in self.symbols}
        self.supertrend_flip_events = {symbol: [] for symbol in self.symbols}
        self.supertrend_touch_events = {symbol: [] for symbol in self.symbols}
        self.last_rsi_div_signal_time = {symbol: None for symbol in self.symbols}
        self.previous_day_levels = {symbol: None for symbol in self.symbols}


        self.gap_alert_sent = {symbol: False for symbol in self.symbols}
        self.first_live_price = {symbol: None for symbol in self.symbols}
        self.last_level_alert_time = {symbol: {} for symbol in self.symbols}
        self.expiry_max_pain_sent = {
            self._short_symbol(sym): None for sym in self.settings.index_symbols
        }

        self.ict_fvg = {symbol: FVGDetector() for symbol in self.symbols}
        self.ict_ob = {symbol: OrderBlockDetector() for symbol in self.symbols}
        self.ict_liq = {symbol: LiquidityTracker() for symbol in self.symbols}
        self.ict_ms = {symbol: MarketStructureTracker() for symbol in self.symbols}
        self.ict_last_processed_bar_time = {symbol: None for symbol in self.symbols}
        self.ict_open_position = {symbol: None for symbol in self.symbols}
        self.ict_trades = {symbol: [] for symbol in self.symbols}
        self.ict_signal_events = {symbol: [] for symbol in self.symbols}
        self.daily_zones = {symbol: {} for symbol in self.symbols}
        self.zone_buffer_pct = 0.002

        # Gamma strategy state
        self.gamma_open_position = {symbol: None for symbol in self.symbols}
        self.gamma_trades = {symbol: [] for symbol in self.symbols}

        self.last_rsi_div_signal_time = {symbol: None for symbol in self.symbols}

        self.eod_summary_sent_for = None

        # Reset modular DDD pipeline state
        if hasattr(self, "bar_aggregator"):
            self.bar_aggregator.reset(self.symbols)
        index_syms = [s for s in self.symbols if self._is_index(s)]
        fo_syms = [s for s in self.symbols if self._is_fo_stock(s)]
        if hasattr(self, "index_pipeline"):
            self.index_pipeline.reset(index_syms)
        if hasattr(self, "fo_pipeline"):
            self.fo_pipeline.reset(fo_syms)
        if hasattr(self, "sr_pipeline"):
            self.sr_pipeline.reset(index_syms)
        if hasattr(self, "gamma_pipeline"):
            self.gamma_pipeline.reset(index_syms)
    def _refresh_premarket_reference_data(self, symbol: str) -> None:
        self._store_recent_daily_history(symbol)
        self._refresh_recent_3min_history(symbol)

    def _store_recent_daily_history(self, symbol: str) -> None:
        recent_daily = self._fetch_recent_daily_context(symbol)
        if not recent_daily:
            return
        payload = pd.DataFrame(recent_daily).rename(columns={"date": "trade_date"})
        payload["timestamp"] = pd.to_datetime(payload["trade_date"], format="mixed")
        self.catalog.write_historical_yearly(payload, symbol, "D")

    def _refresh_recent_3min_history(self, symbol: str) -> None:
        lookback_days = max(self.settings.premarket_intraday_history_days, 1) + 2
        today = self._today_ist()
        
        try:
            data = self.broker_manager.get_historical_data(
                symbol=symbol,
                start_date=today - timedelta(days=lookback_days),
                end_date=today,
                timeframe=str(self.strategy_timeframe_minutes),
            )
            if not data:
                return
            
            records = [
                {
                    "timestamp": d.timestamp,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                    "volume": d.volume,
                }
                for d in data
            ]
            frame = pd.DataFrame(records)
        except Exception as e:
            LOGGER.error(f"Failed to refresh recent {self.strategy_timeframe_minutes}min history for {symbol}: {e}")
            return
        frame = self._limit_to_recent_trading_sessions(
            frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"]),
            self.settings.premarket_intraday_history_days,
        )
        self.strategy_data[symbol] = (
            frame.set_index("timestamp")
        )
        self.catalog.write_historical_yearly(frame, symbol, str(self.strategy_timeframe_minutes))

    @staticmethod
    def _limit_to_recent_trading_sessions(frame: pd.DataFrame, session_count: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        session_count = max(int(session_count), 1)
        session_dates = list(pd.to_datetime(frame["timestamp"], format="mixed").dt.date.drop_duplicates())
        selected_dates = set(session_dates[-session_count:])
        trimmed = frame[pd.to_datetime(frame["timestamp"], format="mixed").dt.date.isin(selected_dates)].copy()
        return trimmed.sort_values("timestamp").reset_index(drop=True)

    def _recent_strategy_context(self, symbol: str) -> pd.DataFrame:
        frame = self.strategy_data[symbol]
        if frame.empty:
            return frame
        dates = pd.Index(frame.index.date).drop_duplicates()
        selected = set(dates[-self.settings.premarket_intraday_history_days :])
        mask = [trade_date in selected for trade_date in frame.index.date]
        return frame.loc[mask]

    def _fetch_seed_minute_data(self, symbol: str, days: int = 3) -> pd.DataFrame:
        to_date = self._today_ist()
        from_date = to_date - timedelta(days=days)
        
        try:
            # Use broker manager for failover-supported fetch
            # factory.py expects datetime objects
            data = self.broker_manager.get_historical_data(
                symbol=symbol,
                start_date=from_date,
                end_date=to_date,
                timeframe="1",
            )
            
            if not data:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                
            # Convert list[HistoricalData] to DataFrame
            records = [
                {
                    "timestamp": d.timestamp,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                    "volume": d.volume,
                }
                for d in data
            ]
            frame = pd.DataFrame(records)
            frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            return frame.set_index("timestamp")
            
        except Exception as e:
            LOGGER.error(f"Failed to fetch seed minute data for {symbol}: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def _load_existing_today_minute_file(self, symbol: str, current_date: date) -> None:
        path = self.catalog.live_bars_path(symbol, 1, current_date.isoformat())
        if not path.exists():
            return
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        if existing.empty:
            return
        existing = existing.set_index("timestamp")
        existing = _sanitize_intraday_minutes(
            existing,
            self._parse_clock(self.settings.market_start),
            self._parse_clock(self.settings.market_end),
        )
        combined = pd.concat([self.minute_data[symbol], existing]).sort_index()
        self.minute_data[symbol] = combined[~combined.index.duplicated(keep="last")]

    def _compute_previous_day_supertrend_seed(self, symbol: str) -> dict[str, float | int]:
        df = self.strategy_data[symbol]
        if df.empty:
            return {"direction": 0, "supertrend": 0.0}
        prev_day = self._today_ist() - timedelta(days=1)
        prev_day_df = df[df.index.date <= prev_day]
        timeframe_df = prev_day_df
        if len(timeframe_df) < self.supertrend_period + 1:
            return {"direction": 0, "supertrend": 0.0}
        st_df = calculate_supertrend(timeframe_df, period=self.supertrend_period, multiplier=self.supertrend_multiplier)
        if st_df.empty:
            return {"direction": 0, "supertrend": 0.0}
        last_row = st_df.iloc[-1]
        return {"direction": int(last_row["supertrend_direction"]), "supertrend": float(last_row["supertrend"])}

    def _prime_intraday_supertrend_state(self, symbol: str, current_date: date) -> None:
        df = self.strategy_data[symbol]
        if df.empty:
            return
        adjusted = get_gap_adjusted_data(df, self.supertrend_period, current_date)
        timeframe = adjusted
        if len(timeframe) < self.supertrend_period + 1:
            return
        st_df = calculate_supertrend(timeframe, period=self.supertrend_period, multiplier=self.supertrend_multiplier)
        if st_df.empty:
            return
        
        st_df_valid = _valid_supertrend_rows(st_df)
        yesterday_df = st_df_valid[st_df_valid.index.date < current_date]
        if not yesterday_df.empty:
            self.last_trend[symbol] = int(yesterday_df.iloc[-1]["supertrend_direction"])
            self.last_processed_trend_bar_time[symbol] = yesterday_df.index[-1]
        else:
            self.last_trend[symbol] = None
            self.last_processed_trend_bar_time[symbol] = None

        today_df = st_df_valid[st_df_valid.index.date == current_date]
        if today_df.empty:
            return

        LOGGER.info(
            "Priming intraday trend for %s. Replaying %d bars from today.",
            symbol,
            len(today_df)
        )
        for bar_time, _ in today_df.iterrows():
            slice_df = st_df.loc[:bar_time]
            self._check_trend_change(symbol, slice_df)


    def _fetch_recent_daily_context(self, symbol: str) -> list[dict[str, float | str]]:
        today = self._today_ist()
        
        try:
            data = self.broker_manager.get_historical_data(
                symbol=symbol,
                start_date=today - timedelta(days=14),
                end_date=today,
                timeframe="DAY",
            )
            if not data:
                return []
                
            records = [
                {
                    "timestamp": d.timestamp,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                    "volume": d.volume,
                }
                for d in data
            ]
            frame = pd.DataFrame(records)
        except Exception as e:
            LOGGER.error(f"Failed to fetch recent daily context for {symbol}: {e}")
            return []
        ordered = frame.sort_values("timestamp").copy()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], format="mixed")
        ordered = ordered[ordered["timestamp"].dt.date < today]
        if ordered.empty:
            return []
        return [
            {
                "date": pd.to_datetime(row["timestamp"], format="mixed").strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for _, row in ordered.tail(self.settings.premarket_daily_lookback_days).iterrows()
        ]

    def _build_premarket_message(
        self,
        symbol: str,
        recent_3min: pd.DataFrame,
        initial_direction: int,
        initial_supertrend: float,
    ) -> str:
        trend = "uptrend" if initial_direction == 1 else "downtrend" if initial_direction == -1 else "neutral"
        first_bar = recent_3min.index.min()
        last_bar = recent_3min.index.max()
        session_count = len(pd.Index(recent_3min.index.date).drop_duplicates())
        return (
            f"📘 <b>{self._short_symbol(symbol)} pre-market</b>\n"
            f"Loaded last {session_count} trading sessions of {self.strategy_timeframe_minutes}-min candles\n"
            f"Range: {pd.Timestamp(first_bar).strftime('%Y-%m-%d %H:%M')} -> {pd.Timestamp(last_bar).strftime('%Y-%m-%d %H:%M')}\n"
            f"Seeded 3-min Supertrend: {trend} | ST {initial_supertrend:.1f}\n"
            f"Live {self.strategy_timeframe_minutes}-min candles will append to this history before each Supertrend update.\n"
            f"Bot starts live processing at {self.settings.market_start}."
        )

    def _build_previous_day_supertrend_message(
        self,
        symbol: str,
        previous_day_seed: dict[str, float | int],
    ) -> str:
        direction = int(previous_day_seed["direction"])
        trend = "UP" if direction == 1 else "DOWN" if direction == -1 else "NEUTRAL"
        supertrend_value = float(previous_day_seed["supertrend"])
        quote = self._fetch_quote_snapshot(symbol)
        reference_price = quote.get("last_price")
        reference_line = "Opening reference price: unavailable"
        status_line = "Status: previous-day Supertrend reference only"
        previous_day = self.previous_day_levels.get(symbol)
        previous_day_levels_line = None
        if previous_day:
            previous_day_levels_line = (
                f"Previous day OHLC: "
                f"O ₹{float(previous_day['open']):.2f} | "
                f"H ₹{float(previous_day['high']):.2f} | "
                f"L ₹{float(previous_day['low']):.2f} | "
                f"C ₹{float(previous_day['close']):.2f}"
            )
        if reference_price is not None:
            reference_line = f"Opening reference price ({self.settings.market_premarket_check}): ₹{reference_price:.2f}"
            status_line = self._build_previous_day_supertrend_opening_status(
                reference_price=reference_price,
                supertrend_value=supertrend_value,
                seed_direction=direction,
            )
        lines = [
            f"🧭 <b>{self._short_symbol(symbol)} previous-day Supertrend</b>",
            f"Time: {self.settings.market_premarket_check}",
            reference_line,
        ]
        if previous_day_levels_line:
            lines.append(previous_day_levels_line)
        lines.extend(
            [
                f"Direction: <b>{trend}</b>",
                f"Supertrend: ₹{supertrend_value:.2f}",
                status_line,
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_previous_day_supertrend_opening_status(
        *,
        reference_price: float,
        supertrend_value: float,
        seed_direction: int,
    ) -> str:
        if supertrend_value <= 0:
            return "Status: previous-day Supertrend unavailable"
        if seed_direction == 1:
            if reference_price < supertrend_value:
                return "Status: 🔴 Opening break below previous-day Supertrend"
            return "Status: 🟢 Opening price holding above previous-day Supertrend"
        if seed_direction == -1:
            if reference_price > supertrend_value:
                return "Status: 🟢 Opening break above previous-day Supertrend"
            return "Status: 🔴 Opening price holding below previous-day Supertrend"
        if reference_price > supertrend_value:
            return "Status: Opening price above previous-day Supertrend"
        if reference_price < supertrend_value:
            return "Status: Opening price below previous-day Supertrend"
        return "Status: Opening price exactly at previous-day Supertrend"

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _level_proximity_threshold(reference_price: float) -> float:
        return max(10.0, reference_price * 0.001)

    def _fetch_quote_snapshot(self, symbol: str) -> dict[str, float | None]:
        get_quotes = getattr(self.broker, "get_quotes", None)
        if not callable(get_quotes):
            return {}
        try:
            payload = get_quotes([symbol]).get(symbol, {})
        except Exception:
            LOGGER.exception("Failed to fetch quote snapshot for %s", symbol)
            return {}
        return {
            "last_price": self._safe_float(payload.get("lp") or payload.get("last_price") or payload.get("ltp")),
            "open": self._safe_float(payload.get("open_price") or payload.get("open")),
            "high": self._safe_float(payload.get("high_price") or payload.get("high")),
            "low": self._safe_float(payload.get("low_price") or payload.get("low")),
            "prev_close": self._safe_float(payload.get("prev_close_price") or payload.get("prev_close")),
            "volume": self._safe_float(payload.get("volume") or payload.get("vol_traded_today") or payload.get("vtt")),
        }

    def _nearest_previous_day_level(self, symbol: str, price: float) -> tuple[str, float, float] | None:
        previous_day = self.previous_day_levels.get(symbol)
        if not previous_day:
            return None
        level_map = {
            "OPEN": float(previous_day["open"]),
            "HIGH": float(previous_day["high"]),
            "LOW": float(previous_day["low"]),
            "CLOSE": float(previous_day["close"]),
        }
        level_name, level_value = min(level_map.items(), key=lambda item: abs(price - item[1]))
        return level_name, level_value, abs(price - level_value)

    @staticmethod
    def _describe_gap(gap_points: float, gap_pct: float, threshold_pct: float) -> str:
        if gap_pct >= threshold_pct:
            return f"Gap Up {gap_points:+.2f} pts ({gap_pct:+.2f}%)"
        if gap_pct <= -threshold_pct:
            return f"Gap Down {gap_points:+.2f} pts ({gap_pct:+.2f}%)"
        return f"Flat/Small Gap {gap_points:+.2f} pts ({gap_pct:+.2f}%)"

    @staticmethod
    def _compute_volume_profile(frame: pd.DataFrame, price_step: float = 10.0) -> VolumeProfileSummary | None:
        if frame.empty or "close" not in frame.columns or "volume" not in frame.columns:
            return None
        profile_frame = frame.copy()
        profile_frame = profile_frame[pd.to_numeric(profile_frame["volume"], errors="coerce").fillna(0) > 0]
        if profile_frame.empty:
            return None
        bins = (profile_frame["close"].astype(float) / price_step).round() * price_step
        histogram = profile_frame.assign(price_bin=bins).groupby("price_bin")["volume"].sum().sort_index()
        if histogram.empty:
            return None
        total_volume = float(histogram.sum())
        prices = list(histogram.index.astype(float))
        volumes = [float(histogram.loc[price]) for price in prices]
        poc_index = max(range(len(prices)), key=lambda idx: volumes[idx])
        included = {poc_index}
        accumulated = volumes[poc_index]
        target = total_volume * 0.70
        low_index = poc_index - 1
        high_index = poc_index + 1
        while accumulated < target and (low_index >= 0 or high_index < len(prices)):
            low_volume = volumes[low_index] if low_index >= 0 else -1.0
            high_volume = volumes[high_index] if high_index < len(prices) else -1.0
            if high_volume >= low_volume:
                included.add(high_index)
                accumulated += volumes[high_index]
                high_index += 1
            else:
                included.add(low_index)
                accumulated += volumes[low_index]
                low_index -= 1
        selected_prices = [prices[idx] for idx in sorted(included)]
        return VolumeProfileSummary(
            point_of_control=prices[poc_index],
            value_area_low=min(selected_prices),
            value_area_high=max(selected_prices),
            total_volume=total_volume,
        )

    @staticmethod
    def _profile_location(price: float, profile: VolumeProfileSummary | None) -> str:
        if profile is None:
            return "Volume profile unavailable"
        if price < profile.value_area_low:
            return "Below value area"
        if price > profile.value_area_high:
            return "Above value area"
        if price < profile.point_of_control:
            return "Inside value area, below POC"
        if price > profile.point_of_control:
            return "Inside value area, above POC"
        return "At POC"

    def _classify_market_mood(
        self,
        *,
        gap_pct: float,
        seed_direction: int,
        profile: VolumeProfileSummary | None,
        reference_price: float,
    ) -> str:
        score = 0.0
        if gap_pct >= self.settings.major_gap_threshold_pct:
            score += 1.0
        elif gap_pct <= -self.settings.major_gap_threshold_pct:
            score -= 1.0
        if seed_direction == 1:
            score += 1.0
        elif seed_direction == -1:
            score -= 1.0
        if profile is not None:
            if reference_price > profile.value_area_high:
                score += 1.0
            elif reference_price < profile.value_area_low:
                score -= 1.0
            elif reference_price > profile.point_of_control:
                score += 0.5
            elif reference_price < profile.point_of_control:
                score -= 0.5
        if score >= 1.5:
            return "Bullish auction bias"
        if score <= -1.5:
            return "Bearish auction bias"
        return "Balanced / rotational bias"

    def _build_premarket_console_summary(
        self,
        symbol: str,
        previous_day_seed: dict[str, float | int],
    ) -> str:
        previous_day = self.previous_day_levels.get(symbol)
        if not previous_day:
            return f"Premarket context | {self._short_symbol(symbol)} | previous-day levels unavailable"

        quote = self._fetch_quote_snapshot(symbol)
        reference_price = quote.get("last_price")
        if reference_price is None:
            return (
                f"Premarket context | {self._short_symbol(symbol)} | no live quote available at {self.settings.market_premarket_check}. "
                f"Previous close: ₹{float(previous_day['close']):.2f}"
            )

        prev_close = float(previous_day["close"])
        gap_points = reference_price - prev_close
        gap_pct = (gap_points / prev_close) * 100.0 if prev_close else 0.0
        gap_label = self._describe_gap(gap_points, gap_pct, self.settings.major_gap_threshold_pct)
        nearest_level = self._nearest_previous_day_level(symbol, reference_price)
        proximity_line = "Previous-day location unavailable"
        if nearest_level is not None:
            level_name, level_value, distance = nearest_level
            threshold = self._level_proximity_threshold(reference_price)
            if distance <= threshold:
                proximity_line = f"Near previous day {level_name} at ₹{level_value:.2f} (distance {distance:.2f})"
            else:
                proximity_line = f"Closest previous-day level: {level_name} at ₹{level_value:.2f} (distance {distance:.2f})"

        profile_source = self.minute_data[symbol]
        profile_source = profile_source[profile_source.index.date < self._today_ist()] if not profile_source.empty else profile_source
        profile = self._compute_volume_profile(profile_source)
        profile_line = "Volume profile unavailable"
        if profile is not None:
            profile_line = (
                f"Volume profile: POC ₹{profile.point_of_control:.2f} | "
                f"VA ₹{profile.value_area_low:.2f} - ₹{profile.value_area_high:.2f} | "
                f"{self._profile_location(reference_price, profile)}"
            )

        seed_direction = int(previous_day_seed["direction"])
        seed_trend = "UP" if seed_direction == 1 else "DOWN" if seed_direction == -1 else "NEUTRAL"
        mood = self._classify_market_mood(
            gap_pct=gap_pct,
            seed_direction=seed_direction,
            profile=profile,
            reference_price=reference_price,
        )
        return "\n".join(
            [
                f"Premarket context | {self._short_symbol(symbol)} | {self.settings.market_premarket_check}",
                f"Reference price: ₹{reference_price:.2f}",
                f"Formation: {gap_label}",
                proximity_line,
                f"Previous-day Supertrend seed: {seed_trend} | ST ₹{float(previous_day_seed['supertrend']):.2f}",
                profile_line,
                f"Market mood: {mood}",
            ]
        )

    def _calculate_daily_zones(self, symbol: str) -> None:
        """Calculate Support/Resistance zones using previous day data."""
        try:
            # Fetch previous day intraday data (using seed data)
            df = self.minute_data[symbol]
            if df.empty:
                return
            
            # Get only the latest complete day
            dates = sorted(list(df.index.date))
            today = self._today_ist()
            prev_dates = [d for d in dates if d < today]
            if not prev_dates:
                return
            
            prev_date = prev_dates[-1]
            prev_day_data = df[df.index.date == prev_date]
            
            # Calculate Volume Profile
            vp_indicator = VolumeProfileIndicator(price_step=5.0)
            profile = vp_indicator.calculate(prev_day_data)
            
            self.daily_zones[symbol] = {
                "PDH": float(prev_day_data["high"].max()),
                "PDL": float(prev_day_data["low"].min()),
                "PDC": float(prev_day_data["close"].iloc[-1]),
                "POC": profile.point_of_control if profile else 0.0,
                "VAL": profile.value_area_low if profile else 0.0,
                "VAH": profile.value_area_high if profile else 0.0,
            }
            LOGGER.info(f"Calculated zones for {symbol}: {self.daily_zones[symbol]}")
        except Exception as e:
            LOGGER.error(f"Failed to calculate daily zones for {symbol}: {e}")

    def _send_morning_zones_summary(self, symbol: str) -> None:
        """Send a summary of S/R zones to Telegram."""
        zones = self.daily_zones.get(symbol)
        if not zones:
            return
        
        lines = [
            f"🏰 <b>{self._short_symbol(symbol)} S/R Zones</b>",
            f"<i>Levels calculated from previous day profile</i>\n",
            f"🔴 <b>Supply (Resistance):</b>",
            f" • PDH: ₹{zones['PDH']:.2f}",
            f" • VAH: ₹{zones['VAH']:.2f}",
            f"\n🔵 <b>Demand (Support):</b>",
            f" • PDL: ₹{zones['PDL']:.2f}",
            f" • VAL: ₹{zones['VAL']:.2f}",
            f"\n⚖️ <b>Neutral (Magnet):</b>",
            f" • POC: ₹{zones['POC']:.2f}",
            f" • PDC: ₹{zones['PDC']:.2f}",
        ]
        self.notifier.send("\n".join(lines))

    def _maybe_alert_zone_proximity(self, symbol: str, price: float, tick_time: datetime) -> None:
        """Alert if price enters a Supply/Demand zone buffer. (Delegated to Alert Agent)"""
        zones = self.daily_zones.get(symbol)
        if not zones:
            return
        
        for name, level in zones.items():
            if level <= 0: continue
            
            # Check proximity (e.g., 0.2%)
            dist_pct = abs(price - level) / level
            if dist_pct <= self.zone_buffer_pct:
                zone_type = "Support" if name in ["PDL", "VAL", "POC"] else "Resistance"
                self.alert_agent.alert_zone_proximity(symbol, name, zone_type, price, level, dist_pct, log_only=True)

    def _check_zone_confluence(self, symbol: str, direction: int, price: float) -> bool:
        """Verify if a trade direction aligns with zone spatial context."""
        zones = self.daily_zones.get(symbol)
        if not zones:
            return True # No zones? Allow trade (fallback)
        
        if direction == 1: # LONG
            # Must be near Support (PDL, VAL, POC)
            support_lvls = [zones["PDL"], zones["VAL"], zones["POC"]]
            return any(abs(price - l) / l <= self.zone_buffer_pct for l in support_lvls if l > 0)
        else: # SHORT
            # Must be near Resistance (PDH, VAH, POC)
            resist_lvls = [zones["PDH"], zones["VAH"], zones["POC"]]
            return any(abs(price - l) / l <= self.zone_buffer_pct for l in resist_lvls if l > 0)

    def _process_tick(self, symbol: str, tick: dict) -> None:
        if not _is_market_timestamp(
            tick["timestamp"],
            self._parse_clock(self.settings.market_start),
            self._parse_clock(self.settings.market_end),
        ):
            return
        self._check_gap_and_previous_day_levels(symbol, tick)
        
        # --- Autonomous Position Management ---
        self.position_manager.process_tick(symbol, float(tick["ltp"]), tick["timestamp"])

        price = float(tick["ltp"])
        tick_time = tick["timestamp"]

        # --- Zone and level proximity (index-only, delegates to SrPipeline) ---
        if self._is_index(symbol) and hasattr(self, "sr_pipeline"):
            self.sr_pipeline.on_tick(symbol, price, tick_time)

        # --- Legacy zone alert (kept for non-index symbols) ---
        if not self._is_index(symbol):
            self._maybe_alert_zone_proximity(symbol, price, tick_time)

        tick_minute = tick["timestamp"].replace(second=0, microsecond=0)
        current_minute = self.current_minute[symbol]
        if current_minute is None:
            self.current_minute[symbol] = tick_minute
        elif tick_minute > current_minute:
            self._flush_symbol_minute(symbol)
            self.current_minute[symbol] = tick_minute
        self.tick_buffer[symbol].append(tick)
        self.last_tick_price[symbol] = float(tick["ltp"])

        # Update live state for dashboard (non-blocking)
        self._live_state.update_tick(symbol, float(tick["ltp"]), float(tick.get("volume", 0)), tick["timestamp"])
        self._live_state.flush()

    def _check_gap_and_previous_day_levels(self, symbol: str, tick: dict) -> None:
        price = float(tick["ltp"])
        tick_time = pd.Timestamp(tick["timestamp"])
        if self.first_live_price[symbol] is None:
            self.first_live_price[symbol] = price
            self._maybe_alert_major_gap(symbol, price, tick_time)
        self._maybe_alert_previous_day_level_touch(symbol, price, tick_time)

    def _maybe_alert_major_gap(self, symbol: str, opening_price: float, tick_time: pd.Timestamp) -> None:
        if self.gap_alert_sent[symbol]:
            return
        previous_day = self.previous_day_levels.get(symbol)
        if not previous_day:
            return
        prev_close = float(previous_day["close"])
        if prev_close <= 0:
            return
        gap_points = opening_price - prev_close
        gap_pct = (gap_points / prev_close) * 100.0
        if abs(gap_pct) < self.settings.major_gap_threshold_pct:
            return
        self.gap_alert_sent[symbol] = True
        direction = "Gap Up" if gap_points > 0 else "Gap Down"
        color = "🟢" if gap_points > 0 else "🔴"
        self.notifier.send(
            f"{color} <b>{self._short_symbol(symbol)} {tick_time.strftime('%H:%M')}</b>\n"
            f"Major {direction}\n"
            f"Open/first live price: ₹{opening_price:.2f}\n"
            f"Previous close: ₹{prev_close:.2f}\n"
            f"Gap: {gap_points:+.2f} pts ({gap_pct:+.2f}%)"
        )

    def _maybe_alert_previous_day_level_touch(self, symbol: str, price: float, tick_time: pd.Timestamp) -> None:
        previous_day = self.previous_day_levels.get(symbol)
        if not previous_day:
            return
        previous_price = self.last_tick_price[symbol]
        level_map = {
            "high": float(previous_day["high"]),
            "low": float(previous_day["low"]),
            "close": float(previous_day["close"]),
        }
        for level_name, level_value in level_map.items():
            # Noise Reduction: Check last alert time for this specific level
            last_alert = self.last_level_alert_time[symbol].get(level_name)
            if last_alert:
                elapsed = (tick_time.to_pydatetime().replace(tzinfo=None) - last_alert).total_seconds()
                if elapsed < self.alert_debounce_seconds:
                    continue

            if self._price_touched_level(previous_price, price, level_value):
                # Delegate to Agent to log for EOD Analysis
                self.alert_agent.alert_retest(symbol, f"PrevDay_{level_name.upper()}", level_value, log_only=True)

    @staticmethod
    def _price_touched_level(previous_price: float | None, current_price: float, level: float) -> bool:
        if previous_price is None:
            return abs(current_price - level) < 1e-9
        lower = min(previous_price, current_price)
        upper = max(previous_price, current_price)
        return lower <= level <= upper

    def _flush_symbol_minute(self, symbol: str) -> None:
        """Complete the current minute bar and route it to the appropriate pipeline."""
        if not self.tick_buffer[symbol]:
            return
        buffered_ticks = self.tick_buffer[symbol]
        minute_bar = create_minute_bar(symbol, buffered_ticks, self.last_cumulative_volume[symbol])
        self.tick_buffer[symbol] = []
        if minute_bar is None:
            return

        # Backfill zero volumes from REST history
        minute_bar = self._patch_minute_volume_from_history(symbol, minute_bar)

        last_tick_volume = float(buffered_ticks[-1].get("volume", 0.0) or 0.0)
        if last_tick_volume > 0:
            self.last_cumulative_volume[symbol] = last_tick_volume

        # Update shared minute_data rolling window (BarAggregator also does this
        # via on_bar_complete, but _flush_symbol_minute is still called directly
        # from _flush_all_open_minutes at market close — keep both paths working)
        minute_df = minute_bar.to_frame().T.reset_index().rename(columns={"index": "timestamp"})
        trade_date = minute_bar.name.date()
        indexed = minute_df.set_index("timestamp")
        combined = pd.concat([self.minute_data[symbol], indexed]).sort_index()
        self.minute_data[symbol] = combined[~combined.index.duplicated(keep="last")].tail(1500)

        # --- Persist via BarStore (replaces direct catalog.append_frame + db_worker.enqueue) ---
        try:
            self.bar_store.save_1min_bar(symbol, minute_bar, trade_date)
        except Exception as exc:
            LOGGER.error("BarStore.save_1min_bar failed for %s: %s", symbol, exc)

        # --- Backfill recent zero-volume bars ---
        self._backfill_recent_live_minute_volumes(symbol, trade_date)

        # --- Strategy-timeframe file + analytics via BarStore ---
        try:
            updated_strategy = self.bar_store.update_strategy_file(
                symbol,
                self.minute_data[symbol],
                trade_date,
                resample_fn=resample_to_timeframe,
                completed_bars_fn=_completed_timeframe_bars,
                sanitize_fn=_sanitize_intraday_minutes,
                latest_session_minute_fn=_latest_session_minute,
            )
            if not updated_strategy.empty:
                self.strategy_data[symbol] = updated_strategy
                # Sync to IndexPipeline so it can query strategy_data via get_strategy_data()
                if self._is_index(symbol):
                    self.index_pipeline._strategy_data[symbol] = updated_strategy
                else:
                    self.fo_pipeline._strategy_data[symbol] = updated_strategy
        except Exception as exc:
            LOGGER.error("BarStore.update_strategy_file failed for %s: %s", symbol, exc)

        # --- Route to analytics pipeline ---
        self._on_bar_complete(symbol, minute_bar, self.minute_data)

        # --- Update live state dashboard ---
        self._live_state.update_minute_bar(
            symbol,
            open_=float(minute_bar.get("open", 0)),
            high=float(minute_bar.get("high", 0)),
            low=float(minute_bar.get("low", 0)),
            close=float(minute_bar.get("close", 0)),
            volume=int(minute_bar.get("volume", 0)),
        )
        self._live_state.update_health(db_stats=self.db_worker.stats(), ws_connected=self.ws_running)
        self._live_state.flush(force=True)

    def _on_bar_complete(
        self,
        symbol: str,
        bar: pd.Series,
        minute_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Central routing point for completed 1-min bars.

        Dispatches to IndexPipeline (full analytics) or FoPipeline (lightweight)
        based on symbol classification.  Also handles concerns that remain in the
        orchestrator: ORB detection, S/R channel touch, Gamma Blast, ICT stream,
        and Confirmed Strategy (until those are migrated to their own modules).
        """
        current_date = self._today_ist()

        if self._is_index(symbol):
            # Full index analytics — SuperTrend flip, RSI div, SMC, Sniper, MWPL
            try:
                self.index_pipeline.on_bar(symbol, bar, minute_data)
                # Sync strategy_data back to orchestrator for legacy method access
                self.strategy_data[symbol] = self.index_pipeline.get_strategy_data(symbol)
                if self.last_trend[symbol] != self.index_pipeline.get_last_trend(symbol):
                    self.last_trend[symbol] = self.index_pipeline.get_last_trend(symbol)
            except Exception as exc:
                LOGGER.error("IndexPipeline.on_bar failed for %s: %s", symbol, exc)

            # S/R, Gamma Blast — routed to dedicated pipelines
            current_date = self._today_ist()

            if hasattr(self, "sr_pipeline"):
                # Sync previous-day levels into sr_pipeline on each bar
                self.sr_pipeline.set_previous_day_levels(symbol, self.previous_day_levels.get(symbol))
                self.sr_pipeline.set_daily_zones(symbol, self.daily_zones.get(symbol, {}))
                self.sr_pipeline.on_bar(symbol, bar, minute_data)

            if hasattr(self, "gamma_pipeline"):
                self.gamma_pipeline.on_bar(symbol, bar, minute_data)
                # Sync open trade back to orchestrator for EOD summary
                open_trade = self.gamma_pipeline.get_open_trade(symbol)
                self.gamma_open_position[symbol] = (
                    open_trade.__dict__ if open_trade else None
                )

            # ICT stream (experimental, remains in orchestrator for now)
            if self.settings.enable_experimental_ict_stream:
                adjusted = get_gap_adjusted_data(
                    self.strategy_data.get(symbol, pd.DataFrame()).sort_index(),
                    self.supertrend_period,
                    current_date,
                )
                self._run_experimental_ict_stream(symbol, adjusted)

            latest_minute = pd.Timestamp(self.strategy_data[symbol].index.max()) \
                if symbol in self.strategy_data and not self.strategy_data[symbol].empty else None
            self._run_confirmed_strategy(
                symbol,
                get_gap_adjusted_data(
                    self.strategy_data.get(symbol, pd.DataFrame()).sort_index(),
                    self.supertrend_period,
                    current_date,
                ),
                latest_minute,
            )

        else:
            # Lightweight F&O analytics — RSI, volume surge, 3m to DB
            try:
                self.fo_pipeline.on_bar(symbol, bar, minute_data)
                # Sync strategy_data back to orchestrator
                self.strategy_data[symbol] = self.fo_pipeline.get_strategy_data(symbol)

                # Dispatch any pending alerts (only sent if enable_fo_telegram_alerts=True)
                pending = self.fo_pipeline.get_pending_alerts()
                for alert in pending:
                    self.notifier.send(alert.message)

            except Exception as exc:
                LOGGER.error("FoPipeline.on_bar failed for %s: %s", symbol, exc)


    def _flush_all_open_minutes(self) -> None:
        for symbol in self.symbols:
            self._flush_symbol_minute(symbol)

    def _check_sr_channel_touch(self, symbol: str, minute_bar: pd.Series) -> None:
        """Evaluate Support/Resistance channels on the 15-minute timeframe and send alerts."""
        df = self.minute_data.get(symbol)
        if df is None or df.empty:
            return
            
        try:
            # Generate 15-minute bars for SR detection
            df_15m = resample_to_timeframe(df, 15).reset_index()
            if len(df_15m) < 25:  # minimal requirement for pivot_period=10 is 2*10+2=22
                return
                
            snapshots = self.sr_detector.calculate(df_15m)
            if not snapshots:
                return
                
            latest_snap = snapshots[-1]
            channels = latest_snap.channels
            if not channels:
                return
                
            current_price = float(minute_bar.get("close", 0.0))
            if current_price == 0.0:
                return
                
            now = self._now_ist()
            last_alerts = self.last_sr_alert_time.get(symbol, {})
            
            # Use a tiny buffer around the channel to define a "touch" (e.g., 0.1% or exactly within channel)
            # The SRChannel is defined by bottom and top. 
            for ch in channels:
                buffer = current_price * 0.001 # 0.1% buffer
                if (ch.low - buffer) <= current_price <= (ch.high + buffer):
                    # Unique ID for this channel
                    channel_id = f"{ch.channel_type}_{ch.low:.1f}_{ch.high:.1f}"
                    last_time = last_alerts.get(channel_id)
                    
                    # Cooldown of 60 minutes for the exact same channel alert
                    if last_time is None or (now - last_time).total_seconds() > 3600:
                        self.last_sr_alert_time[symbol][channel_id] = now
                        
                        ch_type_str = ch.channel_type.capitalize()
                        msg = (
                            f"🔔 <b>{symbol} - 15m SR {ch_type_str} Touch</b>\n\n"
                            f"Price: ₹{current_price:.2f}\n"
                            f"Channel: ₹{ch.low:.2f} - ₹{ch.high:.2f}\n"
                            f"Time: {now.strftime('%H:%M:%S')}"
                        )
                        self.notifier.send(msg)
        except Exception as e:
            LOGGER.error(f"Failed to check SR channel touches for {symbol}: {e}")

    def _evaluate_gamma_blast(self, symbol: str) -> None:
        """Evaluate Gamma Blast strategy and manage active trades."""
        open_trade = self.gamma_open_position.get(symbol)
        if open_trade:
            self._manage_gamma_blast_trade(symbol, open_trade)
            return

        df = self.minute_data.get(symbol)
        if df is None or len(df) < 30:
            return
            
        now = self._now_ist()
        if not (dt_time(14, 30) <= now.time() <= dt_time(15, 5)):
            return

        df_eval = df.tail(100).copy()
        
        today_mask = df_eval.index.date == now.date()
        if not today_mask.any():
            return
            
        df_today = df_eval[today_mask]
        if not df_today.empty:
            typical_price = (df_today['high'] + df_today['low'] + df_today['close']) / 3
            df_eval.loc[today_mask, 'vwap'] = (typical_price * df_today['volume']).cumsum() / df_today['volume'].cumsum()
        
        delta = df_eval['close'].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        df_eval['rsi'] = 100 - (100 / (1 + rs))
        
        df_eval['volume_sma_20'] = df_eval['volume'].rolling(window=20).mean()
        
        oc_analysis = self.latest_oc_analysis.get(self._short_symbol(symbol))
        
        try:
            suggestion = self.gamma_agent.analyze_for_gamma_blast(
                symbol=symbol,
                df=df_eval.reset_index(),
                oc_analysis=oc_analysis
            )
        except Exception as e:
            LOGGER.error(f"GammaBlastAgent error for {symbol}: {e}")
            return
            
        if suggestion:
            LOGGER.info(f"🚀 GAMMA BLAST TRIGGERED: {symbol} - {suggestion.direction.name}")
            
            # Select ATM option from OC
            opt_symbol = "N/A"
            entry_premium = 0.0
            if oc_analysis and oc_analysis.chain_df is not None:
                atm_strike = round(suggestion.entry_zone_low / 50) * 50 if "NIFTY50" in symbol else round(suggestion.entry_zone_low / 100) * 100
                opt_type = "CE" if suggestion.direction == TradeDirection.CALL else "PE"
                mask = (oc_analysis.chain_df['strike'] == atm_strike) & (oc_analysis.chain_df['option_type'] == opt_type)
                if mask.any():
                    opt_row = oc_analysis.chain_df[mask].iloc[0]
                    opt_symbol = opt_row['symbol']
                    entry_premium = float(opt_row['ltp'])
            
            self.gamma_open_position[symbol] = {
                "symbol": symbol,
                "direction": suggestion.direction,
                "entry_time": now,
                "entry_spot": float(df_eval['close'].iloc[-1]),
                "option_symbol": opt_symbol,
                "entry_premium": entry_premium,
                "target_spot": float(suggestion.target),
                "stop_spot": float(suggestion.stop_loss),
                "reason": suggestion.narrative
            }
            
            msg = (
                f"💥 <b>0DTE GAMMA BLAST ALERT: {symbol}</b> 💥\n\n"
                f"Direction: <b>{suggestion.direction.name}</b>\n"
                f"Spot: ₹{df_eval['close'].iloc[-1]:.2f}\n"
                f"Target Spot: ₹{suggestion.target:.2f}\n"
                f"Stop Spot: ₹{suggestion.stop_loss:.2f}\n"
                f"Option: {opt_symbol} @ ₹{entry_premium:.2f}\n\n"
                f"<i>{suggestion.narrative}</i>"
            )
            self.notifier.send(msg)
            
    def _manage_gamma_blast_trade(self, symbol: str, trade: dict[str, object]) -> None:
        """Trailing and exit logic for active Gamma Blast trade."""
        now = self._now_ist()
        
        if now.time() >= dt_time(15, 12):
            self._close_gamma_blast_trade(symbol, trade, "EOD_CLOSE", now)
            return
            
        df = self.minute_data.get(symbol)
        if df is None or df.empty:
            return
            
        curr_spot = float(df['close'].iloc[-1])
        target_spot = float(trade["target_spot"])
        stop_spot = float(trade["stop_spot"])
        direction = trade["direction"]
        
        if direction == TradeDirection.CALL:
            if curr_spot >= target_spot:
                self._close_gamma_blast_trade(symbol, trade, "TARGET_HIT", now)
                return
            if curr_spot <= stop_spot:
                self._close_gamma_blast_trade(symbol, trade, "STOP_LOSS", now)
                return
        else:
            if curr_spot <= target_spot:
                self._close_gamma_blast_trade(symbol, trade, "TARGET_HIT", now)
                return
            if curr_spot >= stop_spot:
                self._close_gamma_blast_trade(symbol, trade, "STOP_LOSS", now)
                return
            
    def _close_gamma_blast_trade(self, symbol: str, trade: dict[str, object], reason: str, exit_time: datetime, exit_premium: float = 0.0) -> None:
        """Close an active gamma blast trade and record PnL."""
        entry_prem = float(trade["entry_premium"])
        
        if exit_premium == 0.0:
            oc = self.latest_oc_analysis.get(self._short_symbol(symbol))
            if oc and oc.chain_df is not None:
                opt_row = oc.chain_df[oc.chain_df['symbol'] == trade["option_symbol"]]
                if not opt_row.empty:
                    exit_premium = float(opt_row.iloc[0]['ltp'])
                    
        pnl = exit_premium - entry_prem
        pnl_pct = (pnl / entry_prem) * 100 if entry_prem > 0 else 0.0
        
        emoji = "🟢" if pnl > 0 else "🔴"
        
        direction_name = trade["direction"].name if hasattr(trade["direction"], "name") else str(trade["direction"])
        msg = (
            f"{emoji} <b>GAMMA BLAST CLOSED: {symbol}</b>\n\n"
            f"Option: {trade['option_symbol']}\n"
            f"Direction: {direction_name}\n"
            f"Reason: <b>{reason}</b>\n\n"
            f"Entry: ₹{entry_prem:.2f}\n"
            f"Exit: ₹{exit_premium:.2f}\n"
            f"PnL: <b>{pnl:+.2f} ({pnl_pct:+.1f}%)</b>"
        )
        self.notifier.send(msg)
        
        trade["exit_time"] = exit_time
        trade["exit_premium"] = exit_premium
        trade["pnl"] = pnl
        trade["pnl_pct"] = pnl_pct
        trade["exit_reason"] = reason
        
        self.gamma_trades[symbol].append(trade)
        self.gamma_open_position[symbol] = None

    def _run_supertrend(self, symbol: str) -> None:
        current_date = self._today_ist()
        timeframe = self.strategy_data[symbol].copy()
        if timeframe.empty:
            return
        timeframe = timeframe.sort_index()
        adjusted = get_gap_adjusted_data(timeframe, self.supertrend_period, current_date)
        if len(timeframe) < self.supertrend_period + 1:
            return
        st_df = calculate_supertrend(adjusted, period=self.supertrend_period, multiplier=self.supertrend_multiplier)
        if st_df.empty:
            return
        today_df = st_df[st_df.index.date == current_date]
        if not today_df.empty:
            self._check_trend_change(symbol, st_df)
            # Push supertrend state to live state for dashboard
            last_st = today_df.iloc[-1]
            self._live_state.update_supertrend(
                symbol,
                direction=int(last_st.get("supertrend_direction", 0)),
                level=float(last_st.get("supertrend", 0.0)),
            )
        
        # --- RSI Divergence ---
        rsi_df = self.rsi_divergence.calculate(adjusted)
        if not rsi_df.empty:
            today_rsi = rsi_df[rsi_df.index.date == current_date]
            if not today_rsi.empty:
                self._check_rsi_divergence(symbol, today_rsi)

        # --- 2. SMC & Technical Alerts (Delegated to Alert Agent) ---
        smc_signal = detect_smc_signal(adjusted)
        if smc_signal is not None:
            self.alert_agent.process_smc_signal(symbol, smc_signal)
        
        if not adjusted.empty:
            self.alert_agent.monitor_technical_extremes(symbol, adjusted)
            self.alert_agent.monitor_squeeze_breakout(symbol, adjusted)
            
            # --- 3. Early Morning Setups (Gap & Go / ORB) ---
            # Using synchronous wrapper or ensuring it runs in the right context
            # Since this is a callback, we handle the async call carefully
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    morning_setups = loop.run_until_complete(
                        self.early_morning_agent.scan_for_setups({symbol: adjusted})
                    )
                finally:
                    loop.close()
                
                for sugg in morning_setups:
                    if "GAP_AND_GO" in sugg.tags:
                        self.alert_agent.alert_gap_and_go(symbol, sugg.direction.value, sugg.entry_zone_high, sugg.stop_loss)
            except Exception as e:
                LOGGER.error(f"Early morning scan failed for {symbol}: {e}")

            # --- 4. Sniper Reversal Check (Forensic Absorption) ---
            sniper_sugg = self.sniper_agent.analyze_for_reversal(symbol, adjusted)
            if sniper_sugg:
                setup_type = "BOTTOM FISHING" if sniper_sugg.direction == TradeDirection.CALL else "TOP SNIPE"
                self.alert_agent.alert_sniper_reversal(symbol, setup_type, sniper_sugg.entry_zone_high, sniper_sugg.stop_loss, sniper_sugg.narrative)

            # --- MWPL Trigger Check ---
            clean_sym = symbol.replace("NSE:", "").replace("-EQ", "")
            if clean_sym in self.mwpl_setups.get("SQUEEZE", []) or clean_sym in self.mwpl_setups.get("UNWINDING", []):
                latest = adjusted.iloc[-1]
                prev = adjusted.iloc[-2] if len(adjusted) >= 2 else latest
                vwap = latest.get("vwap")
                if vwap and vwap > 0:
                    price = float(latest["close"])
                    prev_price = float(prev["close"])
                    setup_type = "SQUEEZE" if clean_sym in self.mwpl_setups["SQUEEZE"] else "UNWINDING"
                    
                    # Cross above VWAP for Squeeze, Below for Unwinding
                    if setup_type == "SQUEEZE" and prev_price <= vwap <= price:
                        data = self.mwpl_analyzer.get_mwpl_data()
                        pct = data[data['SYMBOL'] == clean_sym]['MWPL_PCT'].iloc[0] if not data.empty else 85.0
                        self.alert_agent.alert_mwpl_trigger(symbol, "SQUEEZE", price, pct)
                    elif setup_type == "UNWINDING" and prev_price >= vwap >= price:
                        data = self.mwpl_analyzer.get_mwpl_data()
                        pct = data[data['SYMBOL'] == clean_sym]['MWPL_PCT'].iloc[0] if not data.empty else 90.0
                        self.alert_agent.alert_mwpl_trigger(symbol, "UNWINDING", price, pct)

        if self.settings.enable_experimental_ict_stream:
            self._run_experimental_ict_stream(symbol, adjusted)
        latest_bar = pd.Timestamp(timeframe.index.max()) if not timeframe.empty else None
        self._run_confirmed_strategy(symbol, adjusted, latest_bar)

    def _check_trend_change(self, symbol: str, df: pd.DataFrame) -> None:
        valid_df = _valid_supertrend_rows(df)
        if valid_df.empty:
            return
        
        current_date = self._today_ist()
        today_df = valid_df[valid_df.index.date == current_date]
        if today_df.empty:
            return
            
        last_bar = today_df.iloc[-1]
        bar_time = today_df.index[-1]
        
        if self.last_processed_trend_bar_time[symbol] == bar_time:
            return
            
        current_trend = int(last_bar["supertrend_direction"])
        
        # Locate the last bar index in full valid_df to find previous trend (could be yesterday's last bar)
        try:
            import numpy as np
            idx_loc = valid_df.index.get_loc(bar_time)
            if isinstance(idx_loc, slice):
                idx_val = idx_loc.start
            elif isinstance(idx_loc, (np.ndarray, list)):
                idx_val = int(idx_loc[0])
            else:
                idx_val = int(idx_loc)
            previous_bar = valid_df.iloc[idx_val - 1] if idx_val > 0 else None
        except Exception:
            previous_bar = None
            
        previous_trend = int(previous_bar["supertrend_direction"]) if previous_bar is not None else self.last_trend[symbol]
        
        self._check_supertrend_touch(symbol, last_bar, bar_time)
        if self.last_signal_bar_time[symbol] != bar_time and previous_trend is not None and current_trend != previous_trend:
            close_price = float(last_bar["close"])
            supertrend_val = float(last_bar["supertrend"])
            if current_trend == 1 and close_price <= supertrend_val:
                self.last_trend[symbol] = current_trend
                self.last_processed_trend_bar_time[symbol] = bar_time
                return
            if current_trend == -1 and close_price >= supertrend_val:
                self.last_trend[symbol] = current_trend
                self.last_processed_trend_bar_time[symbol] = bar_time
                return
            self.last_signal_bar_time[symbol] = bar_time
            self._record_trend_flip(symbol, current_trend, bar_time, close_price, supertrend_val)
            action = "BUY CALL" if current_trend == 1 else "BUY PUT"
            color = "🟢" if current_trend == 1 else "🔴"

            # --- Zone Filter & Confluence ---
            has_zone_confluence = self._check_zone_confluence(symbol, current_trend, close_price)
            confluence_label = "💎 High Confluence (Near Zone)" if has_zone_confluence else "⚡ Standard Signal"

            # --- 15-Minute Supertrend Context ---
            st_15_dir_str = "UNKNOWN"
            try:
                full_data = self.strategy_data.get(symbol)
                if full_data is not None and not full_data.empty:
                    df_15m = resample_to_timeframe(full_data, 15)
                    st_15m = calculate_supertrend(df_15m, period=self.supertrend_period, multiplier=self.supertrend_multiplier)
                    if not st_15m.empty:
                        dir_val = st_15m.iloc[-1].get("supertrend_direction")
                        if pd.notna(dir_val):
                            st_15_dir_str = "UP" if dir_val == 1 else "DOWN"
            except Exception as e:
                LOGGER.warning(f"Failed to calculate 15m ST for trend alert: {e}")

            # --- Strike Price Context (CE + PE for 5 adjacent strikes) ---
            strike_info = ""
            clean_sym = symbol.replace("NSE:", "").replace("-INDEX", "").replace("BSE:", "")
            oc_analysis = self.latest_oc_analysis.get(clean_sym)
            
            current_oc_df = None
            if clean_sym in self.oc_analyzers:
                try:
                    current_oc_df = self.oc_analyzers[clean_sym].get_option_chain_df()
                except Exception:
                    pass
            if current_oc_df is None or current_oc_df.empty:
                current_oc_df = self.prev_oc_df.get(clean_sym)

            if current_oc_df is not None and not current_oc_df.empty:
                try:
                    # Derive ATM from OC analysis, or fall back to rounding close_price to nearest 50
                    if oc_analysis and oc_analysis.get("atm", 0):
                        atm = int(oc_analysis.get("atm", 0))
                    else:
                        step = 100 if close_price > 40000 else 50  # BankNifty uses 100, Nifty uses 50
                        atm = int(round(close_price / step) * step)

                    # Fetch 5 adjacent strikes for both CE and PE using current OC snapshot
                    ce_strikes = self._select_adjacent_option_chain_strikes(current_oc_df, atm, "CE", count=5)
                    pe_strikes = self._select_adjacent_option_chain_strikes(current_oc_df, atm, "PE", count=5)

                    # Load historical option chain data from the CSV file
                    date_str = bar_time.strftime("%Y%m%d")
                    history_path = self.settings.option_chain_data_dir / f"{clean_sym}_strikes_{date_str}.csv"
                    history_df = self._load_option_chain_history(history_path)

                    if not history_df.empty:
                        def _build_strike_lines(strikes: list, opt_type: str) -> list[str]:
                            lines = []
                            for s in strikes:
                                res = self._calculate_option_chain_strike_supertrend(history_df, s, opt_type)
                                if res:
                                    st_dir = res["direction"]
                                    st_icon = "🟢" if st_dir == 1 else "🔴"
                                    ltp = res["ltp"]
                                    st_lvl = res.get("st_level")
                                    atm_tag = " ◀ATM" if s == atm else ""
                                    st_str = f" ST:{st_lvl:.1f}" if st_lvl else ""
                                    lines.append(f"  {st_icon} <b>{s}{opt_type}</b> ₹{ltp:.1f}{st_str}{atm_tag}")
                            return lines

                        ce_lines = _build_strike_lines(ce_strikes, "CE")
                        pe_lines = _build_strike_lines(pe_strikes, "PE")

                        if ce_lines or pe_lines:
                            parts = [f"\n\n<b>OI Snapshot (ATM {atm}) — 5 Strikes Each</b>"]
                            if ce_lines:
                                parts.append(f"<b>CALL (CE):</b>\n" + "\n".join(ce_lines))
                            if pe_lines:
                                parts.append(f"<b>PUT (PE):</b>\n" + "\n".join(pe_lines))
                            parts.append("🟢=ST Bullish  🔴=ST Bearish")
                            strike_info = "\n".join(parts)
                except Exception as e:
                    LOGGER.warning(f"Failed to build strike info for trend alert: {e}")

            try:
                # ── Main channel message (disabled by default to keep channel clean) ──
                if getattr(self.settings, "enable_main_channel_supertrend_alerts", False):
                    main_msg = (
                        f"{color} <b>{self._short_symbol(symbol)} {bar_time.strftime('%H:%M')}</b>\n"
                        f"Direction changed to <b>{'UP' if current_trend == 1 else 'DOWN'}</b> ({self.strategy_timeframe_minutes}m)\n"
                        f"15m Trend: <b>{st_15_dir_str}</b>\n"
                        f"Close: ₹{close_price:.2f}\n"
                        f"Supertrend: ₹{supertrend_val:.2f}\n"
                        f"Confluence: <b>{confluence_label}</b>\n"
                        f"Action: <b>{action}</b>"
                        f"{strike_info}"
                    )
                    self.notifier.send(main_msg)

                # ── ST_CONFIRMED channel — rich message with strike ST directions ─
                confirmed_strike_block = strike_info  # reuse the CE/PE block built above
                if not confirmed_strike_block and current_oc_df is not None and not current_oc_df.empty:
                    # Attempt a wider 7-strike build specifically for the confirmed channel
                    try:
                        date_str_c = bar_time.strftime("%Y%m%d")
                        hist_path_c = self.settings.option_chain_data_dir / f"{clean_sym}_strikes_{date_str_c}.csv"
                        hist_c = self._load_option_chain_history(hist_path_c)
                        if not hist_c.empty:
                            ce7 = self._select_adjacent_option_chain_strikes(current_oc_df, atm, "CE", count=7)
                            pe7 = self._select_adjacent_option_chain_strikes(current_oc_df, atm, "PE", count=7)
                            ce7_lines = self._build_option_chain_strike_lines(hist_c, ce7, "CE", atm)
                            pe7_lines = self._build_option_chain_strike_lines(hist_c, pe7, "PE", atm)
                            if ce7_lines and pe7_lines:
                                confirmed_strike_block = (
                                    f"\n\n<b>Strike Supertrend (ATM {atm}) — 7 Strikes Each</b>\n"
                                    f"━━━━━━━━━━━━\n"
                                    f"<b>CALL (CE) Supertrend:</b>\n{ce7_lines}\n"
                                    f"━━━━━━━━━━━━\n"
                                    f"<b>PUT (PE) Supertrend:</b>\n{pe7_lines}\n"
                                    f"━━━━━━━━━━━━\n"
                                    f"🟢=ST Bullish  🔴=ST Bearish"
                                )
                    except Exception as _cse:
                        LOGGER.warning("Failed to build extended strike block for confirmed channel: %s", _cse)

                confirmed_msg = (
                    f"{color} <b>⚡ ST FLIP — {self._short_symbol(symbol)} {bar_time.strftime('%H:%M')}</b>\n"
                    f"Timeframe: <b>{self.strategy_timeframe_minutes}m Supertrend</b>\n"
                    f"New Direction: <b>{'🟢 UP (BULLISH)' if current_trend == 1 else '🔴 DOWN (BEARISH)'}</b>\n"
                    f"15m Trend Alignment: <b>{st_15_dir_str}</b>\n"
                    f"Close: ₹{close_price:.2f}  |  ST Level: ₹{supertrend_val:.2f}\n"
                    f"Confluence: <b>{confluence_label}</b>\n"
                    f"Suggested Action: <b>{action}</b>"
                    f"{confirmed_strike_block}"
                )
                if self._is_index(symbol):
                    self.confirmed_notifier.send(confirmed_msg)
            except Exception as e:
                LOGGER.exception(f"Failed to process trend change alert: {e}")

            LOGGER.info(
                "Supertrend crossover for %s at %s | trend=%s close=%.2f st=%.2f",
                symbol,
                bar_time,
                current_trend,
                close_price,
                supertrend_val,
            )
        self.last_trend[symbol] = current_trend
        self.last_processed_trend_bar_time[symbol] = bar_time

    def _check_rsi_divergence(self, symbol: str, df: pd.DataFrame) -> None:
        """Check for fresh RSI divergence signals and send alerts."""
        if df.empty: return
        
        last_bar = df.iloc[-1]
        bar_time = df.index[-1]
        
        now = datetime.now()
        last_alert_time = self.last_rsi_div_signal_time[symbol]
        
        if last_alert_time is not None and (now - last_alert_time).total_seconds() < 900:
            return
            
        signal = last_bar.get("signal", 0)
        if signal != 0:
            self.last_rsi_div_signal_time[symbol] = now
            
            direction = "BULLISH" if signal == 1 else "BEARISH"
            color = "💎" if signal == 1 else "🔥"
            action = "LONG Opportunity" if signal == 1 else "SHORT Opportunity"
            
            self.notifier.send(
                f"{color} <b>{self._short_symbol(symbol)} RSI Divergence</b>\n"
                f"Momentum Signal: <b>{direction}</b>\n"
                f"Time: {bar_time.strftime('%H:%M')}\n"
                f"Price: ₹{last_bar['close']:.2f}\n"
                f"Action: <b>{action}</b>"
            )
            LOGGER.info(f"RSI Divergence for {symbol} at {bar_time} | signal={direction}")

    def _check_supertrend_touch(self, symbol: str, last_bar: pd.Series, bar_time: pd.Timestamp) -> None:
        if self.last_touch_bar_time[symbol] == bar_time:
            return
        if "low" not in last_bar or "high" not in last_bar:
            return
        supertrend_val = float(last_bar["supertrend"])
        if not (float(last_bar["low"]) <= supertrend_val <= float(last_bar["high"])):
            return
        direction = int(last_bar["supertrend_direction"])
        close_price = float(last_bar["close"])
        self.last_touch_bar_time[symbol] = bar_time
        self.supertrend_touch_events[symbol].append(
            {
                "bar_time": bar_time,
                "direction": direction,
                "close": close_price,
                "supertrend": supertrend_val,
            }
        )
        self.alert_agent.alert_retest(symbol, "Supertrend", supertrend_val)
        LOGGER.info(
            "Supertrend touch for %s at %s | trend=%s close=%.2f st=%.2f",
            symbol,
            bar_time,
            direction,
            close_price,
            supertrend_val,
        )

    def _record_trend_flip(
        self,
        symbol: str,
        direction: int,
        bar_time: pd.Timestamp,
        close_price: float,
        supertrend_val: float,
    ) -> None:
        prior_run = self.current_trend_run[symbol]
        points_before_flip = None
        duration_minutes = None
        if prior_run is not None:
            points_before_flip = abs(close_price - prior_run.start_price)
            duration_minutes = int((bar_time - prior_run.start_bar_time).total_seconds() // 60)
        self.supertrend_flip_events[symbol].append(
            {
                "bar_time": bar_time,
                "direction": direction,
                "close": close_price,
                "supertrend": supertrend_val,
                "points_before_flip": points_before_flip,
                "duration_minutes": duration_minutes,
                "timeframe_minutes": self.strategy_timeframe_minutes,
            }
        )
        self.current_trend_run[symbol] = TrendRun(
            direction=direction,
            start_bar_time=bar_time,
            start_price=close_price,
        )

    def _run_confirmed_strategy(self, symbol: str, adjusted: pd.DataFrame, latest_minute: pd.Timestamp | None) -> None:
        if not self._is_index(symbol):
            return
        timeframe = resample_to_timeframe(adjusted, self.confirmed_timeframe_minutes)
        timeframe = _completed_timeframe_bars(timeframe, self.confirmed_timeframe_minutes, latest_minute)
        if len(timeframe) < self.supertrend_period + 2:
            return
        st_df = calculate_supertrend(timeframe, period=self.supertrend_period, multiplier=self.supertrend_multiplier)
        if st_df.empty:
            return
        data = _prepare_confirmed_strategy_frame(st_df)
        today_df = _valid_supertrend_rows(data[data.index.date == self._today_ist()])
        if today_df.empty:
            return
        last_bar = today_df.iloc[-1]
        bar_time = today_df.index[-1]
        if self.last_confirmed_bar_time[symbol] == bar_time:
            return

        position = self.confirmed_position[symbol]
        if position is not None:
            exit_payload = _confirmed_exit_payload(position, last_bar, bar_time)
            if exit_payload is not None:
                self.confirmed_notifier.send(
                    _build_confirmed_exit_message(symbol, exit_payload, self._short_symbol(symbol))
                )
                self.confirmed_position[symbol] = None
                position = None

        if position is None and bar_time.time() <= self.confirmed_entry_cutoff:
            entry_payload = _confirmed_entry_payload(today_df)
            if entry_payload is not None and pd.Timestamp(entry_payload["bar_time"]) == bar_time:
                self.confirmed_notifier.send(
                    _build_confirmed_entry_message(symbol, entry_payload, self._short_symbol(symbol))
                )
                self.confirmed_position[symbol] = entry_payload

        self.last_confirmed_bar_time[symbol] = bar_time

    def _update_intraday_yearly_3min_file(self, symbol: str, current_date: date) -> None:
        today_minutes = _sanitize_intraday_minutes(
            self.minute_data[symbol][self.minute_data[symbol].index.date == current_date],
            self._parse_clock(self.settings.market_start),
            self._parse_clock(self.settings.market_end),
        )
        if today_minutes.empty:
            return
        path = self.catalog.historical_year_path(symbol, str(self.strategy_timeframe_minutes), current_date.year)
        existing = pd.read_csv(path, parse_dates=["timestamp"]) if path.exists() else pd.DataFrame()
        if not existing.empty:
            existing = _sanitize_intraday_minutes(
                existing.set_index("timestamp"),
                self._parse_clock(self.settings.market_start),
                self._parse_clock(self.settings.market_end),
            ).reset_index()
        today_bars = resample_to_timeframe(today_minutes, self.strategy_timeframe_minutes)
        latest_minute = _latest_session_minute(today_minutes, current_date)
        today_bars = _completed_timeframe_bars(today_bars, self.strategy_timeframe_minutes, latest_minute).reset_index()
        if today_bars.empty:
            return
        merged = _merge_intraday_3min_bars(existing=existing, today_bars=today_bars)
        self.strategy_data[symbol] = merged.set_index("timestamp").sort_index().tail(1500)
        merged.to_csv(path, index=False)

    def _patch_minute_volume_from_history(self, symbol: str, minute_bar: pd.Series) -> pd.Series:
        volume = float(minute_bar.get("volume", 0.0) or 0.0)
        if volume > 0:
            return minute_bar
        minute_ts = pd.Timestamp(minute_bar.name)
        try:
            history = self._fetch_intraday_history_with_cache(symbol, minute_ts.date(), min_timestamp=minute_ts)
        except Exception:
            LOGGER.exception("Failed to backfill minute volume from FYERS history for %s at %s", symbol, minute_ts)
            return minute_bar
        if history.empty:
            return minute_bar
        match = history[history.index == minute_ts]
        if match.empty:
            return minute_bar
        patched = minute_bar.copy()
        patched["volume"] = float(match.iloc[-1]["volume"])
        return patched

    def _fetch_intraday_history_with_cache(
        self,
        symbol: str,
        trade_date: date,
        min_timestamp: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        cached = self.history_volume_cache.get(symbol, pd.DataFrame())
        if not cached.empty and cached.index.max().date() == trade_date:
            if min_timestamp is None or pd.Timestamp(cached.index.max()) >= pd.Timestamp(min_timestamp):
                return cached
        try:
            data = self.broker_manager.get_historical_data(
                symbol=symbol,
                start_date=trade_date,
                end_date=trade_date,
                timeframe="1",
            )
            if not data:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                
            records = [
                {
                    "timestamp": d.timestamp,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                    "volume": d.volume,
                }
                for d in data
            ]
            history = pd.DataFrame(records)
            history = history.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            history.set_index("timestamp", inplace=True)
        except Exception as e:
            LOGGER.error(f"Failed to fetch intraday history with cache for {symbol}: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            
        if history.empty:
            return history
        if "symbol" in history.columns:
            history = history.drop(columns=["symbol"])
        self.history_volume_cache[symbol] = history
        return history

    def _backfill_recent_live_minute_volumes(self, symbol: str, trade_date: date, lookback_minutes: int = 5) -> None:
        frame = self.minute_data[symbol]
        if frame.empty or "volume" not in frame.columns:
            return
        today_frame = frame[frame.index.date == trade_date]
        if today_frame.empty:
            return
        zero_mask = pd.to_numeric(today_frame["volume"], errors="coerce").fillna(0) <= 0
        recent_zero_rows = today_frame.loc[zero_mask].tail(lookback_minutes)
        if recent_zero_rows.empty:
            return
        latest_needed = pd.Timestamp(recent_zero_rows.index.max())
        try:
            history = self._fetch_intraday_history_with_cache(symbol, trade_date, min_timestamp=latest_needed)
        except Exception:
            LOGGER.exception("Failed recent volume backfill for %s on %s", symbol, trade_date)
            return
        if history.empty or "volume" not in history.columns:
            return
        updated = False
        for ts in recent_zero_rows.index:
            match = history[history.index == ts]
            if match.empty:
                continue
            volume = float(match.iloc[-1]["volume"] or 0.0)
            if volume <= 0:
                continue
            self.minute_data[symbol].loc[ts, "volume"] = volume
            updated = True
        if not updated:
            return
        today_path = self.catalog.live_bars_path(symbol, 1, trade_date.isoformat())
        payload = self.minute_data[symbol][self.minute_data[symbol].index.date == trade_date].reset_index()
        payload = payload.rename(columns={"index": "timestamp"}).sort_values("timestamp")
        if "symbol" in payload.columns:
            payload = payload.drop(columns=["symbol"])
        payload.to_csv(today_path, index=False)

    def _start_breakout_thread(self) -> None:
        if self.breakout_thread and self.breakout_thread.is_alive():
            return
            
        def _breakout_loop():
            LOGGER.info("Starting live Intraday Sector Breakout Screener loop (5m interval).")
            while self.ws_running and not self.shutdown:
                try:
                    now = self._now_ist()
                    # Only run between 9:15 and 15:30
                    if self._parse_clock(self.settings.market_start) <= now.time() < self._parse_clock(self.settings.market_end):
                        breakouts = self.breakout_screener.scan_for_breakouts(use_sector_filter=False)
                        if breakouts:
                            self._send_breakout_alerts(breakouts)
                except Exception as e:
                    LOGGER.error(f"Error in breakout screener loop: {e}")
                
                time.sleep(300)
                
        self.breakout_thread = threading.Thread(target=_breakout_loop, daemon=True)
        self.breakout_thread.start()

    def _send_breakout_alerts(self, breakouts: list[dict]):
        now = self._now_ist()
        alerts_to_send = []
        for b in breakouts:
            sym = b['symbol']
            
            # Suppress all FO stocks (non-indices) from Telegram ORB alerts
            if "-INDEX" not in sym:
                LOGGER.debug("Telegram ORB message suppressed for FO stock %s", sym)
                continue
                        
            if sym in self.last_breakout_alerts:
                if (now - self.last_breakout_alerts[sym]).total_seconds() < 3600:
                    continue
            alerts_to_send.append(b)
            self.last_breakout_alerts[sym] = now
            
        if not alerts_to_send or not self.notifier:
            return
            
        # Group by direction
        long_alerts = [b for b in alerts_to_send if b.get('direction', 'LONG') == 'LONG']
        short_alerts = [b for b in alerts_to_send if b.get('direction') == 'SHORT']
        
        # Helper to send in chunks of 5 alerts to respect Telegram's length limits
        def send_chunked(alerts, title_prefix, icon):
            chunk_size = 5
            for i in range(0, len(alerts), chunk_size):
                chunk = alerts[i:i+chunk_size]
                lines = [
                    f"{icon} <b>{title_prefix} (Part {i//chunk_size + 1})</b>",
                    f"Time: {now.strftime('%H:%M')}",
                    ""
                ]
                for b in chunk:
                    clean_sym = b['symbol'].replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                    lines.append(f"{icon} <b>{clean_sym}</b> ({b.get('sector', 'INDEX')})")
                    if b.get('direction', 'LONG') == 'LONG':
                        lines.append(f"Breakout Spot: ₹{b['close']:.2f} (Above ORB: ₹{b['orb_high']:.2f})")
                    else:
                        lines.append(f"Breakdown Spot: ₹{b['close']:.2f} (Below ORB: ₹{b['orb_low']:.2f})")
                    lines.append(f"Volume Surge: {b['volume']/b['vol_sma']:.1f}x (vs 20SMA {b['vol_sma']:.0f})")
                    lines.append("")
                
                self.notifier.send("\n".join(lines))
        
        if long_alerts:
            send_chunked(long_alerts, "[LIVE INDEX BREAKOUT — LONG]", "📈")
        if short_alerts:
            send_chunked(short_alerts, "[LIVE INDEX BREAKDOWN — SHORT]", "📉")
            
        LOGGER.info(f"Sent Live Breakout/Breakdown alerts for {len(alerts_to_send)} index symbols.")

    # ── Intraday Option Edge Pipeline ──────────────────────────────────────────

    def _start_option_edge_thread(self) -> None:
        if self.option_edge_thread and self.option_edge_thread.is_alive():
            return

        def _option_edge_loop():
            LOGGER.info("Starting Intraday Option Edge pipeline loop (5m interval).")
            while self.ws_running and not self.shutdown:
                try:
                    now = self._now_ist()
                    # Only run between 9:30 and 14:00 (no new entries after 2 PM)
                    if self._parse_clock("09:30") <= now.time() < self._parse_clock("14:00"):
                        alerts = self.option_edge_pipeline.scan()
                        if alerts:
                            self._send_option_edge_alerts(alerts)
                except Exception as e:
                    LOGGER.error(f"Error in Option Edge pipeline loop: {e}")

                time.sleep(300)  # 5 minutes

        self.option_edge_thread = threading.Thread(target=_option_edge_loop, daemon=True)
        self.option_edge_thread.start()

    def _send_option_edge_alerts(self, alerts) -> None:
        """Send Intraday Option Edge alerts via Telegram with deduplication."""
        now = self._now_ist()
        alerts_to_send = []

        for alert in alerts:
            sym = alert.symbol
            # Cooldown: don't re-alert the same stock within 60 minutes
            if sym in self.last_option_edge_alerts:
                if (now - self.last_option_edge_alerts[sym]).total_seconds() < 3600:
                    continue
            alerts_to_send.append(alert)
            self.last_option_edge_alerts[sym] = now

        if not alerts_to_send or not self.notifier:
            return

        for alert in alerts_to_send:
            try:
                msg = alert.format_telegram()
                self.notifier.send(msg)
            except Exception as e:
                LOGGER.error(f"Failed to send Option Edge alert for {alert.symbol}: {e}")

        LOGGER.info(f"Sent {len(alerts_to_send)} Intraday Option Edge alerts.")

    def _start_option_chain_thread(self) -> None:
        if self.oc_running:
            return
        self.oc_running = True
        self.oc_thread = threading.Thread(target=self._run_option_chain_loop, daemon=True)
        self.oc_thread.start()

    def _run_option_chain_loop(self) -> None:
        indices = [self._short_symbol(sym) for sym in self.settings.index_symbols]
        while self.oc_running and not self.shutdown:
            try:
                now = self._now_ist().replace(tzinfo=None)
                market_start = self._parse_clock(self.settings.market_start)
                market_end = self._parse_clock(self.settings.market_end)
                
                if market_start <= now.time() <= market_end:
                    for symbol in indices:
                        if self.shutdown: break
                        self._run_option_chain_cycle(symbol, now)
                        # Brief sleep between symbols
                        time.sleep(5)
            except Exception:
                LOGGER.exception("Option-chain cycle failed.")
            time.sleep(self.settings.option_chain_interval_seconds)

    def _run_option_chain_cycle(self, symbol: str, now: datetime) -> None:
        analyzer = self._get_option_chain_analyzer(symbol)
        if analyzer is None:
            return
        current_df = analyzer.get_option_chain_df()
        if current_df is None or current_df.empty:
            LOGGER.warning("Option-chain snapshot skipped for %s: empty dataframe.", symbol)
            return
        spot_price, vix = analyzer._fetch_spot_and_vix()
        self._save_option_chain_snapshot(symbol, now, current_df, spot_price, vix)
        analysis = analyzer.analyze(
            df=current_df,
            spot_price=spot_price,
            vix=vix,
            prev_df=self.prev_oc_df.get(symbol),
        )
        if analysis:
            self.latest_oc_analysis[symbol] = analysis
            # 1. Alert on direction flip (Market Nature)
            self._maybe_alert_option_chain_direction_change(symbol, analysis, spot_price, now, current_df)
            
            # 2. Monitor for sudden forensics shifts (VOI, PCR, IV Skew)
            full_symbol = f"BSE:{symbol}-INDEX" if symbol == "SENSEX" else f"NSE:{symbol}-INDEX"
            self.alert_agent.monitor_option_chain_changes(full_symbol, analysis)

            # 3. Dynamic Differential Data Change Forensics (Delta OI, Max Pain drift, Traps)
            try:
                self.oc_monitor_agent.process_snapshot(
                    symbol=full_symbol,
                    oc_df=current_df,
                    spot_price=spot_price or 0.0,
                    timestamp=now,
                    expiry=getattr(analyzer, "nearest_expiry", None),
                )
            except Exception as _oc_err:
                LOGGER.warning("OptionChainMonitorAgent processing failed for %s: %s", full_symbol, _oc_err)

            # Expiry Max Pain Alert at 2 PM (14:00 IST)
            if symbol in ["NIFTY50", "SENSEX"]:
                today_str = now.strftime("%d-%m-%Y")
                nearest_expiry = getattr(analyzer, "nearest_expiry", None)
                if nearest_expiry == today_str:
                    if now.time() >= dt_time(14, 0):
                        if self.expiry_max_pain_sent.get(symbol) != now.date():
                            max_pain = analysis.max_pain if analysis else None
                            if max_pain:
                                spot_str = f"₹{spot_price:.2f}" if spot_price else "N/A"
                                msg = (
                                    f"🎯 <b>{symbol} EXPIRY MAX PAIN</b>\n\n"
                                    f"<b>Expiry Date:</b> {nearest_expiry}\n"
                                    f"<b>Time:</b> {now.strftime('%H:%M:%S')}\n"
                                    f"<b>Spot Price:</b> {spot_str}\n"
                                    f"<b>Max Pain Strike:</b> ₹{max_pain:.1f}\n"
                                )
                                self.notifier.send(msg)
                                LOGGER.info(f"Sent expiry Max Pain alert for {symbol}: Max Pain = {max_pain}")
                                self.expiry_max_pain_sent[symbol] = now.date()
            
        self.prev_oc_df[symbol] = current_df.copy()

    def _get_option_chain_analyzer(self, symbol: str):
        if symbol in self.oc_analyzers:
            analyzer = self.oc_analyzers[symbol]
            analyzer.fyers = self.broker.fyers
            return analyzer
        try:
            from trade_system.domains.analysis.application.analysis.option_chain_analyzer import OptionChainAnalyzer
        except Exception:
            LOGGER.exception("Failed to import OptionChainAnalyzer.")
            return None
        self.oc_analyzers[symbol] = OptionChainAnalyzer(fyers_client=self.broker.fyers, symbol=symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", ""), strike_count=20)
        return self.oc_analyzers[symbol]

    def _save_option_chain_snapshot(self, symbol: str, now: datetime, current_df: pd.DataFrame, spot_price: float | None, vix: float | None) -> None:
        date_str = now.strftime("%Y%m%d")
        oc_path = self.settings.option_chain_data_dir / f"{symbol}_strikes_{date_str}.csv"

        snap = current_df.copy()
        snap.rename(columns={"oi": "open_interest"}, inplace=True)
        snap.insert(0, "timestamp", now.strftime("%Y-%m-%d %H:%M:%S"))
        snap.insert(1, "spot_price", spot_price if spot_price else 0)
        self._append_option_chain_snapshot_csv(oc_path, snap)

        if vix is not None:
            vix_path = self.settings.option_chain_data_dir / f"VIX_{date_str}.csv"
            vix_df = pd.DataFrame([{"timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "vix": vix}])
            self._append_option_chain_snapshot_csv(vix_path, vix_df)

        # Save to Database
        try:
            with Session(self.engine) as session:
                # underlying symbol is now dynamic
                full_symbol = f"BSE:{symbol}-INDEX" if symbol == "SENSEX" else f"NSE:{symbol}-INDEX"
                db_records = current_df.to_dict('records')
                from trade_system.domains.market_data.infrastructure.database.repository import save_option_chain_batch
                save_option_chain_batch(session, full_symbol, now, db_records)
        except Exception as e:
            LOGGER.error(f"Failed to save option chain snapshot to DB for {symbol}: {e}")

        vix_path = self.settings.option_chain_data_dir / f"VIX_{symbol}_{date_str}.csv"
        vix_row = pd.DataFrame(
            [
                {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "vix": vix if vix else 0,
                    "spot_price": spot_price if spot_price else 0,
                }
            ]
        )
        write_vix_header = not vix_path.exists()
        vix_row.to_csv(vix_path, mode="a", header=write_vix_header, index=False)
        LOGGER.info("Option-chain snapshot saved to %s", oc_path)

    def _append_option_chain_snapshot_csv(self, path: Path, snap: pd.DataFrame) -> None:
        expected_columns = list(snap.columns)
        if not path.exists():
            snap.to_csv(path, index=False)
            return
        try:
            existing = pd.read_csv(path, parse_dates=["timestamp"])
            for column in expected_columns:
                if column not in existing.columns:
                    existing[column] = 0.0 if column in {"delta", "gamma", "theta", "vega", "rho"} else pd.NA
            existing = existing[expected_columns]
        except Exception:
            existing = self._load_option_chain_snapshot_csv_robust(path, expected_columns)
        merged = pd.concat([existing, snap[expected_columns]], ignore_index=True)
        merged.to_csv(path, index=False)

    @staticmethod
    def _load_option_chain_snapshot_csv_robust(path: Path, expected_columns: list[str]) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            try:
                original_header = next(reader)
            except StopIteration:
                return pd.DataFrame(columns=expected_columns)
            for row in reader:
                if not row:
                    continue
                if len(row) == len(expected_columns):
                    row_header = expected_columns
                else:
                    row_header = original_header
                payload = dict(zip(row_header, row))
                normalized = {column: payload.get(column, pd.NA) for column in expected_columns}
                records.append(normalized)
        if not records:
            return pd.DataFrame(columns=expected_columns)
        return pd.DataFrame(records, columns=expected_columns)

    def _maybe_alert_option_chain_direction_change(
        self,
        symbol: str,
        analysis: dict,
        spot_price: float | None,
        timestamp: datetime,
        current_df: pd.DataFrame,
    ) -> None:
        direction = self._extract_option_chain_direction(analysis)
        if direction is None:
            self.last_option_chain_direction[symbol] = None
            return
        if direction == self.last_option_chain_direction.get(symbol):
            return
        if self._send_option_chain_strike_summary(symbol, direction, spot_price, timestamp, current_df, analysis):
            self.last_option_chain_direction[symbol] = direction

    def _extract_option_chain_direction(self, analysis: dict) -> int | None:
        label = (analysis.get("market_nature") or {}).get("label")
        if not label:
            return None
        label = label.upper()
        if "BULL" in label:
            return 1
        if "BEAR" in label:
            return -1
        return None

    def _send_option_chain_strike_summary(
        self,
        symbol: str,
        direction: int,
        spot_price: float | None,
        timestamp: datetime,
        current_df: pd.DataFrame,
        analysis: dict,
    ) -> bool:
        if spot_price is None or spot_price <= 0:
            LOGGER.warning("OC trend flip detected for %s but spot price missing; skipping strike breakdown.", symbol)
            return False
        date_str = timestamp.strftime("%Y%m%d")
        history_path = self.settings.option_chain_data_dir / f"{symbol}_strikes_{date_str}.csv"
        history = self._load_option_chain_history(history_path)
        if history.empty:
            LOGGER.warning("OC strike history missing (%s) for %s; cannot compute strike Supertrend.", history_path, symbol)
            return False


        atm = int(round(spot_price / 50.0) * 50)
        ce_strikes = self._select_adjacent_option_chain_strikes(
            current_df,
            atm,
            "CE",
            count=self.settings.option_chain_adjacent_strikes,
        )
        pe_strikes = self._select_adjacent_option_chain_strikes(
            current_df,
            atm,
            "PE",
            count=self.settings.option_chain_adjacent_strikes,
        )
        if not ce_strikes and not pe_strikes:
            LOGGER.warning("No local strikes found around ATM %s for CE/PE strike summary", atm)
            return False

        ce_lines = self._build_option_chain_strike_lines(history, ce_strikes, "CE", atm)
        pe_lines = self._build_option_chain_strike_lines(history, pe_strikes, "PE", atm)

        if ce_lines is None or pe_lines is None:
            LOGGER.info("Skipping OC strike summary due to insufficient data for CE or PE strikes.")
            return False

        label = (analysis.get("market_nature") or {}).get("label", "OC direction")
        color = "🟢" if direction == 1 else "🔴"
        msg = (
            f"{color} <b>OC Direction Flip</b> {timestamp.strftime('%H:%M')}\n"
            f"{label} — spot ₹{spot_price:,.1f}\n"
            f"ATM {atm} | {self.settings.option_chain_adjacent_strikes} adjacent strikes each side\n"
            f"━━━━━━━━━━━━━━\n"
            f"<b>CE Supertrend</b>\n{ce_lines}\n"
            f"━━━━━━━━━━━━━━\n"
            f"<b>PE Supertrend</b>\n{pe_lines}\n"
            f"━━━━━━━━━━━━━━\n"
            f"🟢 = ST bullish (buy/hold)  🔴 = ST bearish (avoid/sell)"
        )
        self.notifier.send(msg)
        LOGGER.info(
            "OC strike ST summary sent | direction=%s label=%s ce_strikes=%s pe_strikes=%s",
            direction,
            label,
            ce_strikes,
            pe_strikes,
        )
        return True

    def _build_option_chain_strike_lines(
        self,
        history: pd.DataFrame,
        strikes: list[int],
        option_type: str,
        atm: int,
    ) -> str | None:
        if not strikes:
            return None
        lines = []
        for strike in strikes:
            res = self._calculate_option_chain_strike_supertrend(history, strike, option_type)
            if res is None:
                return None
            st_dir = res["direction"]
            st_lvl = res["st_level"]
            ltp = res["ltp"]
            st_color = "🟢" if st_dir == 1 else "🔴"
            st_str = f"ST {st_lvl:.1f}" if st_lvl is not None else "ST —"
            atm_tag = " ← ATM" if strike == atm else ""
            lines.append(f"  {st_color} <b>{strike}{option_type}</b>  LTP {ltp:.1f}  {st_str}{atm_tag}")
        return "\n".join(lines)

    def _load_option_chain_history(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if "open_interest" in df.columns and "oi" not in df.columns:
                df = df.rename(columns={"open_interest": "oi"})
            return df
        except Exception as exc:
            LOGGER.warning("Failed to read OC history %s: %s", path, exc)
            return pd.DataFrame()

    def _load_previous_day_option_chain_snapshot(self, symbol: str, timestamp: datetime) -> pd.DataFrame:
        current_tag = timestamp.strftime("%Y%m%d")
        clean_sym = symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "")
        candidates = sorted(self.settings.option_chain_data_dir.glob(f"{clean_sym}_strikes_*.csv"))
        previous_files = [path for path in candidates if path.stem.split("_")[-1] < current_tag]
        if not previous_files:
            return pd.DataFrame()
        history = self._load_option_chain_history(previous_files[-1])
        if history.empty or "timestamp" not in history.columns:
            return pd.DataFrame()
        latest_ts = pd.to_datetime(history["timestamp"], format="mixed").max()
        snapshot = history[pd.to_datetime(history["timestamp"], format="mixed") == latest_ts].copy()
        return snapshot.reset_index(drop=True)

    def _select_adjacent_option_chain_strikes(
        self,
        current_df: pd.DataFrame,
        atm: int,
        option_type: str,
        count: int = 7,
    ) -> list[int]:
        strikes = current_df.loc[current_df['option_type'].str.upper() == option_type.upper(), 'strike']
        strikes = strikes.dropna().astype(float)
        if strikes.empty:
            return []
        strike_set = {int(round(float(s))) for s in strikes}
        strike_set.add(atm)
        ordered = sorted(strike_set, key=lambda strike: (abs(strike - atm), strike))
        return ordered[:count]

    def _calculate_option_chain_strike_supertrend(
        self,
        history_df: pd.DataFrame,
        strike: int,
        option_type: str,
    ) -> dict | None:
        subset = history_df.copy()
        subset = subset[subset['option_type'].str.upper() == option_type.upper()]
        subset = subset[subset['strike'].round().astype(int) == strike]
        if subset.empty:
            return None
        subset = subset.sort_values('timestamp').reset_index(drop=True)
        prices = subset['ltp'].astype(float).values
        if len(prices) < self.supertrend_period + 2:
            return None
        opens = prices[:-1]
        closes = prices[1:]
        highs = np.maximum(opens, closes) * 1.001
        lows = np.minimum(opens, closes) * 0.999
        
        # Pass actual timestamps to ensure data continuity gap checks work
        timestamps = subset['timestamp'].iloc[1:].values
        ohlc = pd.DataFrame({
            'open': opens, 
            'high': highs, 
            'low': lows, 
            'close': closes,
            'timestamp': timestamps
        }, index=timestamps)
        
        st_df = calculate_supertrend(ohlc, period=self.supertrend_period, multiplier=self.supertrend_multiplier)
        if st_df is None or st_df.empty:
            return None
        last = st_df.iloc[-1]
        return {
            'strike': strike,
            'option_type': option_type,
            'direction': int(last['supertrend_direction']),
            'st_level': float(last['supertrend']),
            'ltp': float(last['close']),
        }

    def _run_experimental_ict_stream(self, symbol: str, adjusted: pd.DataFrame) -> None:
        history = adjusted.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in history.columns:
            return
        history = history.sort_values("timestamp").reset_index(drop=True)
        last_processed = self.ict_last_processed_bar_time[symbol]

        for idx in range(len(history)):
            ts = pd.Timestamp(history.iloc[idx]["timestamp"])
            if last_processed is not None and ts <= last_processed:
                continue
            window = history.iloc[: idx + 1].copy()
            self._process_ict_bar(symbol, window)
            self.ict_last_processed_bar_time[symbol] = ts

    def _process_ict_bar(self, symbol: str, window: pd.DataFrame) -> None:
        if window.empty:
            return
        row = window.iloc[-1]
        ts = pd.Timestamp(row["timestamp"])
        active_fvgs = self.ict_fvg[symbol].update(window)
        _ = active_fvgs
        self.ict_ob[symbol].update(window)
        liq_snapshot = self.ict_liq[symbol].update(window)
        ms_snapshot = self.ict_ms[symbol].update(window)

        price = float(row["close"])
        st_dir = int(row["supertrend_direction"]) if pd.notna(row.get("supertrend_direction")) else 0
        bull_fvg = self.ict_fvg[symbol].in_zone(price, "BFVG")
        bear_fvg = self.ict_fvg[symbol].in_zone(price, "BKFVG")
        bull_ob = self.ict_ob[symbol].in_zone(price, 1)
        bear_ob = self.ict_ob[symbol].in_zone(price, -1)
        sweep = liq_snapshot["swept"]
        bullish_structure = bool(ms_snapshot["bos"] and ms_snapshot["bos"]["type"] == "BOS_UP") or bool(
            ms_snapshot["choch"] and ms_snapshot["choch"]["type"] == "CHOCH_UP"
        ) or int(ms_snapshot["trend"]) == 1
        bearish_structure = bool(ms_snapshot["bos"] and ms_snapshot["bos"]["type"] == "BOS_DOWN") or bool(
            ms_snapshot["choch"] and ms_snapshot["choch"]["type"] == "CHOCH_DOWN"
        ) or int(ms_snapshot["trend"]) == -1

        if ts.date() != self._today_ist():
            return

        open_position = self.ict_open_position[symbol]
        if open_position is not None:
            self.ict_rules._update_excursions(open_position, row)
            closed = self.ict_rules._check_exit(open_position, row, ts, sweep, bullish_structure, bearish_structure)
            if closed is not None:
                self.ict_open_position[symbol] = None
                trade = IctLiveTrade(
                    direction=1 if closed["direction"] == "LONG" else -1,
                    entry_time=pd.Timestamp(closed["entry_time"]),
                    entry_price=float(closed["entry_price"]),
                    stop_loss=float(closed["stop_loss"]),
                    take_profit=float(closed["take_profit"]),
                    exit_time=pd.Timestamp(closed["exit_time"]),
                    exit_price=float(closed["exit_price"]),
                    exit_reason=str(closed["exit_reason"]),
                    trigger=str(closed["trigger"]),
                    confluence_score=int(closed["confluence_score"]),
                    points_captured=float(closed["points_captured"]),
                    risk_points=float(closed["risk_points"]),
                    r_multiple=float(closed["r_multiple"]),
                )
                self.ict_trades[symbol].append(trade)
                self.notifier.send(self._build_ict_exit_message(symbol, trade))

        if self.ict_open_position[symbol] is not None:
            return
        if ts.time().hour > 14 or (ts.time().hour == 14 and ts.time().minute > 45):
            return

        if sweep and int(sweep["direction"]) == 1:
            score = int(bullish_structure) + int(bull_fvg is not None) + int(bull_ob is not None) + int(st_dir == 1)
            if score >= 2:
                position = self.ict_rules._build_position(
                    direction=1,
                    row=row,
                    ts=ts,
                    confluence_score=score,
                    trigger="SSL_SWEEP",
                    sweep=sweep,
                    fvg_zone=bull_fvg,
                    ob_zone=bull_ob,
                    target_level=self.ict_liq[symbol].nearest_bsl(price),
                )
                if position is not None:
                    self.ict_open_position[symbol] = position
                    self.ict_signal_events[symbol].append({"time": ts, "direction": 1, "score": score, "trigger": "SSL_SWEEP"})
                    self.notifier.send(self._build_ict_entry_message(symbol, position))
        elif sweep and int(sweep["direction"]) == -1:
            score = int(bearish_structure) + int(bear_fvg is not None) + int(bear_ob is not None) + int(st_dir == -1)
            if score >= 2:
                position = self.ict_rules._build_position(
                    direction=-1,
                    row=row,
                    ts=ts,
                    confluence_score=score,
                    trigger="BSL_SWEEP",
                    sweep=sweep,
                    fvg_zone=bear_fvg,
                    ob_zone=bear_ob,
                    target_level=self.ict_liq[symbol].nearest_ssl(price),
                )
                if position is not None:
                    self.ict_open_position[symbol] = position
                    self.ict_signal_events[symbol].append({"time": ts, "direction": -1, "score": score, "trigger": "BSL_SWEEP"})
                    self.notifier.send(self._build_ict_entry_message(symbol, position))

    def _build_ict_entry_message(self, symbol: str, position: dict[str, object]) -> str:
        direction = "LONG" if int(position["direction"]) == 1 else "SHORT"
        return (
            f"🧪 <b>ICT Experimental Entry</b>\n"
            f"{self._short_symbol(symbol)} {pd.Timestamp(position['entry_time']).strftime('%H:%M')} | {direction}\n"
            f"Entry: ₹{float(position['entry_price']):.2f}\n"
            f"SL: ₹{float(position['stop_loss']):.2f}\n"
            f"Target: ₹{float(position['take_profit']):.2f}\n"
            f"Trigger: {position['trigger']} | Score: {int(position['confluence_score'])}"
        )

    def _build_ict_exit_message(self, symbol: str, trade: IctLiveTrade) -> str:
        direction = "LONG" if trade.direction == 1 else "SHORT"
        return (
            f"🧪 <b>ICT Experimental Exit</b>\n"
            f"{self._short_symbol(symbol)} {trade.exit_time.strftime('%H:%M')} | {direction}\n"
            f"Exit: ₹{trade.exit_price:.2f} | Reason: {trade.exit_reason}\n"
            f"P&L: {trade.points_captured:+.2f} pts | R: {trade.r_multiple:+.2f}"
        )

    def _build_ict_eod_lines(self, symbol: str) -> list[str]:
        trades = self.ict_trades[symbol]
        if not trades:
            return [f"\n<b>{self._short_symbol(symbol)} ICT experimental</b>\nTrades: 0"]
        wins = [trade for trade in trades if trade.points_captured > 0]
        losses = [trade for trade in trades if trade.points_captured < 0]
        net_points = sum(trade.points_captured for trade in trades)
        avg_r = sum(trade.r_multiple for trade in trades) / len(trades)
        target_hits = sum(1 for trade in trades if trade.exit_reason == "TARGET_HIT")
        sl_hits = sum(1 for trade in trades if trade.exit_reason == "SL_HIT")
        opposite_exits = sum(1 for trade in trades if trade.exit_reason == "OPPOSITE_SIGNAL")
        eod_exits = sum(1 for trade in trades if trade.exit_reason == "EOD_SQUARE_OFF")
        open_position = self.ict_open_position[symbol]
        lines = [
            f"\n<b>{self._short_symbol(symbol)} ICT experimental</b>",
            f"Trades: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)}",
            f"Net points: {net_points:+.2f} | Avg R: {avg_r:+.2f}",
            f"Target hits: {target_hits} | SL hits: {sl_hits} | Opposite exits: {opposite_exits} | EOD exits: {eod_exits}",
        ]
        if open_position is not None:
            lines.append(
                f"Open position at close: {'LONG' if int(open_position['direction']) == 1 else 'SHORT'} "
                f"from ₹{float(open_position['entry_price']):.2f}"
            )
        return lines



    def _refresh_token_via_totp(self) -> bool:
        script_path = Path(__file__).resolve().parents[4] / "scripts" / "authenticate_fyers_totp.py"
        if not script_path.exists():
            LOGGER.error("Cannot refresh FYERS token because %s is missing.", script_path)
            return False
        try:
            root_path = Path(__file__).resolve().parents[4]
            subprocess.run([sys.executable, str(script_path)], cwd=root_path, check=True)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("FYERS TOTP refresh script failed: %s", exc)
            return False
        new_settings = Settings.load()
        new_token = new_settings.fyers.access_token
        if not new_token:
            LOGGER.error("Token refresh completed but no FYERS_ACCESS_TOKEN was written.")
            return False
        self.settings = new_settings
        self.indicator_config = new_settings.indicator_config
        self.notifier = TelegramNotifier(self.settings.telegram.bot_token, self.settings.telegram.chat_id)
        self.confirmed_notifier = TelegramNotifier(
            self.settings.st_confirmed_telegram.bot_token or self.settings.telegram.bot_token,
            self.settings.st_confirmed_telegram.chat_id or self.settings.telegram.chat_id,
        )
        self.broker = FyersBrokerClient(
            client_id=new_settings.fyers.client_id,
            access_token=new_token,
            user_id=new_settings.fyers.user_id,
        )
        return True

    def _send_eod_summary(self, trade_date: date) -> None:
        if self.eod_summary_sent_for == trade_date:
            return
        self.eod_summary_sent_for = trade_date
        lines = [
            f"📊 <b>End of Day Summary</b> {trade_date.isoformat()}",
            f"Supertrend timeframe: <b>{self.strategy_timeframe_minutes} min</b>",
        ]
        for symbol in self.symbols:
            flips = self.supertrend_flip_events[symbol]
            touches = self.supertrend_touch_events[symbol]
            completed_moves = [float(item["points_before_flip"]) for item in flips if item.get("points_before_flip") is not None]
            avg_move = sum(completed_moves) / len(completed_moves) if completed_moves else 0.0
            max_move = max(completed_moves) if completed_moves else 0.0
            lines.append(
                f"\n<b>{self._short_symbol(symbol)}</b>\n"
                f"Flip signals: {len(flips)}\n"
                f"Touch signals: {len(touches)}\n"
                f"Completed move segments: {len(completed_moves)}\n"
                f"Avg points before direction change: {avg_move:.2f}\n"
                f"Max points before direction change: {max_move:.2f}"
            )
            if flips:
                last_flip = flips[-1]
                lines.append(
                    f"Last flip: {pd.Timestamp(last_flip['bar_time']).strftime('%H:%M')} | "
                    f"{'UP' if int(last_flip['direction']) == 1 else 'DOWN'} | "
                    f"Close ₹{float(last_flip['close']):.2f}"
                )
            if self.settings.enable_experimental_ict_stream:
                lines.extend(self._build_ict_eod_lines(symbol))
                self._write_ict_eod_trade_log(symbol, trade_date)
        if self._is_index(symbol) or getattr(self.settings, "enable_fo_telegram_alerts", False):
            self.notifier.send("\n".join(lines))

        # Trigger EOD Post-Market Swarm Analysis automatically
        try:
            LOGGER.info("Market closed. Triggering Post-Market Swarm Analysis...")
            from trade_system.domains.advisory.application.agent.postmarket_improver_agent import PostMarketImproverAgent
            improver = PostMarketImproverAgent(broker=self.broker)
            import asyncio
            import threading
            
            def run_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Run the async post-market analysis
                    loop.run_until_complete(improver.run_post_market_analysis())
                    LOGGER.info("✅ Post-Market Swarm Analysis successfully run.")
                except Exception as ex:
                    LOGGER.error(f"EOD Post-Market Swarm Analysis failed: {ex}")
                finally:
                    loop.close()
            
            threading.Thread(target=run_in_thread, name="EOD_Swarm_Analysis", daemon=True).start()
        except Exception as e:
            LOGGER.error(f"Failed to trigger Post-Market Swarm Analysis: {e}")

    def _write_ict_eod_trade_log(self, symbol: str, trade_date: date) -> None:
        trades = self.ict_trades[symbol]
        if not trades:
            return
        output_dir = Path("reports/live_ict")
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = pd.DataFrame(
            [
                {
                    "symbol": self._short_symbol(symbol),
                    "direction": "LONG" if trade.direction == 1 else "SHORT",
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "trigger": trade.trigger,
                    "confluence_score": trade.confluence_score,
                    "points_captured": trade.points_captured,
                    "risk_points": trade.risk_points,
                    "r_multiple": trade.r_multiple,
                }
                for trade in trades
            ]
        )
        payload.to_csv(output_dir / f"{self._short_symbol(symbol)}_ict_live_{trade_date.isoformat()}.csv", index=False)

    @staticmethod
    def _short_symbol(symbol: str) -> str:
        return symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "")

    @staticmethod
    def _parse_clock(value: str) -> dt_time:
        return datetime.strptime(value, "%H:%M").time()


def _merge_intraday_3min_bars(*, existing: pd.DataFrame, today_bars: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    existing_frame = existing.copy()
    if existing_frame.empty:
        existing_frame = pd.DataFrame(columns=base_columns)
    existing_frame = existing_frame[[col for col in existing_frame.columns if col in base_columns]].copy()
    if "timestamp" not in existing_frame.columns:
        existing_frame["timestamp"] = pd.to_datetime([])
    existing_frame["timestamp"] = pd.to_datetime(existing_frame["timestamp"], format="mixed")

    today_payload = today_bars.copy()[base_columns]
    today_payload["timestamp"] = pd.to_datetime(today_payload["timestamp"], format="mixed")

    merged = pd.concat([existing_frame, today_payload], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return merged[base_columns]
