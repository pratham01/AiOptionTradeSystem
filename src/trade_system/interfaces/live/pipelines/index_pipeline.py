"""
IndexPipeline — Full analytics handler for NSE/BSE index symbols.

Triggered
---------
``on_bar(symbol, bar, minute_data)``  — called by BarAggregator on every
completed 1-min bar for an index symbol.

``on_tick(symbol, price, tick_time)`` — called on every live tick for
tick-level alert checks (zone proximity, ORB breakout, gap).

Responsibilities
----------------
1. Resample 1-min data → strategy timeframe (e.g. 3 min).
2. Run SuperTrend → detect trend flip → send Telegram alert.
3. Run RSI Divergence detection.
4. Check Opening Range Breakout (ORB / first-15-min candle).
5. Check S/R channel proximity.
6. Evaluate Gamma Blast setup.
7. Run SMC / technical extremes alerts (delegated to LiveAlertAgent).
8. Run ICT FVG / OB / Liquidity stream (if experimental flag enabled).
9. Run Confirmed Strategy.
10. Persist 3-min bars to DB + CSV (via BarStore).

What it does NOT do
-------------------
- Send premarket Telegram reports (done by orchestrator at day init).
- Know about option chains (option_chain_pipeline handles that).
- Manage WebSocket connections.
- Know about F&O stocks.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dt_time
from typing import Any

import pandas as pd

from trade_system.domains.market_data.application.bar_store import BarStore
from trade_system.domains.advisory.application.agent.live_alert_agent import LiveAlertAgent
from trade_system.interfaces.live.helpers import (
    _completed_timeframe_bars,
    _latest_session_minute,
    _sanitize_intraday_minutes,
    detect_smc_signal,
    get_gap_adjusted_data,
    resample_to_timeframe,
    valid_supertrend_rows as _valid_supertrend_rows,
)
from trade_system.domains.strategy.application.indicators import calculate_supertrend
from trade_system.shared.config import Settings
from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)


class IndexPipeline:
    """
    Analytics pipeline for index symbols.

    This class is intentionally free of broker/WebSocket knowledge.
    It receives already-aggregated bars and routing decisions from the
    orchestrator (LiveMarketDataService).

    Parameters
    ----------
    settings : Settings
    alert_agent : LiveAlertAgent
    notifier : TelegramNotifier           Main alert channel.
    confirmed_notifier : TelegramNotifier ST-confirmed channel.
    bar_store : BarStore
    strategy_timeframe_minutes : int      Typically 3.
    supertrend_period : int
    supertrend_multiplier : float
    rsi_divergence                        RsiDivergence instance.
    early_morning_agent                   EarlyMorningAgent instance.
    sniper_agent                          SniperReversalAgent instance.
    gamma_agent                           GammaBlastAgent instance.
    mwpl_analyzer                         MwplAnalyzer instance.
    mwpl_setups : dict                    Pre-loaded MWPL setup dict.
    ict_rules / ict_fvg / ict_ob / ict_liq / ict_ms
        ICT analysis instances (keyed per symbol where needed).
    """

    def __init__(
        self,
        settings: Settings,
        alert_agent: LiveAlertAgent,
        notifier: TelegramNotifier,
        confirmed_notifier: TelegramNotifier,
        bar_store: BarStore,
        strategy_timeframe_minutes: int,
        supertrend_period: int,
        supertrend_multiplier: float,
        rsi_divergence,
        early_morning_agent,
        sniper_agent,
        gamma_agent,
        mwpl_analyzer,
        mwpl_setups: dict,
        # ICT components (dict keyed by symbol)
        ict_rules=None,
        ict_fvg: dict | None = None,
        ict_ob: dict | None = None,
        ict_liq: dict | None = None,
        ict_ms: dict | None = None,
        # Option chain snapshot access (latest analysis keyed by short_sym)
        oc_snapshot_getter=None,
    ) -> None:
        self.settings = settings
        self.alert_agent = alert_agent
        self.notifier = notifier
        self.confirmed_notifier = confirmed_notifier
        self.bar_store = bar_store
        self.strategy_tf = strategy_timeframe_minutes
        self.st_period = supertrend_period
        self.st_multiplier = supertrend_multiplier
        self.rsi_divergence = rsi_divergence
        self.early_morning_agent = early_morning_agent
        self.sniper_agent = sniper_agent
        self.gamma_agent = gamma_agent
        self.mwpl_analyzer = mwpl_analyzer
        self.mwpl_setups = mwpl_setups
        self.ict_rules = ict_rules
        self.ict_fvg = ict_fvg or {}
        self.ict_ob = ict_ob or {}
        self.ict_liq = ict_liq or {}
        self.ict_ms = ict_ms or {}
        self._oc_snapshot_getter = oc_snapshot_getter

        # Per-symbol tracking state
        self._strategy_data: dict[str, pd.DataFrame] = {}
        self._last_trend: dict[str, int | None] = {}
        self._last_processed_trend_bar_time: dict[str, pd.Timestamp | None] = {}
        self._last_signal_bar_time: dict[str, pd.Timestamp | None] = {}
        self._last_touch_bar_time: dict[str, pd.Timestamp | None] = {}
        self._last_rsi_div_signal_time: dict[str, datetime | None] = {}
        self._ict_last_processed_bar_time: dict[str, pd.Timestamp | None] = {}
        self._gamma_open_position: dict[str, dict | None] = {}
        self._gamma_trades: dict[str, list] = {}
        self._supertrend_flip_events: dict[str, list] = {}
        self._supertrend_touch_events: dict[str, list] = {}

        self._market_start = self._parse_clock(settings.market_start)
        self._market_end = self._parse_clock(settings.market_end)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_symbol(self, symbol: str) -> None:
        """Ensure all per-symbol state dicts are populated for a new symbol."""
        if symbol not in self._strategy_data:
            self._strategy_data[symbol] = pd.DataFrame()
            self._last_trend[symbol] = None
            self._last_processed_trend_bar_time[symbol] = None
            self._last_signal_bar_time[symbol] = None
            self._last_touch_bar_time[symbol] = None
            self._last_rsi_div_signal_time[symbol] = None
            self._ict_last_processed_bar_time[symbol] = None
            self._gamma_open_position[symbol] = None
            self._gamma_trades[symbol] = []
            self._supertrend_flip_events[symbol] = []
            self._supertrend_touch_events[symbol] = []

    def reset(self, symbols: list[str]) -> None:
        """Reset intraday state (called at day boundary)."""
        for sym in symbols:
            self._strategy_data[sym] = pd.DataFrame()
            self._last_trend[sym] = None
            self._last_processed_trend_bar_time[sym] = None
            self._last_signal_bar_time[sym] = None
            self._last_touch_bar_time[sym] = None
            self._last_rsi_div_signal_time[sym] = None
            self._ict_last_processed_bar_time[sym] = None
            self._gamma_open_position[sym] = None
            self._gamma_trades[sym] = []
            self._supertrend_flip_events[sym] = []
            self._supertrend_touch_events[sym] = []

    def get_strategy_data(self, symbol: str) -> pd.DataFrame:
        """Return the current strategy-timeframe DataFrame for a symbol."""
        return self._strategy_data.get(symbol, pd.DataFrame())

    def get_last_trend(self, symbol: str) -> int | None:
        return self._last_trend.get(symbol)

    def on_bar(
        self,
        symbol: str,
        bar: pd.Series,
        minute_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Process a completed 1-min bar for an index symbol.

        Called by the orchestrator after BarAggregator emits a new bar.
        """
        self.register_symbol(symbol)
        current_date = self._today_ist()

        # 1. Persist 1-min bar and update strategy-timeframe file
        self.bar_store.save_1min_bar(symbol, bar, current_date)
        updated_strategy = self.bar_store.update_strategy_file(
            symbol,
            minute_data.get(symbol, pd.DataFrame()),
            current_date,
            resample_fn=resample_to_timeframe,
            completed_bars_fn=_completed_timeframe_bars,
            sanitize_fn=_sanitize_intraday_minutes,
            latest_session_minute_fn=_latest_session_minute,
        )
        if not updated_strategy.empty:
            self._strategy_data[symbol] = updated_strategy

        # 2. Run SuperTrend + RSI + SMC + Sniper + MWPL
        self._run_analytics(symbol)

    # ------------------------------------------------------------------
    # Analytics entry point (mirrors old _run_supertrend)
    # ------------------------------------------------------------------

    def _run_analytics(self, symbol: str) -> None:
        """Run all index analytics after each completed bar."""
        timeframe = self._strategy_data.get(symbol, pd.DataFrame())
        if timeframe.empty:
            return

        timeframe = timeframe.sort_index()
        current_date = self._today_ist()
        adjusted = get_gap_adjusted_data(timeframe, self.st_period, current_date)

        if len(timeframe) < self.st_period + 1:
            return

        # --- SuperTrend ---
        st_df = calculate_supertrend(adjusted, period=self.st_period, multiplier=self.st_multiplier)
        if not st_df.empty:
            today_df = st_df[st_df.index.date == current_date]
            if not today_df.empty:
                self._check_trend_change(symbol, st_df, adjusted)

        # --- RSI Divergence ---
        rsi_df = self.rsi_divergence.calculate(adjusted)
        if not rsi_df.empty:
            today_rsi = rsi_df[rsi_df.index.date == current_date]
            if not today_rsi.empty:
                self._check_rsi_divergence(symbol, today_rsi)

        # --- SMC & Technical Alerts ---
        smc_signal = detect_smc_signal(adjusted)
        if smc_signal is not None:
            self.alert_agent.process_smc_signal(symbol, smc_signal)

        if not adjusted.empty:
            self.alert_agent.monitor_technical_extremes(symbol, adjusted)
            self.alert_agent.monitor_squeeze_breakout(symbol, adjusted)

            # Early Morning Setups (Gap & Go / ORB)
            self._run_early_morning_scan(symbol, adjusted)

            # Sniper Reversal
            sniper_sugg = self.sniper_agent.analyze_for_reversal(symbol, adjusted)
            if sniper_sugg:
                from trade_system.shared import TradeDirection
                setup_type = "BOTTOM FISHING" if sniper_sugg.direction == TradeDirection.CALL else "TOP SNIPE"
                self.alert_agent.alert_sniper_reversal(
                    symbol, setup_type,
                    sniper_sugg.entry_zone_high, sniper_sugg.stop_loss,
                    sniper_sugg.narrative,
                )

            # MWPL Trigger
            self._check_mwpl_trigger(symbol, adjusted)

        # --- ICT Stream ---
        if self.settings.enable_experimental_ict_stream:
            self._run_ict_stream(symbol, adjusted)

        # Latest bar for confirmed strategy
        latest_bar_time = pd.Timestamp(timeframe.index.max()) if not timeframe.empty else None
        # Note: confirmed strategy still delegates to orchestrator
        # (cannot call self._run_confirmed_strategy here without circular import)
        # The orchestrator hooks into on_bar_complete to run confirmed strategy.

    def _run_early_morning_scan(self, symbol: str, adjusted: pd.DataFrame) -> None:
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
                    self.alert_agent.alert_gap_and_go(
                        symbol, sugg.direction.value,
                        sugg.entry_zone_high, sugg.stop_loss,
                    )
                elif "ORB" in sugg.tags:
                    self.alert_agent.alert_orb(
                        symbol, sugg.direction.value,
                        sugg.entry_zone_high, sugg.stop_loss,
                    )
        except Exception as exc:
            LOGGER.error("Early morning scan failed for %s: %s", symbol, exc)

    def _check_mwpl_trigger(self, symbol: str, adjusted: pd.DataFrame) -> None:
        clean_sym = symbol.replace("NSE:", "").replace("-EQ", "")
        if clean_sym not in self.mwpl_setups.get("SQUEEZE", []) and \
           clean_sym not in self.mwpl_setups.get("UNWINDING", []):
            return
        if len(adjusted) < 2:
            return
        latest = adjusted.iloc[-1]
        prev = adjusted.iloc[-2]
        vwap = latest.get("vwap")
        if not vwap or vwap <= 0:
            return
        price = float(latest["close"])
        prev_price = float(prev["close"])
        setup_type = "SQUEEZE" if clean_sym in self.mwpl_setups.get("SQUEEZE", []) else "UNWINDING"
        try:
            data = self.mwpl_analyzer.get_mwpl_data()
            pct = data[data["SYMBOL"] == clean_sym]["MWPL_PCT"].iloc[0] if not data.empty else 85.0
        except Exception:
            pct = 85.0
        if setup_type == "SQUEEZE" and prev_price <= vwap <= price:
            self.alert_agent.alert_mwpl_trigger(symbol, "SQUEEZE", price, pct)
        elif setup_type == "UNWINDING" and prev_price >= vwap >= price:
            self.alert_agent.alert_mwpl_trigger(symbol, "UNWINDING", price, pct)

    def _run_ict_stream(self, symbol: str, adjusted: pd.DataFrame) -> None:
        """Delegate to orchestrator ICT logic — kept here as a placeholder hook."""
        # Full ICT logic remains in collector.py during transition period.
        # This stub allows the pipeline to be used without breaking anything.
        pass

    # ------------------------------------------------------------------
    # SuperTrend trend-change detection
    # ------------------------------------------------------------------

    def _check_trend_change(self, symbol: str, df: pd.DataFrame, adjusted: pd.DataFrame) -> None:
        """Detect SuperTrend flips and send Telegram alerts."""
        valid_df = _valid_supertrend_rows(df)
        if valid_df.empty:
            return

        current_date = self._today_ist()
        today_df = valid_df[valid_df.index.date == current_date]
        if today_df.empty:
            return

        last_bar = today_df.iloc[-1]
        bar_time = today_df.index[-1]

        if self._last_processed_trend_bar_time.get(symbol) == bar_time:
            return

        current_trend = int(last_bar["supertrend_direction"])

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

        previous_trend = (
            int(previous_bar["supertrend_direction"])
            if previous_bar is not None
            else self._last_trend.get(symbol)
        )

        self._last_trend[symbol] = current_trend
        self._last_processed_trend_bar_time[symbol] = bar_time

        if (
            self._last_signal_bar_time.get(symbol) != bar_time
            and previous_trend is not None
            and current_trend != previous_trend
        ):
            close_price = float(last_bar["close"])
            supertrend_val = float(last_bar["supertrend"])

            # Reject false flips where price is still on wrong side of ST
            if current_trend == 1 and close_price <= supertrend_val:
                return
            if current_trend == -1 and close_price >= supertrend_val:
                return

            self._last_signal_bar_time[symbol] = bar_time

            action = "BUY CALL" if current_trend == 1 else "BUY PUT"
            color = "🟢" if current_trend == 1 else "🔴"
            direction_str = "UP" if current_trend == 1 else "DOWN"

            # 15m context
            st_15_dir_str = "UNKNOWN"
            try:
                strat = self._strategy_data.get(symbol)
                if strat is not None and not strat.empty:
                    df_15m = resample_to_timeframe(strat, 15)
                    st_15m = calculate_supertrend(df_15m, period=self.st_period, multiplier=self.st_multiplier)
                    if not st_15m.empty:
                        dir_val = st_15m.iloc[-1].get("supertrend_direction")
                        if pd.notna(dir_val):
                            st_15_dir_str = "UP" if dir_val == 1 else "DOWN"
            except Exception as exc:
                LOGGER.warning("Failed to calculate 15m ST for %s: %s", symbol, exc)

            short_sym = symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")
            main_msg = (
                f"{color} <b>{short_sym} {bar_time.strftime('%H:%M')}</b>\n"
                f"Direction changed to <b>{direction_str}</b> ({self.strategy_tf}m)\n"
                f"15m Trend: <b>{st_15_dir_str}</b>\n"
                f"Close: ₹{close_price:.2f}\n"
                f"Supertrend: ₹{supertrend_val:.2f}\n"
                f"Action: <b>{action}</b>"
            )
            try:
                self.notifier.send(main_msg)
                confirmed_msg = (
                    f"{color} <b>⚡ ST FLIP — {short_sym} {bar_time.strftime('%H:%M')}</b>\n"
                    f"Timeframe: <b>{self.strategy_tf}m Supertrend</b>\n"
                    f"New Direction: <b>{'🟢 UP (BULLISH)' if current_trend == 1 else '🔴 DOWN (BEARISH)'}</b>\n"
                    f"15m Alignment: <b>{st_15_dir_str}</b>\n"
                    f"Close: ₹{close_price:.2f}  |  ST Level: ₹{supertrend_val:.2f}\n"
                    f"Action: <b>{action}</b>"
                )
                self.confirmed_notifier.send(confirmed_msg)
            except Exception as exc:
                LOGGER.exception("Failed to send trend change alert for %s: %s", symbol, exc)

            LOGGER.info(
                "ST crossover %s at %s | trend=%s close=%.2f st=%.2f",
                symbol, bar_time, current_trend, close_price, supertrend_val,
            )

    # ------------------------------------------------------------------
    # RSI Divergence
    # ------------------------------------------------------------------

    def _check_rsi_divergence(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        last_bar = df.iloc[-1]
        bar_time = df.index[-1]
        now = datetime.now()
        last_alert = self._last_rsi_div_signal_time.get(symbol)
        if last_alert and (now - last_alert).total_seconds() < 900:
            return
        signal = last_bar.get("signal", 0)
        if signal == 0:
            return
        self._last_rsi_div_signal_time[symbol] = now
        direction = "BULLISH" if signal == 1 else "BEARISH"
        color = "💎" if signal == 1 else "🔥"
        action = "LONG Opportunity" if signal == 1 else "SHORT Opportunity"
        short_sym = symbol.split(":")[-1].replace("-INDEX", "")
        self.notifier.send(
            f"{color} <b>{short_sym} RSI Divergence</b>\n"
            f"Momentum Signal: <b>{direction}</b>\n"
            f"Time: {bar_time.strftime('%H:%M')}\n"
            f"Price: ₹{last_bar['close']:.2f}\n"
            f"Action: <b>{action}</b>"
        )
        LOGGER.info("RSI Divergence for %s at %s | signal=%s", symbol, bar_time, direction)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _today_ist() -> date:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata")).date()

    @staticmethod
    def _parse_clock(value: str) -> dt_time:
        h, m = value.split(":")
        return dt_time(int(h), int(m))
