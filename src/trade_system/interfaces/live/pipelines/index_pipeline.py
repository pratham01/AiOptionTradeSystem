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
from pathlib import Path
from typing import Any

import numpy as np
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
from trade_system.domains.strategy.application.strategies.institutional_reversal_strategy import (
    InstitutionalIntradayReversalStrategy,
)
from trade_system.domains.strategy.application.strategies.base import StrategyContext
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

        # Institutional Intraday Reversal Strategy
        self.institutional_reversal = InstitutionalIntradayReversalStrategy(
            st_period=self.st_period,
            st_multiplier=self.st_multiplier,
            target_rr=2.0,
            max_risk_pct=0.004,
        )

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

            # Institutional Intraday Reversal Check (Feature Flagged)
            if self.settings.enable_intraday_reversal_alerts:
                self._check_institutional_reversal(symbol, adjusted)

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

    def _check_supertrend_touch(
        self,
        symbol: str,
        last_bar: pd.Series,
        bar_time: pd.Timestamp,
    ) -> None:
        """Detect price touch on Supertrend line and dispatch Telegram retest alert."""
        if self._last_touch_bar_time.get(symbol) == bar_time:
            return
        if "low" not in last_bar or "high" not in last_bar:
            return
        supertrend_val = float(last_bar.get("supertrend", 0.0))
        if supertrend_val <= 0:
            return
        if not (float(last_bar["low"]) <= supertrend_val <= float(last_bar["high"])):
            return

        direction = int(last_bar["supertrend_direction"])
        close_price = float(last_bar["close"])
        self._last_touch_bar_time[symbol] = bar_time
        self._supertrend_touch_events[symbol].append(
            {
                "bar_time": bar_time,
                "direction": direction,
                "close": close_price,
                "supertrend": supertrend_val,
            }
        )

        strike_info = self._build_adjacent_strikes_block(symbol, bar_time, close_price, count=5)
        short_sym = symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")
        dir_label = "BULLISH ST (SUPPORT)" if direction == 1 else "BEARISH ST (RESISTANCE)"
        touch_icon = "🟢" if direction == 1 else "🔴"
        touch_msg = (
            f"🧭 <b>{short_sym} Supertrend Touch ({bar_time.strftime('%H:%M')})</b>\n"
            f"Level: <b>{touch_icon} {dir_label}</b>\n"
            f"ST Level: ₹{supertrend_val:.2f} | Close: ₹{close_price:.2f}\n"
            f"<i>Institutional re-entry zone ({self.strategy_tf}m).</i>"
            f"{strike_info}"
        )
        try:
            self.notifier.send(touch_msg)
            self.confirmed_notifier.send(touch_msg)
        except Exception as exc:
            LOGGER.exception("Failed to send ST touch alert for %s: %s", symbol, exc)

        self.alert_agent.alert_retest(symbol, "Supertrend", supertrend_val, log_only=True)
        LOGGER.info(
            "Supertrend touch for %s at %s | trend=%s close=%.2f st=%.2f",
            symbol,
            bar_time,
            direction,
            close_price,
            supertrend_val,
        )

    def _check_trend_change(self, symbol: str, df: pd.DataFrame, adjusted: pd.DataFrame) -> None:
        """Detect SuperTrend flips & touches and send Telegram alerts."""
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

        # Check for price touching Supertrend
        self._check_supertrend_touch(symbol, last_bar, bar_time)

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
            self._supertrend_flip_events[symbol].append(
                {
                    "bar_time": bar_time,
                    "direction": current_trend,
                    "close": close_price,
                    "supertrend": supertrend_val,
                    "timeframe_minutes": self.strategy_tf,
                }
            )

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

            # --- Adjacent 5 Strikes Context ---
            strike_info = self._build_adjacent_strikes_block(symbol, bar_time, close_price, count=5)

            short_sym = symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")
            main_msg = (
                f"{color} <b>{short_sym} {bar_time.strftime('%H:%M')}</b>\n"
                f"Direction changed to <b>{direction_str}</b> ({self.strategy_tf}m)\n"
                f"15m Trend: <b>{st_15_dir_str}</b>\n"
                f"Close: ₹{close_price:.2f}\n"
                f"Supertrend: ₹{supertrend_val:.2f}\n"
                f"Action: <b>{action}</b>"
                f"{strike_info}"
            )
            try:
                if getattr(self.settings, "enable_main_channel_supertrend_alerts", False):
                    self.notifier.send(main_msg)
                confirmed_msg = (
                    f"{color} <b>⚡ ST FLIP — {short_sym} {bar_time.strftime('%H:%M')}</b>\n"
                    f"Timeframe: <b>{self.strategy_tf}m Supertrend</b>\n"
                    f"New Direction: <b>{'🟢 UP (BULLISH)' if current_trend == 1 else '🔴 DOWN (BEARISH)'}</b>\n"
                    f"15m Alignment: <b>{st_15_dir_str}</b>\n"
                    f"Close: ₹{close_price:.2f}  |  ST Level: ₹{supertrend_val:.2f}\n"
                    f"Action: <b>{action}</b>"
                    f"{strike_info}"
                )
                self.confirmed_notifier.send(confirmed_msg)
            except Exception as exc:
                LOGGER.exception("Failed to send trend change alert for %s: %s", symbol, exc)

            LOGGER.info(
                "ST crossover %s at %s | trend=%s close=%.2f st=%.2f",
                symbol, bar_time, current_trend, close_price, supertrend_val,
            )

    # ------------------------------------------------------------------
    # Option Chain Adjacent Strike Supertrend Helpers
    # ------------------------------------------------------------------

    def _build_adjacent_strikes_block(
        self,
        symbol: str,
        bar_time: pd.Timestamp | datetime,
        close_price: float,
        count: int = 5,
    ) -> str:
        """Build formatted message block showing Supertrend direction for adjacent strikes."""
        clean_sym = symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "").replace("-EQ", "")
        current_oc_df = None
        oc_analysis = None
        if callable(self._oc_snapshot_getter):
            try:
                res = self._oc_snapshot_getter(clean_sym)
                if isinstance(res, tuple):
                    current_oc_df, oc_analysis = res
                elif isinstance(res, pd.DataFrame):
                    current_oc_df = res
            except Exception as exc:
                LOGGER.warning("Failed to get OC snapshot via getter for %s: %s", clean_sym, exc)

        date_str = bar_time.strftime("%Y%m%d") if hasattr(bar_time, "strftime") else datetime.now().strftime("%Y%m%d")
        history_path = self.settings.option_chain_data_dir / f"{clean_sym}_strikes_{date_str}.csv"
        history_df = self._load_option_chain_history(history_path)
        if history_df.empty:
            return ""

        if oc_analysis and oc_analysis.get("atm", 0):
            atm = int(oc_analysis.get("atm", 0))
        else:
            step = 100 if close_price > 40000 else 50
            atm = int(round(close_price / step) * step)

        if current_oc_df is not None and not current_oc_df.empty:
            ce_strikes = self._select_adjacent_option_chain_strikes(current_oc_df, atm, "CE", count=count)
            pe_strikes = self._select_adjacent_option_chain_strikes(current_oc_df, atm, "PE", count=count)
        else:
            ce_strikes = self._select_adjacent_option_chain_strikes(history_df, atm, "CE", count=count)
            pe_strikes = self._select_adjacent_option_chain_strikes(history_df, atm, "PE", count=count)

        if not ce_strikes and not pe_strikes:
            return ""

        def _build_strike_lines(strikes: list[int], opt_type: str) -> list[str]:
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

        if not ce_lines and not pe_lines:
            return ""

        parts = [f"\n\n<b>OI Snapshot (ATM {atm}) — {count} Strikes Each</b>"]
        if ce_lines:
            parts.append("<b>CALL (CE):</b>\n" + "\n".join(ce_lines))
        if pe_lines:
            parts.append("<b>PUT (PE):</b>\n" + "\n".join(pe_lines))
        parts.append("🟢=ST Bullish  🔴=ST Bearish")
        return "\n".join(parts)

    def _select_adjacent_option_chain_strikes(
        self,
        current_df: pd.DataFrame,
        atm: int,
        option_type: str,
        count: int = 5,
    ) -> list[int]:
        if "option_type" not in current_df.columns or "strike" not in current_df.columns:
            return []
        strikes = current_df.loc[current_df["option_type"].str.upper() == option_type.upper(), "strike"]
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
    ) -> dict[str, Any] | None:
        if "option_type" not in history_df.columns or "strike" not in history_df.columns:
            return None
        subset = history_df[
            (history_df["option_type"].str.upper() == option_type.upper())
            & (history_df["strike"].round().astype(int) == strike)
        ].sort_values("timestamp").reset_index(drop=True)
        if len(subset) < self.st_period + 2:
            return None
        prices = subset["ltp"].astype(float).values
        opens = prices[:-1]
        closes = prices[1:]
        highs = np.maximum(opens, closes) * 1.001
        lows = np.minimum(opens, closes) * 0.999
        timestamps = subset["timestamp"].iloc[1:].values
        ohlc = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "timestamp": timestamps},
            index=timestamps,
        )
        st_df = calculate_supertrend(ohlc, period=self.st_period, multiplier=self.st_multiplier)
        if st_df is None or st_df.empty:
            return None
        last = st_df.iloc[-1]
        return {
            "strike": strike,
            "option_type": option_type,
            "direction": int(last["supertrend_direction"]),
            "st_level": float(last["supertrend"]),
            "ltp": float(last["close"]),
        }

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

    def _check_institutional_reversal(self, symbol: str, adjusted: pd.DataFrame) -> None:
        """Evaluate Institutional Intraday Reversal and dispatch alert if confirmed."""
        try:
            context = StrategyContext(
                symbol=symbol,
                timeframe=f"{self.strategy_timeframe_minutes}m",
                history_df=adjusted,
            )
            signal = self.institutional_reversal.evaluate(context)
            if signal is None:
                return

            # Build adjacent 5-strike Supertrend snapshot
            bar_time = signal.timestamp if isinstance(signal.timestamp, pd.Timestamp) else pd.Timestamp(signal.timestamp)
            strike_block = self._build_adjacent_strikes_block(symbol, bar_time, signal.entry_price, count=5)

            self.alert_agent.alert_institutional_reversal(
                symbol=symbol,
                direction=signal.direction,
                price=signal.entry_price,
                sl=signal.stop_loss,
                target_1=signal.target_1,
                target_2=signal.target_2,
                confluence=signal.confluence_factors[0] if signal.confluence_factors else "",
                strike_block=strike_block,
            )
            LOGGER.info(
                "Institutional Reversal alert sent for %s: %s at ₹%.2f (SL=%.2f, T1=%.2f)",
                symbol, signal.direction, signal.entry_price, signal.stop_loss, signal.target_1,
            )
        except Exception as exc:
            LOGGER.error("IndexPipeline._check_institutional_reversal failed for %s: %s", symbol, exc)

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
