"""
GammaPipeline — 0DTE Gamma Blast strategy.

Triggered
---------
``on_bar(symbol, bar, minute_data, oc_analysis)`` — evaluated on each
completed 1-min bar during the Gamma window (14:30–15:05 IST).

Responsibilities
----------------
1. In the Gamma window: ask GammaBlastAgent for a trade suggestion.
2. If triggered: record the trade, select ATM option from OC, send alert.
3. On subsequent bars while a trade is open: check target/stop, close at
   15:12 EOD, record PnL, send Telegram close message.
4. Expose trade history and open position for EOD dashboard reporting.

What it does NOT do
-------------------
- Know about WebSocket or tick routing.
- Compute RSI or SuperTrend.
- Manage position sizing (that is the execution engine's job).

Index-only
----------
Gamma Blast is only triggered for index instruments where a live option
chain is available.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from typing import Any

import pandas as pd

from trade_system.shared import TradeDirection

LOGGER = logging.getLogger(__name__)

_GAMMA_START = dt_time(14, 30)
_GAMMA_END = dt_time(15, 5)
_EOD_CLOSE_TIME = dt_time(15, 12)
_MIN_BARS = 30


class GammaTrade:
    """Immutable-ish record of a single Gamma Blast trade."""

    __slots__ = (
        "symbol", "direction", "entry_time", "entry_spot",
        "option_symbol", "entry_premium", "target_spot", "stop_spot",
        "reason", "exit_time", "exit_premium", "pnl", "pnl_pct", "exit_reason",
    )

    def __init__(
        self,
        symbol: str,
        direction: TradeDirection,
        entry_time: datetime,
        entry_spot: float,
        option_symbol: str,
        entry_premium: float,
        target_spot: float,
        stop_spot: float,
        reason: str,
    ) -> None:
        self.symbol = symbol
        self.direction = direction
        self.entry_time = entry_time
        self.entry_spot = entry_spot
        self.option_symbol = option_symbol
        self.entry_premium = entry_premium
        self.target_spot = target_spot
        self.stop_spot = stop_spot
        self.reason = reason
        # Set at close
        self.exit_time: datetime | None = None
        self.exit_premium: float = 0.0
        self.pnl: float = 0.0
        self.pnl_pct: float = 0.0
        self.exit_reason: str = ""


class GammaPipeline:
    """
    0DTE Gamma Blast strategy pipeline.

    Parameters
    ----------
    gamma_agent : GammaBlastAgent
    notifier : TelegramNotifier
    latest_oc_analysis : dict
        Shared reference to the orchestrator's latest_oc_analysis dict
        (keyed by short symbol, e.g. "NIFTY50").
    """

    def __init__(
        self,
        gamma_agent,
        notifier,
        latest_oc_analysis: dict[str, Any],
    ) -> None:
        self.gamma_agent = gamma_agent
        self.notifier = notifier
        self._oc_analysis = latest_oc_analysis

        # Per-symbol state
        self._open_trade: dict[str, GammaTrade | None] = {}
        self._trade_history: dict[str, list[GammaTrade]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_symbol(self, symbol: str) -> None:
        if symbol not in self._open_trade:
            self._open_trade[symbol] = None
            self._trade_history[symbol] = []

    def reset(self, symbols: list[str]) -> None:
        """Reset intraday state at day boundary (preserve trade_history for EOD)."""
        for sym in symbols:
            self._open_trade[sym] = None

    def get_open_trade(self, symbol: str) -> GammaTrade | None:
        return self._open_trade.get(symbol)

    def get_trade_history(self, symbol: str) -> list[GammaTrade]:
        return self._trade_history.get(symbol, [])

    # ------------------------------------------------------------------
    # Bar-level hook
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: pd.Series,
        minute_data: dict[str, pd.DataFrame],
    ) -> None:
        """Evaluate Gamma Blast setup or manage an open trade."""
        self.register_symbol(symbol)

        open_trade = self._open_trade.get(symbol)
        if open_trade:
            self._manage_trade(symbol, open_trade, minute_data)
            return

        df = minute_data.get(symbol)
        if df is None or len(df) < _MIN_BARS:
            return

        now = self._now()
        if not (_GAMMA_START <= now.time() <= _GAMMA_END):
            return

        self._evaluate_entry(symbol, df, now)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self,
        symbol: str,
        df: pd.DataFrame,
        now: datetime,
    ) -> None:
        """Ask GammaBlastAgent for a suggestion and open a trade if found."""
        df_eval = df.tail(100).copy()
        today_mask = df_eval.index.date == now.date()
        if not today_mask.any():
            return

        df_today = df_eval[today_mask]
        if not df_today.empty:
            typical_price = (df_today["high"] + df_today["low"] + df_today["close"]) / 3
            df_eval.loc[today_mask, "vwap"] = (
                (typical_price * df_today["volume"]).cumsum()
                / df_today["volume"].cumsum()
            )

        # RSI (14) for GammaBlastAgent
        delta = df_eval["close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        df_eval["rsi"] = 100 - (100 / (1 + gain / loss))
        df_eval["volume_sma_20"] = df_eval["volume"].rolling(20).mean()

        short_sym = self._short(symbol)
        oc_analysis = self._oc_analysis.get(short_sym)

        try:
            suggestion = self.gamma_agent.analyze_for_gamma_blast(
                symbol=symbol,
                df=df_eval.reset_index(),
                oc_analysis=oc_analysis,
            )
        except Exception as exc:
            LOGGER.error("GammaBlastAgent error for %s: %s", symbol, exc)
            return

        if suggestion is None:
            return

        LOGGER.info("🚀 GAMMA BLAST triggered for %s — %s", symbol, suggestion.direction.name)

        # Select ATM option
        opt_symbol = "N/A"
        entry_premium = 0.0
        if oc_analysis and oc_analysis.chain_df is not None:
            entry_spot = float(df_eval["close"].iloc[-1])
            step = 100 if "BANK" in symbol or entry_spot > 40000 else 50
            atm_strike = round(suggestion.entry_zone_low / step) * step
            opt_type = "CE" if suggestion.direction == TradeDirection.CALL else "PE"
            mask = (
                (oc_analysis.chain_df["strike"] == atm_strike)
                & (oc_analysis.chain_df["option_type"] == opt_type)
            )
            if mask.any():
                row = oc_analysis.chain_df[mask].iloc[0]
                opt_symbol = row["symbol"]
                entry_premium = float(row["ltp"])

        trade = GammaTrade(
            symbol=symbol,
            direction=suggestion.direction,
            entry_time=now,
            entry_spot=float(df_eval["close"].iloc[-1]),
            option_symbol=opt_symbol,
            entry_premium=entry_premium,
            target_spot=float(suggestion.target),
            stop_spot=float(suggestion.stop_loss),
            reason=suggestion.narrative,
        )
        self._open_trade[symbol] = trade

        self.notifier.send(
            f"💥 <b>0DTE GAMMA BLAST: {self._short(symbol)}</b> 💥\n\n"
            f"Direction: <b>{suggestion.direction.name}</b>\n"
            f"Spot: ₹{trade.entry_spot:.2f}\n"
            f"Target: ₹{trade.target_spot:.2f}\n"
            f"Stop:   ₹{trade.stop_spot:.2f}\n"
            f"Option: {opt_symbol} @ ₹{entry_premium:.2f}\n\n"
            f"<i>{suggestion.narrative}</i>"
        )

    def _manage_trade(
        self,
        symbol: str,
        trade: GammaTrade,
        minute_data: dict[str, pd.DataFrame],
    ) -> None:
        """Check target/stop and EOD close for an open Gamma trade."""
        now = self._now()

        if now.time() >= _EOD_CLOSE_TIME:
            self._close_trade(symbol, trade, "EOD_CLOSE", now)
            return

        df = minute_data.get(symbol)
        if df is None or df.empty:
            return

        curr = float(df["close"].iloc[-1])
        if trade.direction == TradeDirection.CALL:
            if curr >= trade.target_spot:
                self._close_trade(symbol, trade, "TARGET_HIT", now)
            elif curr <= trade.stop_spot:
                self._close_trade(symbol, trade, "STOP_LOSS", now)
        else:
            if curr <= trade.target_spot:
                self._close_trade(symbol, trade, "TARGET_HIT", now)
            elif curr >= trade.stop_spot:
                self._close_trade(symbol, trade, "STOP_LOSS", now)

    def _close_trade(
        self,
        symbol: str,
        trade: GammaTrade,
        reason: str,
        exit_time: datetime,
    ) -> None:
        """Close the active trade, compute PnL, send alert, archive."""
        exit_premium = 0.0
        short_sym = self._short(symbol)
        oc = self._oc_analysis.get(short_sym)
        if oc and oc.chain_df is not None:
            row = oc.chain_df[oc.chain_df["symbol"] == trade.option_symbol]
            if not row.empty:
                exit_premium = float(row.iloc[0]["ltp"])

        pnl = exit_premium - trade.entry_premium
        pnl_pct = (pnl / trade.entry_premium * 100) if trade.entry_premium > 0 else 0.0

        trade.exit_time = exit_time
        trade.exit_premium = exit_premium
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.exit_reason = reason

        emoji = "🟢" if pnl > 0 else "🔴"
        self.notifier.send(
            f"{emoji} <b>GAMMA BLAST CLOSED: {short_sym}</b>\n\n"
            f"Option: {trade.option_symbol}\n"
            f"Direction: {trade.direction.name}\n"
            f"Reason: <b>{reason}</b>\n\n"
            f"Entry: ₹{trade.entry_premium:.2f}\n"
            f"Exit:  ₹{exit_premium:.2f}\n"
            f"PnL:   <b>{pnl:+.2f} ({pnl_pct:+.1f}%)</b>"
        )
        self._trade_history[symbol].append(trade)
        self._open_trade[symbol] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _short(symbol: str) -> str:
        return symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")

    @staticmethod
    def _now() -> datetime:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
