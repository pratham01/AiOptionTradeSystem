"""
Institutional SMC Backtesting Engine (Backend Service).

Provides parameterized, vector-accelerated backtesting for Smart Money Concepts:
- Order Block Taps (Bullish / Bearish)
- Liquidity Sweeps & Reclaims (Turtle Soup)
- Internal Structure CHoCH Realignments
Across customizable timeframes (1m, 3m, 5m, 15m, Daily) and intraday time slots.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from trade_system.domains.strategy.application.indicators.order_blocks import identify_order_blocks
from trade_system.domains.strategy.application.indicators.smc_structure import SMCStructureEngine


# Presets for quick selection
TOP_10_SYMBOLS = [
    "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:INFY-EQ",
    "NSE:TCS-EQ", "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ",
    "NSE:LT-EQ", "NSE:AXISBANK-EQ"
]

TOP_25_SYMBOLS = TOP_10_SYMBOLS + [
    "NSE:ITC-EQ", "NSE:BAJFINANCE-EQ", "NSE:MARUTI-EQ", "NSE:TATAMOTORS-EQ",
    "NSE:SUNPHARMA-EQ", "NSE:TITAN-EQ", "NSE:TATASTEEL-EQ", "NSE:NTPC-EQ",
    "NSE:POWERGRID-EQ", "NSE:HINDUNILVR-EQ", "NSE:ASIANPAINT-EQ", "NSE:WIPRO-EQ",
    "NSE:M&M-EQ", "NSE:ULTRACEMCO-EQ", "NSE:TECHM-EQ"
]


@dataclass
class SMCBacktestParams:
    symbols: List[str]
    timeframe: str = "15m"  # "1m", "3m", "5m", "15m", "Daily"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    start_time: Optional[dtime] = None  # e.g., 09:15
    end_time: Optional[dtime] = None    # e.g., 15:15
    setups: Set[str] = field(default_factory=lambda: {
        "bullish_sweep", "bearish_sweep", "bullish_ob_tap", "bearish_ob_tap", "realign"
    })
    exit_mode: str = "bars"  # "bars", "rr", "eod"
    hold_bars: int = 8
    rr_ratio: float = 2.0
    vol_surge_threshold: float = 1.0  # Require volume surge >= threshold
    trend_filter: bool = False       # Filter signals with 50 EMA


class SMCBacktestService:
    """Backend engine for running user-configured SMC backtests."""

    def __init__(self, db_path: str = "data/trade_system.db"):
        self.db_path = db_path
        self.structure_engine = SMCStructureEngine(swing_length=5, internal_length=2)

    def get_available_symbols(self) -> List[str]:
        """Fetch all unique symbols available in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql("SELECT DISTINCT symbol FROM ohlcv_daily ORDER BY symbol ASC", conn)
                return df["symbol"].tolist()
        except Exception:
            return TOP_25_SYMBOLS

    def run_backtest(
        self,
        params: SMCBacktestParams,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute parameterized backtest and return structured performance metrics."""
        table_map = {
            "1m": "ohlcv_1m",
            "3m": "ohlcv_3m",
            "5m": "ohlcv_5m",
            "15m": "ohlcv_15m",
            "Daily": "ohlcv_daily",
        }
        table_name = table_map.get(params.timeframe, "ohlcv_15m")
        is_daily = params.timeframe == "Daily"

        trades: List[Dict[str, Any]] = []
        symbols = params.symbols if params.symbols else TOP_10_SYMBOLS
        total_symbols = len(symbols)

        with sqlite3.connect(self.db_path) as conn:
            for s_idx, sym in enumerate(symbols):
                if progress_callback:
                    pct = (s_idx / max(total_symbols, 1))
                    clean_name = sym.replace("NSE:", "").replace("-EQ", "")
                    progress_callback(pct, f"Simulating {clean_name} ({s_idx + 1}/{total_symbols})...")

                # Fetch candle series
                q = f"SELECT timestamp, open, high, low, close, volume FROM {table_name} WHERE symbol = ? ORDER BY timestamp ASC"
                df = pd.read_sql(q, conn, params=(sym,))
                if df.empty or len(df) < 50:
                    continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
                
                # Filter Date Range if provided
                if params.start_date:
                    df = df[df["timestamp"].dt.date >= params.start_date]
                if params.end_date:
                    df = df[df["timestamp"].dt.date <= params.end_date]
                if len(df) < 40:
                    continue

                # Add moving average for trend filter and volume baseline
                if params.trend_filter:
                    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
                vol_sma = df["volume"].rolling(20).mean().replace(0, 1)
                df["vol_surge"] = df["volume"] / vol_sma

                # Execute backtest on the asset
                sym_trades = self._simulate_symbol(df, sym, params, is_daily)
                trades.extend(sym_trades)

        if progress_callback:
            progress_callback(1.0, f"Completed! Evaluated {len(trades)} setups.")

        return self._compile_results(trades)

    def _simulate_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: SMCBacktestParams,
        is_daily: bool,
    ) -> List[Dict[str, Any]]:
        trades = []
        n = len(df)
        max_fwd = max(params.hold_bars, 20)
        clean_sym = symbol.replace("NSE:", "").replace("-EQ", "")

        # Step through candles
        for i in range(30, n - max_fwd):
            curr = df.iloc[i]
            curr_ts = curr["timestamp"]

            # Filter Time Slot (if Intraday)
            if not is_daily:
                bar_time = curr_ts.time()
                if params.start_time and bar_time < params.start_time:
                    continue
                if params.end_time and bar_time > params.end_time:
                    continue

            # Volume filter
            if params.vol_surge_threshold > 1.0 and curr["vol_surge"] < params.vol_surge_threshold:
                continue

            past_df = df.iloc[max(0, i - 40) : i + 1]
            prev_candle = past_df.iloc[-2] if len(past_df) >= 2 else curr
            c_open = float(curr["open"])
            c_high = float(curr["high"])
            c_low = float(curr["low"])
            c_close = float(curr["close"])
            c_range = c_high - c_low
            if c_range <= 0:
                continue
            c_body = abs(c_close - c_open)
            c_upper_wick = c_high - max(c_close, c_open)
            c_lower_wick = min(c_close, c_open) - c_low

            sig_name = None
            direction = None  # 1 = Long, -1 = Short
            sl_level = None
            tp_level = None

            # --- 1. Liquidity Sweeps with Rejection Wick (>= 35% range and >= body) ---
            prior_low = float(past_df["low"].iloc[-21:-1].min())
            prior_high = float(past_df["high"].iloc[-21:-1].max())

            if "bullish_sweep" in params.setups and (c_low < prior_low and c_close > prior_low and c_close > c_open and (c_lower_wick / c_range) >= 0.35 and c_lower_wick >= c_body):
                if not params.trend_filter or (c_close > curr.get("ema50", 0)):
                    sig_name = "🟢 Bullish Sweep & Reclaim"
                    direction = 1
                    sl_level = c_low * 0.998
                    tp_level = prior_high

            elif "bearish_sweep" in params.setups and (c_high > prior_high and c_close < prior_high and c_close < c_open and (c_upper_wick / c_range) >= 0.35 and c_upper_wick >= c_body):
                if not params.trend_filter or (c_close < curr.get("ema50", 999999)):
                    sig_name = "🔴 Bearish Sweep & Reject"
                    direction = -1
                    sl_level = c_high * 1.002
                    tp_level = prior_low

            # --- 2. Order Block Taps (Fresh touch from outside & confirming reaction) ---
            if not sig_name and ("bullish_ob_tap" in params.setups or "bearish_ob_tap" in params.setups):
                obs = identify_order_blocks(past_df.iloc[:-1], atr_multiplier=1.2)
                if "bullish_ob_tap" in params.setups:
                    for ob in obs.get("bullish", []):
                        if prev_candle["low"] > ob["top"] and c_low <= ob["top"] and c_close >= ob["bottom"]:
                            if c_close >= c_open or c_lower_wick >= c_body * 0.8:
                                if not params.trend_filter or (c_close > curr.get("ema50", 0)):
                                    sig_name = "🟢 Bullish OB Tap & Bounce"
                                    direction = 1
                                    sl_level = ob["bottom"] * 0.998
                                    tp_level = prior_high
                                    break

                if not sig_name and "bearish_ob_tap" in params.setups:
                    for ob in obs.get("bearish", []):
                        if prev_candle["high"] < ob["bottom"] and c_high >= ob["bottom"] and c_close <= ob["top"]:
                            if c_close <= c_open or c_upper_wick >= c_body * 0.8:
                                if not params.trend_filter or (c_close < curr.get("ema50", 999999)):
                                    sig_name = "🔴 Bearish OB Tap & Reject"
                                    direction = -1
                                    sl_level = ob["top"] * 1.002
                                    tp_level = prior_low
                                    break

            # --- 3. Internal CHoCH Realignments (Triggered on exact bar of realignment) ---
            if not sig_name and "realign" in params.setups:
                state = self.structure_engine.analyze(past_df)
                if state.is_internal_realigned and state.realignment_event and state.realignment_event.bar_idx == len(past_df) - 1:
                    if state.swing_trend.value == "BULLISH":
                        sig_name = "🟢 Bullish Realignment (Internal CHoCH)"
                        direction = 1
                        sl_level = state.strong_protected_level or (c_close * 0.98)
                        tp_level = state.weak_target_level or (c_close * 1.04)
                    elif state.swing_trend.value == "BEARISH":
                        sig_name = "🔴 Bearish Realignment (Internal CHoCH)"
                        direction = -1
                        sl_level = state.strong_protected_level or (c_close * 1.02)
                        tp_level = state.weak_target_level or (c_close * 0.96)

            # If trigger validated, simulate exit mechanics
            if sig_name and direction:
                entry_price = c_close
                exit_price = entry_price
                exit_reason = "Hold Limit"
                bars_held = params.hold_bars
                exit_ts = df.iloc[min(i + params.hold_bars, n - 1)]["timestamp"]

                # Adjust TP based on RR if in RR exit mode
                if params.exit_mode == "rr" and sl_level:
                    risk = abs(entry_price - sl_level)
                    if risk > 0:
                        tp_level = entry_price + (params.rr_ratio * risk) if direction == 1 else entry_price - (params.rr_ratio * risk)

                curr_date = curr_ts.date()

                for fwd in range(1, params.hold_bars + 1):
                    f_bar = df.iloc[i + fwd]
                    f_high = float(f_bar["high"])
                    f_low = float(f_bar["low"])
                    f_close = float(f_bar["close"])

                    # Intraday EOD Cutoff
                    if not is_daily and f_bar["timestamp"].date() != curr_date:
                        exit_price = df.iloc[i + fwd - 1]["close"]
                        exit_reason = "EOD Cutoff"
                        exit_ts = df.iloc[i + fwd - 1]["timestamp"]
                        bars_held = fwd - 1
                        break

                    # Check SL
                    if sl_level is not None:
                        if direction == 1 and f_low <= sl_level:
                            exit_price = sl_level
                            exit_reason = "Stop Loss"
                            exit_ts = f_bar["timestamp"]
                            bars_held = fwd
                            break
                        elif direction == -1 and f_high >= sl_level:
                            exit_price = sl_level
                            exit_reason = "Stop Loss"
                            exit_ts = f_bar["timestamp"]
                            bars_held = fwd
                            break

                    # Check TP
                    if tp_level is not None:
                        if direction == 1 and f_high >= tp_level:
                            exit_price = tp_level
                            exit_reason = "Target Profit"
                            exit_ts = f_bar["timestamp"]
                            bars_held = fwd
                            break
                        elif direction == -1 and f_low <= tp_level:
                            exit_price = tp_level
                            exit_reason = "Target Profit"
                            exit_ts = f_bar["timestamp"]
                            bars_held = fwd
                            break

                    # If reached hold bar limit
                    if fwd == params.hold_bars:
                        exit_price = f_close
                        exit_reason = f"Max Bars ({params.hold_bars})"
                        exit_ts = f_bar["timestamp"]
                        bars_held = fwd

                pnl_pct = ((exit_price - entry_price) / entry_price * 100) if direction == 1 else ((entry_price - exit_price) / entry_price * 100)

                trades.append({
                    "timestamp": curr_ts,
                    "symbol": clean_sym,
                    "direction": "🟢 LONG" if direction == 1 else "🔴 SHORT",
                    "signal": sig_name,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "exit_timestamp": exit_ts,
                    "bars_held": bars_held,
                })

        return trades

    def _compile_results(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate trade records into statistical KPIs and equity curve."""
        if not trades:
            return {
                "trades_df": pd.DataFrame(),
                "summary": {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_trade_pnl": 0.0,
                    "profit_factor": 0.0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "max_drawdown": 0.0,
                },
                "by_setup": pd.DataFrame(),
                "equity_curve": pd.DataFrame(),
            }

        tdf = pd.DataFrame(trades).sort_values("timestamp").reset_index(drop=True)
        tdf["is_win"] = tdf["pnl_pct"] > 0

        tot = len(tdf)
        wins = int(tdf["is_win"].sum())
        losses = tot - wins
        win_rate = (wins / tot * 100) if tot > 0 else 0.0
        tot_pnl = float(tdf["pnl_pct"].sum())
        avg_pnl = float(tdf["pnl_pct"].mean())

        pos_sum = float(tdf[tdf["pnl_pct"] > 0]["pnl_pct"].sum())
        neg_sum = abs(float(tdf[tdf["pnl_pct"] <= 0]["pnl_pct"].sum()))
        profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else 999.0

        # Equity Curve and Max Drawdown
        tdf["cum_pnl"] = tdf["pnl_pct"].cumsum()
        running_max = tdf["cum_pnl"].cummax()
        drawdown = running_max - tdf["cum_pnl"]
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

        # Summary by Setup
        by_setup = []
        for sig, grp in tdf.groupby("signal"):
            g_wins = (grp["pnl_pct"] > 0).sum()
            g_tot = len(grp)
            by_setup.append({
                "Setup": sig,
                "Trades": g_tot,
                "Win Rate": f"{(g_wins / g_tot * 100):.1f}%",
                "Avg Return": f"{grp['pnl_pct'].mean():+.2f}%",
                "Total PnL": f"{grp['pnl_pct'].sum():+.2f}%",
            })
        by_setup_df = pd.DataFrame(by_setup)

        return {
            "trades_df": tdf,
            "summary": {
                "total_trades": tot,
                "win_rate": win_rate,
                "total_pnl": tot_pnl,
                "avg_trade_pnl": avg_pnl,
                "profit_factor": profit_factor,
                "winning_trades": wins,
                "losing_trades": losses,
                "max_drawdown": max_drawdown,
            },
            "by_setup": by_setup_df,
            "equity_curve": tdf[["timestamp", "cum_pnl"]].copy(),
        }
