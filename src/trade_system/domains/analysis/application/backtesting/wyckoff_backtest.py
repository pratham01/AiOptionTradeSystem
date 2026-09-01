"""
Wyckoff & VSA Backtesting Engine
================================
Validates Wyckoff Spring, UTAD, and Volume-Spread Analysis setups over multi-year historical data.
Computes institutional performance metrics: Win Rate, Profit Factor, Expectancy, Max Drawdown, and PnL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from trade_system.domains.strategy.application.strategies.wyckoff_strategy import WyckoffVsaStrategy, WyckoffSetup
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)


@dataclass
class WyckoffTrade:
    """Historical trade executed during backtest simulation."""
    symbol: str
    action: str
    direction: int
    pattern_type: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_1: float
    exit_date: str
    exit_price: float
    exit_reason: str              # "TARGET_HIT", "STOP_LOSS_HIT", "TIME_EXIT"
    points_captured: float
    pnl_pct: float
    r_multiple: float
    holding_bars: int


@dataclass
class WyckoffBacktestSummary:
    """Aggregated performance results from backtesting simulation."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    avg_trade_pnl_pct: float = 0.0
    avg_r_multiple: float = 0.0
    max_drawdown_pct: float = 0.0
    spring_trades_count: int = 0
    spring_win_rate: float = 0.0
    utad_trades_count: int = 0
    utad_win_rate: float = 0.0
    sos_trades_count: int = 0
    sos_win_rate: float = 0.0
    trades: List[WyckoffTrade] = field(default_factory=list)

    def summary_markdown(self) -> str:
        return f"""
# 🏛️ Institutional Wyckoff & VSA Strategy Backtest Results

## Performance Overview
- **Total Trades:** {self.total_trades}
- **Win Rate:** {self.win_rate_pct:.1f}% ({self.winning_trades} Wins / {self.losing_trades} Losses)
- **Profit Factor:** {self.profit_factor:.2f}
- **Average R-Multiple:** {self.avg_r_multiple:.2f}R
- **Total Cumulative Return:** {self.total_pnl_pct:.2f}%
- **Max Drawdown:** {self.max_drawdown_pct:.2f}%

## Performance by Pattern
- **Wyckoff Spring (Accumulation):** {self.spring_trades_count} trades | {self.spring_win_rate:.1f}% Win Rate
- **Wyckoff UTAD (Distribution):** {self.utad_trades_count} trades | {self.utad_win_rate:.1f}% Win Rate
- **Sign of Strength (SOS / SOW):** {self.sos_trades_count} trades | {self.sos_win_rate:.1f}% Win Rate
"""


class WyckoffBacktestEngine:
    """
    Simulates Wyckoff Spring / UTAD entries over historical daily candle sequences.
    """

    def __init__(
        self,
        max_holding_bars: int = 15,
        min_rrr: float = 1.5,
    ) -> None:
        self.strategy = WyckoffVsaStrategy(min_rrr=min_rrr)
        self.max_holding_bars = max_holding_bars

    def run_backtest_on_dataframe(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> List[WyckoffTrade]:
        """Runs rolling bar-by-bar simulation on a single symbol's historical data."""
        trades: List[WyckoffTrade] = []
        if df.empty or len(df) < 50:
            return trades

        clean_df = df.copy().sort_values("timestamp").reset_index(drop=True)
        n = len(clean_df)
        in_trade = False
        trade_exit_bar = -1

        for i in range(35, n - 2):
            if in_trade and i <= trade_exit_bar:
                continue

            window = clean_df.iloc[: i + 1]
            setup = self.strategy.analyze_setup(window, symbol=symbol)

            if setup:
                entry_bar = clean_df.iloc[i + 1]
                entry_date = str(entry_bar["timestamp"])
                entry_price = float(entry_bar["open"])  # Execute at next day open
                direction = setup.direction
                sl = setup.stop_loss
                target = setup.target_1

                # Forward simulate trade
                exit_price = entry_price
                exit_date = entry_date
                exit_reason = "TIME_EXIT"
                holding = 0

                for j in range(i + 1, min(n, i + 1 + self.max_holding_bars)):
                    holding += 1
                    bar = clean_df.iloc[j]
                    b_high = float(bar["high"])
                    b_low = float(bar["low"])
                    b_close = float(bar["close"])
                    b_date = str(bar["timestamp"])

                    if direction == 1:
                        # Bullish (Long)
                        if b_low <= sl:
                            exit_price = sl
                            exit_date = b_date
                            exit_reason = "STOP_LOSS_HIT"
                            break
                        elif b_high >= target:
                            exit_price = target
                            exit_date = b_date
                            exit_reason = "TARGET_HIT"
                            break
                    else:
                        # Bearish (Short)
                        if b_high >= sl:
                            exit_price = sl
                            exit_date = b_date
                            exit_reason = "STOP_LOSS_HIT"
                            break
                        elif b_low <= target:
                            exit_price = target
                            exit_date = b_date
                            exit_reason = "TARGET_HIT"
                            break

                    if holding >= self.max_holding_bars:
                        exit_price = b_close
                        exit_date = b_date
                        exit_reason = "TIME_EXIT"

                # Calculate PnL
                if direction == 1:
                    pts = exit_price - entry_price
                    pnl_pct = (pts / entry_price) * 100
                    risk = entry_price - sl
                    r_mult = pts / risk if risk > 0 else 0.0
                else:
                    pts = entry_price - exit_price
                    pnl_pct = (pts / entry_price) * 100
                    risk = sl - entry_price
                    r_mult = pts / risk if risk > 0 else 0.0

                trades.append(WyckoffTrade(
                    symbol=symbol,
                    action=setup.action,
                    direction=direction,
                    pattern_type=setup.pattern_type,
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    stop_loss=round(sl, 2),
                    target_1=round(target, 2),
                    exit_date=exit_date,
                    exit_price=round(exit_price, 2),
                    exit_reason=exit_reason,
                    points_captured=round(pts, 2),
                    pnl_pct=round(pnl_pct, 2),
                    r_multiple=round(r_mult, 2),
                    holding_bars=holding
                ))

                in_trade = True
                trade_exit_bar = i + holding

        return trades

    def evaluate_universe(self, symbols: Optional[List[str]] = None) -> WyckoffBacktestSummary:
        """Runs backtest across the F&O universe from the SQLite database."""
        engine = get_engine()
        from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
        sym_list = symbols or (["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"] + get_fo_universe()[:60]) # Top 60 liquid stocks for fast robust validation

        all_trades: List[WyckoffTrade] = []
        LOGGER.info("Starting Wyckoff backtest validation on %d symbols...", len(sym_list))

        with engine.connect() as conn:
            for sym in sym_list:
                try:
                    df = pd.read_sql(
                        __import__("sqlalchemy").text("SELECT timestamp, open, high, low, close, volume FROM ohlcv_daily WHERE symbol = :sym ORDER BY timestamp ASC"),
                        conn,
                        params={"sym": sym}
                    )
                    if not df.empty and len(df) >= 60:
                        trades = self.run_backtest_on_dataframe(df, symbol=sym)
                        all_trades.extend(trades)
                except Exception as exc:
                    LOGGER.debug("Backtest error for %s: %s", sym, exc)

        if not all_trades:
            return WyckoffBacktestSummary()

        wins = [t for t in all_trades if t.points_captured > 0]
        losses = [t for t in all_trades if t.points_captured <= 0]
        total_win_pts = sum(t.points_captured for t in wins)
        total_loss_pts = abs(sum(t.points_captured for t in losses))
        pf = total_win_pts / total_loss_pts if total_loss_pts > 0 else 99.0

        springs = [t for t in all_trades if "SPRING" in t.pattern_type]
        spring_wins = [t for t in springs if t.points_captured > 0]
        utads = [t for t in all_trades if "UTAD" in t.pattern_type]
        utad_wins = [t for t in utads if t.points_captured > 0]
        soss = [t for t in all_trades if "SOS" in t.pattern_type or "SOW" in t.pattern_type]
        sos_wins = [t for t in soss if t.points_captured > 0]

        # Drawdown calculation
        cumulative = np.cumsum([t.pnl_pct for t in all_trades])
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return WyckoffBacktestSummary(
            total_trades=len(all_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round((len(wins) / len(all_trades)) * 100, 2),
            profit_factor=round(pf, 2),
            total_pnl_pct=round(sum(t.pnl_pct for t in all_trades), 2),
            avg_trade_pnl_pct=round(float(np.mean([t.pnl_pct for t in all_trades])), 2),
            avg_r_multiple=round(float(np.mean([t.r_multiple for t in all_trades])), 2),
            max_drawdown_pct=round(max_dd, 2),
            spring_trades_count=len(springs),
            spring_win_rate=round((len(spring_wins) / len(springs) * 100) if springs else 0.0, 1),
            utad_trades_count=len(utads),
            utad_win_rate=round((len(utad_wins) / len(utads) * 100) if utads else 0.0, 1),
            sos_trades_count=len(soss),
            sos_win_rate=round((len(sos_wins) / len(soss) * 100) if soss else 0.0, 1),
            trades=all_trades
        )
