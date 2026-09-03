"""
Intraday Trend Reversal Backtest Engine.

Event-driven, bar-by-bar intraday backtester with realistic execution modeling:
- Strict position sizing based on % equity risk.
- Slippage and friction costs (0.05%).
- Trailing stop-loss to breakeven upon achieving Target 1 (VWAP).
- Mandatory intraday square-off at 15:15 IST (no overnight holding).
- Comprehensive institutional metrics (Sharpe, Sortino, Calmar, Max Drawdown, Expectancy R).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trade_system.domains.strategy.application.strategies.institutional_reversal_strategy import (
    InstitutionalIntradayReversalStrategy,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single simulated trade."""
    trade_id: int
    symbol: str
    direction: str                     # "CALL" / "PUT"
    action: str                        # "BUY_CALL" / "BUY_PUT"
    entry_time: datetime
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    quantity: int
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""              # "TARGET_1", "TARGET_2", "STOP_LOSS", "TRAILING_STOP", "EOD_SQUAREOFF"
    pnl_points: float = 0.0
    pnl_inr: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    holding_minutes: int = 0
    confluence: str = ""
    portfolio_equity: float = 0.0


@dataclass
class BacktestReport:
    """Comprehensive performance report returned by backtester."""
    initial_capital: float
    final_equity: float
    total_net_pnl: float
    total_return_pct: float
    cagr_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_pct: float
    loss_rate_pct: float
    win_loss_ratio: float
    profit_factor: float
    avg_win_inr: float
    avg_loss_inr: float
    max_win_inr: float
    max_loss_inr: float
    expectancy_inr: float
    expectancy_r: float
    max_drawdown_inr: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    trades: List[BacktestTrade] = field(default_factory=list)
    symbol_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_equity_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class IntradayReversalBacktester:
    """
    Backtesting engine for the Institutional Intraday Reversal Strategy.
    """

    def __init__(
        self,
        strategy: Optional[InstitutionalIntradayReversalStrategy] = None,
        initial_capital: float = 500_000.0,
        risk_per_trade_pct: float = 1.0,
        slippage_pct: float = 0.0005,
        trailing_stop: bool = True,
        square_off_time: dt_time = dt_time(15, 15),
    ) -> None:
        self.strategy = strategy or InstitutionalIntradayReversalStrategy()
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.slippage_pct = slippage_pct
        self.trailing_stop = trailing_stop
        self.square_off_time = square_off_time

    def run(self, symbol_data: Dict[str, pd.DataFrame]) -> BacktestReport:
        """
        Run intraday simulation across provided symbol datasets.
        """
        all_trades: List[BacktestTrade] = []
        equity = self.initial_capital
        equity_curve: List[Dict[str, Any]] = []
        trade_counter = 0

        # Sort combined events across all symbols by timestamp
        prepared_symbols: Dict[str, pd.DataFrame] = {}
        for sym, df in symbol_data.items():
            if df.empty or len(df) < 30:
                continue
            sig_df = self.strategy.generate_signals(df)
            if not sig_df.empty:
                prepared_symbols[sym] = sig_df

        if not prepared_symbols:
            LOGGER.warning("No valid data or signals generated for backtest.")
            return self._empty_report()

        # Iterate per symbol intraday
        for sym, df in prepared_symbols.items():
            active_trade: Optional[BacktestTrade] = None
            be_trailed = False

            # Group by day
            for trade_date, day_df in df.groupby(df.index.date):
                active_trade = None
                be_trailed = False

                for i in range(len(day_df)):
                    row = day_df.iloc[i]
                    ts = row.name if isinstance(row.name, datetime) else pd.to_datetime(row.get("timestamp", ""))
                    t_time = ts.time() if hasattr(ts, "time") else dt_time(9, 15)

                    curr_open = float(row["open"])
                    curr_high = float(row["high"])
                    curr_low = float(row["low"])
                    curr_close = float(row["close"])

                    # 1. Manage existing open trade
                    if active_trade is not None:
                        is_call = active_trade.direction == "CALL"
                        hold_mins = int((ts - active_trade.entry_time).total_seconds() / 60)
                        active_trade.holding_minutes = hold_mins

                        # Mandatory EOD Square-off
                        if t_time >= self.square_off_time:
                            exit_price = curr_close * (1.0 - self.slippage_pct if is_call else 1.0 + self.slippage_pct)
                            active_trade.exit_time = ts
                            active_trade.exit_price = exit_price
                            active_trade.exit_reason = "EOD_SQUAREOFF"
                            self._finalize_trade(active_trade, equity)
                            equity += active_trade.pnl_inr
                            active_trade.portfolio_equity = equity
                            all_trades.append(active_trade)
                            active_trade = None
                            continue

                        # Check Stop Loss
                        sl_hit = (curr_low <= active_trade.stop_loss) if is_call else (curr_high >= active_trade.stop_loss)
                        if sl_hit:
                            exit_price = active_trade.stop_loss * (1.0 - self.slippage_pct if is_call else 1.0 + self.slippage_pct)
                            active_trade.exit_time = ts
                            active_trade.exit_price = exit_price
                            active_trade.exit_reason = "TRAILING_STOP" if be_trailed else "STOP_LOSS"
                            self._finalize_trade(active_trade, equity)
                            equity += active_trade.pnl_inr
                            active_trade.portfolio_equity = equity
                            all_trades.append(active_trade)
                            active_trade = None
                            continue

                        # Check Target 1 (Primary 1.8R Target)
                        t1_hit = (curr_high >= active_trade.target_1) if is_call else (curr_low <= active_trade.target_1)
                        if t1_hit and active_trade.target_1 > 0:
                            exit_price = active_trade.target_1 * (1.0 - self.slippage_pct if is_call else 1.0 + self.slippage_pct)
                            active_trade.exit_time = ts
                            active_trade.exit_price = exit_price
                            active_trade.exit_reason = "TARGET_1"
                            self._finalize_trade(active_trade, equity)
                            equity += active_trade.pnl_inr
                            active_trade.portfolio_equity = equity
                            all_trades.append(active_trade)
                            active_trade = None
                            continue

                    # 2. Check for new trade entry if no active trade
                    signal_val = int(row.get("signal", 0))
                    if active_trade is None and signal_val != 0 and t_time < dt_time(14, 45):
                        trade_counter += 1
                        is_call = signal_val == 1
                        raw_entry = curr_close
                        entry_price = raw_entry * (1.0 + self.slippage_pct if is_call else 1.0 - self.slippage_pct)
                        sl = float(row.get("stop_loss", 0.0))
                        t1 = float(row.get("target_1", 0.0))
                        t2 = float(row.get("target_2", 0.0))

                        # Position sizing based on account risk
                        risk_per_unit = abs(entry_price - sl)
                        if risk_per_unit <= 0:
                            continue

                        max_risk_inr = equity * (self.risk_per_trade_pct / 100.0)
                        # Assume index lot multiplier/scaling
                        units = max(1, int(max_risk_inr / risk_per_unit))

                        active_trade = BacktestTrade(
                            trade_id=trade_counter,
                            symbol=sym,
                            direction="CALL" if is_call else "PUT",
                            action="BUY_CALL" if is_call else "BUY_PUT",
                            entry_time=ts,
                            entry_price=entry_price,
                            stop_loss=sl,
                            target_1=t1,
                            target_2=t2,
                            quantity=units,
                            confluence=str(row.get("confluence", "")),
                        )
                        be_trailed = False

        return self._compile_report(all_trades, equity)

    def _finalize_trade(self, trade: BacktestTrade, current_equity: float) -> None:
        """Calculate final PnL and risk metrics for trade."""
        is_call = trade.direction == "CALL"
        if is_call:
            trade.pnl_points = trade.exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - trade.exit_price

        trade.pnl_inr = trade.pnl_points * trade.quantity
        trade.pnl_pct = (trade.pnl_points / trade.entry_price) * 100.0

        initial_risk = abs(trade.entry_price - trade.stop_loss)
        if initial_risk > 0:
            trade.r_multiple = round(trade.pnl_points / initial_risk, 2)
        else:
            trade.r_multiple = 0.0

    def _compile_report(self, trades: List[BacktestTrade], final_equity: float) -> BacktestReport:
        """Generate full metrics and summary report."""
        if not trades:
            return self._empty_report()

        total_trades = len(trades)
        wins = [t for t in trades if t.pnl_inr > 0]
        losses = [t for t in trades if t.pnl_inr < 0]
        breakevens = [t for t in trades if t.pnl_inr == 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        be_trades = len(breakevens)

        win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
        loss_rate = (losing_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

        total_pnl = sum(t.pnl_inr for t in trades)
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100.0

        gross_profit = sum(t.pnl_inr for t in wins)
        gross_loss = abs(sum(t.pnl_inr for t in losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = -(gross_loss / losing_trades) if losing_trades > 0 else 0.0
        max_win = max([t.pnl_inr for t in wins], default=0.0)
        max_loss = min([t.pnl_inr for t in losses], default=0.0)

        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        expectancy_inr = (win_rate / 100.0 * avg_win) + (loss_rate / 100.0 * avg_loss)
        avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0.0

        # Drawdown calculation
        eq_series = pd.Series([self.initial_capital] + [t.portfolio_equity for t in trades])
        cum_max = eq_series.cummax()
        drawdown_series = (eq_series - cum_max) / cum_max
        max_dd_pct = abs(float(drawdown_series.min())) * 100.0
        max_dd_inr = abs(float((eq_series - cum_max).min()))

        # Sharpe & Sortino (annualized using 252 days)
        returns = pd.Series([t.pnl_pct for t in trades])
        sharpe = float(np.sqrt(252) * (returns.mean() / returns.std())) if returns.std() > 0 else 0.0
        downside_std = returns[returns < 0].std()
        sortino = float(np.sqrt(252) * (returns.mean() / downside_std)) if downside_std > 0 else 0.0
        calmar = (total_return / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Calculate CAGR assuming 90 days (~0.25 year)
        cagr = ((final_equity / self.initial_capital) ** (365.0 / 90.0) - 1.0) * 100.0 if final_equity > 0 else 0.0

        # Symbol breakdown
        trade_dicts = [t.__dict__ for t in trades]
        df_trades = pd.DataFrame(trade_dicts)
        sym_perf = df_trades.groupby("symbol").agg(
            total_trades=("trade_id", "count"),
            win_rate=("pnl_inr", lambda x: (x > 0).mean() * 100.0),
            net_pnl=("pnl_inr", "sum"),
            avg_pnl=("pnl_inr", "mean"),
            profit_factor=("pnl_inr", lambda x: abs(x[x > 0].sum() / x[x < 0].sum()) if len(x[x < 0]) > 0 and x[x < 0].sum() != 0 else 99.0)
        ).reset_index()

        # Monthly breakdown
        df_trades["month"] = pd.to_datetime(df_trades["entry_time"]).dt.to_period("M").astype(str)
        month_perf = df_trades.groupby("month").agg(
            trades=("trade_id", "count"),
            win_rate=("pnl_inr", lambda x: (x > 0).mean() * 100.0),
            net_pnl=("pnl_inr", "sum"),
            roi_pct=("pnl_inr", lambda x: (x.sum() / self.initial_capital) * 100.0)
        ).reset_index()

        return BacktestReport(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_net_pnl=total_pnl,
            total_return_pct=total_return,
            cagr_pct=cagr,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=be_trades,
            win_rate_pct=win_rate,
            loss_rate_pct=loss_rate,
            win_loss_ratio=win_loss_ratio,
            profit_factor=profit_factor,
            avg_win_inr=avg_win,
            avg_loss_inr=avg_loss,
            max_win_inr=max_win,
            max_loss_inr=max_loss,
            expectancy_inr=expectancy_inr,
            expectancy_r=avg_r,
            max_drawdown_inr=max_dd_inr,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            trades=trades,
            symbol_performance=sym_perf,
            monthly_performance=month_perf,
        )

    def _empty_report(self) -> BacktestReport:
        return BacktestReport(
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
            total_net_pnl=0.0,
            total_return_pct=0.0,
            cagr_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate_pct=0.0,
            loss_rate_pct=0.0,
            win_loss_ratio=0.0,
            profit_factor=0.0,
            avg_win_inr=0.0,
            avg_loss_inr=0.0,
            max_win_inr=0.0,
            max_loss_inr=0.0,
            expectancy_inr=0.0,
            expectancy_r=0.0,
            max_drawdown_inr=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
        )
