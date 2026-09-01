"""
Quantitative Performance Analytics & Metrics Engine.

Calculates institutional risk-adjusted metrics:
- CAGR %, Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown (MDD %, MDD ₹, Drawdown Duration in days)
- Profit Factor, Expectancy (₹ and R-Multiples), Win/Loss Ratios
- Attribution Breakdown (by Pattern, Sector, Confidence Tier, Market Regime)
- Monthly & Yearly Return Matrix
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BacktestTradeRecord:
    trade_id: int
    symbol: str
    sector: str
    direction: str  # "CALL" or "PUT"
    entry_date: date
    exit_date: date
    holding_days: int
    entry_price: float
    exit_price: float
    shares: int
    capital_invested: float
    stop_loss: float
    target_1: float
    target_2: float
    gross_pnl: float
    costs: float  # Brokerage + STT + Slippage
    net_pnl: float
    return_pct: float
    r_multiple: float
    exit_reason: str  # "TARGET_1", "TARGET_2", "STOP_LOSS", "TIME_EXPIRY", "TRAILING_SL"
    confidence: str
    probability: int
    confluences: List[str] = field(default_factory=list)
    mfe_pct: float = 0.0
    mae_pct: float = 0.0


@dataclass
class QuantitativeReport:
    # Capital & Return
    initial_capital: float
    final_equity: float
    net_profit: float
    total_return_pct: float
    cagr_pct: float
    
    # Win / Loss Statistics
    total_signals: int
    triggered_trades: int
    trigger_rate_pct: float
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    loss_rate_pct: float
    win_loss_ratio: float
    
    # Trade PnL Dynamics
    profit_factor: float
    expectancy_inr: float
    expectancy_r: float
    avg_trade_pnl: float
    avg_win_pnl: float
    avg_loss_pnl: float
    avg_return_pct: float
    avg_r_multiple: float
    avg_holding_days: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Risk-Adjusted Volatility & Drawdown
    max_drawdown_pct: float
    max_drawdown_inr: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Tables & DataFrames
    equity_curve: pd.DataFrame
    trades_df: pd.DataFrame
    monthly_returns: pd.DataFrame
    confluence_attribution: pd.DataFrame
    sector_attribution: pd.DataFrame
    confidence_attribution: pd.DataFrame


class PerformanceAnalytics:
    """Calculates comprehensive quantitative risk-adjusted performance metrics."""

    @staticmethod
    def compute(
        trades: List[BacktestTradeRecord],
        daily_equity_series: pd.DataFrame,
        initial_capital: float,
        total_signals_generated: int = 0
    ) -> QuantitativeReport:
        if not trades:
            empty_df = pd.DataFrame()
            return QuantitativeReport(
                initial_capital=initial_capital,
                final_equity=initial_capital,
                net_profit=0.0,
                total_return_pct=0.0,
                cagr_pct=0.0,
                total_signals=total_signals_generated,
                triggered_trades=0,
                trigger_rate_pct=0.0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                loss_rate_pct=0.0,
                win_loss_ratio=0.0,
                profit_factor=0.0,
                expectancy_inr=0.0,
                expectancy_r=0.0,
                avg_trade_pnl=0.0,
                avg_win_pnl=0.0,
                avg_loss_pnl=0.0,
                avg_return_pct=0.0,
                avg_r_multiple=0.0,
                avg_holding_days=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                max_drawdown_pct=0.0,
                max_drawdown_inr=0.0,
                max_drawdown_duration_days=0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                equity_curve=daily_equity_series,
                trades_df=empty_df,
                monthly_returns=empty_df,
                confluence_attribution=empty_df,
                sector_attribution=empty_df,
                confidence_attribution=empty_df
            )

        df_trades = pd.DataFrame([t.__dict__ for t in trades])
        n_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        n_wins = len(wins)
        n_losses = len(losses)

        total_net_pnl = sum(t.net_pnl for t in trades)
        final_equity = initial_capital + total_net_pnl
        total_return_pct = (final_equity - initial_capital) / initial_capital * 100.0

        # Holding period & CAGR
        if not daily_equity_series.empty and len(daily_equity_series) > 1:
            total_days = max(1, (daily_equity_series["date"].max() - daily_equity_series["date"].min()).days)
            years = max(total_days / 365.25, 0.05)
            cagr = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0
        else:
            total_days = 30
            cagr = total_return_pct

        win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
        loss_rate = (n_losses / n_trades * 100.0) if n_trades > 0 else 0.0
        trigger_rate = (n_trades / total_signals_generated * 100.0) if total_signals_generated > 0 else 100.0

        gross_profits = sum(t.net_pnl for t in wins)
        gross_losses = abs(sum(t.net_pnl for t in losses))
        profit_factor = round(gross_profits / gross_losses, 2) if gross_losses > 0 else (999.0 if gross_profits > 0 else 0.0)

        avg_win = (gross_profits / n_wins) if n_wins > 0 else 0.0
        avg_loss = (gross_losses / n_losses) if n_losses > 0 else 0.0
        win_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
        expectancy_inr = round((win_rate / 100.0 * avg_win) - (loss_rate / 100.0 * avg_loss), 2)
        avg_r = float(df_trades["r_multiple"].mean()) if "r_multiple" in df_trades.columns else 0.0
        expectancy_r = round(avg_r, 2)

        # Consecutive streaks
        max_c_wins = 0
        max_c_losses = 0
        curr_w = 0
        curr_l = 0
        for t in trades:
            if t.net_pnl > 0:
                curr_w += 1
                curr_l = 0
                max_c_wins = max(max_c_wins, curr_w)
            else:
                curr_l += 1
                curr_w = 0
                max_c_losses = max(max_c_losses, curr_l)

        # Drawdown Calculations
        max_dd_pct = 0.0
        max_dd_inr = 0.0
        max_dd_duration = 0
        sharpe = 0.0
        sortino = 0.0
        calmar = 0.0

        if not daily_equity_series.empty:
            eq = daily_equity_series["equity"].copy()
            peak = eq.cummax()
            dd_inr = peak - eq
            dd_pct = (dd_inr / peak) * 100.0
            max_dd_pct = round(float(dd_pct.max()), 2)
            max_dd_inr = round(float(dd_inr.max()), 2)

            # Drawdown duration in days
            is_dd = dd_inr > 0
            dd_groups = (~is_dd).cumsum()
            dd_durations = is_dd.groupby(dd_groups).sum()
            max_dd_duration = int(dd_durations.max()) if not dd_durations.empty else 0

            # Daily Returns & Sharpe/Sortino
            daily_returns = eq.pct_change().dropna()
            if not daily_returns.empty and daily_returns.std() > 0:
                rf_daily = 0.06 / 252  # 6% risk free rate
                excess_returns = daily_returns - rf_daily
                sharpe = round(float(excess_returns.mean() / daily_returns.std() * np.sqrt(252)), 2)
                downside = daily_returns[daily_returns < 0].std()
                sortino = round(float(excess_returns.mean() / downside * np.sqrt(252)), 2) if downside > 0 else 0.0

            calmar = round(cagr / max_dd_pct, 2) if max_dd_pct > 0 else 0.0

        # Monthly Returns Table
        if not daily_equity_series.empty:
            df_eq = daily_equity_series.copy()
            df_eq["date"] = pd.to_datetime(df_eq["date"])
            df_eq["year"] = df_eq["date"].dt.year
            df_eq["month"] = df_eq["date"].dt.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            monthly_table = []
            for (yr, mo), grp in df_eq.groupby(["year", "month"]):
                start_val = grp["equity"].iloc[0]
                end_val = grp["equity"].iloc[-1]
                m_ret = (end_val - start_val) / start_val * 100.0
                monthly_table.append({"Year": yr, "Month": mo, "Return %": round(m_ret, 2)})
            df_monthly = pd.DataFrame(monthly_table)
        else:
            df_monthly = pd.DataFrame()

        # Confluence Attribution
        confluence_stats: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            for c in t.confluences:
                if c not in confluence_stats:
                    confluence_stats[c] = {"Trades": 0, "Wins": 0, "Losses": 0, "Net PnL": 0.0, "Total R": 0.0}
                confluence_stats[c]["Trades"] += 1
                if t.net_pnl > 0:
                    confluence_stats[c]["Wins"] += 1
                else:
                    confluence_stats[c]["Losses"] += 1
                confluence_stats[c]["Net PnL"] += t.net_pnl
                confluence_stats[c]["Total R"] += t.r_multiple

        confluence_rows = []
        for c, st in confluence_stats.items():
            if st["Trades"] >= 3:
                c_wr = round(st["Wins"] / st["Trades"] * 100.0, 1)
                c_avg_r = round(st["Total R"] / st["Trades"], 2)
                confluence_rows.append({
                    "Confluence": c,
                    "Trades": st["Trades"],
                    "Win Rate %": c_wr,
                    "Net PnL (₹)": round(st["Net PnL"], 2),
                    "Avg R": c_avg_r
                })
        df_confluence = pd.DataFrame(confluence_rows)
        if not df_confluence.empty:
            df_confluence.sort_values(by="Win Rate %", ascending=False, inplace=True)

        # Sector Attribution
        sector_rows = []
        for sec, grp in df_trades.groupby("sector"):
            s_trades = len(grp)
            s_wins = len(grp[grp["net_pnl"] > 0])
            s_wr = round(s_wins / s_trades * 100.0, 1)
            s_pnl = round(float(grp["net_pnl"].sum()), 2)
            sector_rows.append({
                "Sector": sec,
                "Trades": s_trades,
                "Win Rate %": s_wr,
                "Net PnL (₹)": s_pnl
            })
        df_sector = pd.DataFrame(sector_rows).sort_values(by="Net PnL (₹)", ascending=False) if sector_rows else pd.DataFrame()

        # Confidence Attribution
        conf_rows = []
        for conf, grp in df_trades.groupby("confidence"):
            c_trades = len(grp)
            c_wins = len(grp[grp["net_pnl"] > 0])
            c_wr = round(c_wins / c_trades * 100.0, 1)
            c_pnl = round(float(grp["net_pnl"].sum()), 2)
            c_pf = round(abs(grp[grp["net_pnl"] > 0]["net_pnl"].sum() / grp[grp["net_pnl"] < 0]["net_pnl"].sum()), 2) if len(grp[grp["net_pnl"] < 0]) > 0 else 999.0
            conf_rows.append({
                "Tier": conf,
                "Trades": c_trades,
                "Win Rate %": c_wr,
                "Profit Factor": c_pf,
                "Net PnL (₹)": c_pnl
            })
        df_conf = pd.DataFrame(conf_rows).sort_values(by="Win Rate %", ascending=False) if conf_rows else pd.DataFrame()

        return QuantitativeReport(
            initial_capital=initial_capital,
            final_equity=round(final_equity, 2),
            net_profit=round(total_net_pnl, 2),
            total_return_pct=round(total_return_pct, 2),
            cagr_pct=round(cagr, 2),
            total_signals=total_signals_generated,
            triggered_trades=n_trades,
            trigger_rate_pct=round(trigger_rate, 1),
            winning_trades=n_wins,
            losing_trades=n_losses,
            win_rate_pct=round(win_rate, 1),
            loss_rate_pct=round(loss_rate, 1),
            win_loss_ratio=win_loss_ratio,
            profit_factor=profit_factor,
            expectancy_inr=expectancy_inr,
            expectancy_r=expectancy_r,
            avg_trade_pnl=round(total_net_pnl / n_trades, 2),
            avg_win_pnl=round(avg_win, 2),
            avg_loss_pnl=round(avg_loss, 2),
            avg_return_pct=round(float(df_trades["return_pct"].mean()), 2),
            avg_r_multiple=round(avg_r, 2),
            avg_holding_days=round(float(df_trades["holding_days"].mean()), 1),
            max_consecutive_wins=max_c_wins,
            max_consecutive_losses=max_c_losses,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_inr=max_dd_inr,
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            equity_curve=daily_equity_series,
            trades_df=df_trades,
            monthly_returns=df_monthly,
            confluence_attribution=df_confluence,
            sector_attribution=df_sector,
            confidence_attribution=df_conf
        )
