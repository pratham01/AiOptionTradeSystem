"""
Backtest Reporting & Export Engine.

Generates:
1. Terminal ASCII KPI Dashboards
2. GitHub Markdown Reports with Tables and Alerts
3. JSON & CSV File Exports
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


class BacktestReporter:
    """Generates formatted reports across terminal, markdown, JSON, and CSV."""

    @staticmethod
    def generate_ascii_summary(result: Any) -> str:
        m = result.metrics
        c = result.config
        lines = [
            "╔══════════════════════════════════════════════════════════════════════════════════════════╗",
            f"║                      QUANTITATIVE BACKTEST PERFORMANCE REPORT                            ║",
            f"║  Symbol: {c.symbol:<20} Timeframe: {c.timeframe:<8} Bars Analyzed: {result.candles_count:<18} ║",
            "╠══════════════════════════════════════════════════════════════════════════════════════════╣",
            "║  CAPITAL & RETURNS:                                                                      ║",
            f"║    Initial Capital : ₹{m['initial_capital']:>14,.2f}     Final Equity    : ₹{m['final_equity']:>14,.2f}  ║",
            f"║    Net Profit      : ₹{m['net_profit']:>14,.2f}     Total Return %  : {m['total_return_pct']:>13,.2f}%  ║",
            f"║    Profit Factor   : {m['profit_factor']:>15,.2f}      Expectancy (₹)  : ₹{m['expectancy_inr']:>13,.2f}  ║",
            "╠══════════════════════════════════════════════════════════════════════════════════════════╣",
            "║  TRADE COUNTS & WIN/LOSS RATIOS:                                                         ║",
            f"║    Total Trades    : {m['total_trades']:>15}      Win Rate %      : {m['win_rate_pct']:>13,.2f}%  ║",
            f"║    Winning Trades  : {m['winning_trades']:>15}      Loss Rate %     : {m['loss_rate_pct']:>13,.2f}%  ║",
            f"║    Losing Trades   : {m['losing_trades']:>15}      Breakeven Trades: {m['breakeven_trades']:>14}  ║",
            f"║    Payoff Ratio    : {m['payoff_ratio']:>15,.2f}      Avg R-Multiple  : {m['avg_r_multiple']:>14,.2f}  ║",
            f"║    Avg Win PnL     : ₹{m['avg_win_pnl']:>14,.2f}     Avg Loss PnL    : ₹{m['avg_loss_pnl']:>14,.2f}  ║",
            "╠══════════════════════════════════════════════════════════════════════════════════════════╣",
            "║  RISK & DRAWDOWN:                                                                        ║",
            f"║    Max Drawdown (₹): ₹{m['max_drawdown_inr']:>14,.2f}     Max Drawdown %  : {m['max_drawdown_pct']:>13,.2f}%  ║",
            f"║    Max DD Duration : {m['max_drawdown_bars']:>11} bars     Sharpe Ratio    : {m['sharpe_ratio']:>14,.2f}  ║",
            f"║    Sortino Ratio   : {m['sortino_ratio']:>15,.2f}      Calmar Ratio    : {m['calmar_ratio']:>14,.2f}  ║",
            f"║    Max Win Streak  : {m['max_consecutive_wins']:>15}      Max Loss Streak : {m['max_consecutive_losses']:>14}  ║",
            "╠══════════════════════════════════════════════════════════════════════════════════════════╣",
            "║  DURATION & EXCURSION DYNAMICS:                                                          ║",
            f"║    Avg Bars Held   : {m['avg_bars_held']:>11} bars     Avg Time (Mins) : {m['avg_duration_mins']:>10.1f} min ║",
            f"║    Avg Winner Bars : {m['avg_win_bars']:>11} bars     Avg Loser Bars  : {m['avg_loss_bars']:>11} bars ║",
            f"║    Avg Run-Up (MFE): {m['avg_mfe_pct']:>14,.2f}%     Avg Drawdown(MAE: {m['avg_mae_pct']:>13,.2f}%  ║",
            "╠══════════════════════════════════════════════════════════════════════════════════════════╣",
            "║  EXIT REASONS BREAKDOWN:                                                                 ║",
        ]
        for reason, count in m.get("exit_reasons", {}).items():
            pct = (count / m["total_trades"]) * 100.0 if m["total_trades"] > 0 else 0
            lines.append(f"║    • {reason:<25}: {count:>5} ({pct:>5.1f}%)                                  ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    @staticmethod
    def generate_markdown_report(result: Any, title: str = "Backtest Performance Report") -> str:
        m = result.metrics
        c = result.config
        status_emoji = "🟢" if m["net_profit"] >= 0 else "🔴"

        md = f"""# {status_emoji} {title}

**Symbol:** `{c.symbol}` | **Timeframe:** `{c.timeframe}` | **Bars Analyzed:** `{result.candles_count:,}`
**Date Window:** `{c.from_date or 'Earliest'} → {c.to_date or 'Latest'}`

---

## 🎯 Executive KPI Summary

| Key Metric | Value | Risk-Adjusted Metric | Value |
| :--- | :--- | :--- | :--- |
| **Initial Capital** | ₹{m['initial_capital']:,.2f} | **Profit Factor** | **{m['profit_factor']:.2f}** |
| **Final Equity** | **₹{m['final_equity']:,.2f}** | **Win Rate %** | **{m['win_rate_pct']:.2f}%** |
| **Net Profit (₹)** | **₹{m['net_profit']:,.2f}** | **Win / Loss Ratio** | {m['winning_trades']}W / {m['losing_trades']}L |
| **Total Return %** | **{m['total_return_pct']:.2f}%** | **Payoff Ratio (Avg W/L)** | {m['payoff_ratio']:.2f} |
| **Max Drawdown %** | **{m['max_drawdown_pct']:.2f}%** | **Sharpe Ratio** | {m['sharpe_ratio']:.2f} |
| **Max Drawdown (₹)** | ₹{m['max_drawdown_inr']:,.2f} | **Sortino Ratio** | {m['sortino_ratio']:.2f} |
| **Expectancy / Trade** | ₹{m['expectancy_inr']:,.2f} | **Calmar Ratio** | {m['calmar_ratio']:.2f} |

---

## ⏱️ Trade Duration & Excursion Profile

* **Average Holding Time:** `{m['avg_bars_held']} bars` (`{m['avg_duration_mins']} minutes`)
* **Winning Trades Avg Duration:** `{m['avg_win_bars']} bars`
* **Losing Trades Avg Duration:** `{m['avg_loss_bars']} bars`
* **Maximum Favorable Excursion (MFE):** Avg Run-up before exit = `+{m['avg_mfe_pct']:.2f}%`
* **Maximum Adverse Excursion (MAE):** Avg Drawdown before exit = `-{m['avg_mae_pct']:.2f}%`
* **Streaks:** Max Consecutive Wins = `{m['max_consecutive_wins']}` | Max Consecutive Losses = `{m['max_consecutive_losses']}`

---

## 🚪 Exit Triggers Distribution

| Exit Trigger | Trade Count | Percentage % |
| :--- | :---: | :---: |
"""
        for reason, count in m.get("exit_reasons", {}).items():
            pct = (count / m["total_trades"]) * 100.0 if m["total_trades"] > 0 else 0
            md += f"| **{reason}** | {count} | {pct:.1f}% |\n"

        # Trade Table sample
        if not result.trades_df.empty:
            sample_df = result.trades_df.head(15)
            md += "\n---\n\n## 📋 Recent Trades Ledger (First 15 Fills)\n\n"
            md += "| # | Side | Entry Time | Entry ₹ | Exit Time | Exit ₹ | Reason | Bars | Net PnL (₹) | Return % | R:R |\n"
            md += "| :-: | :-: | :--- | :---: | :--- | :---: | :--- | :-: | :---: | :---: | :---: |\n"
            for _, r in sample_df.iterrows():
                pnl_color = "**" if r["net_pnl"] > 0 else ""
                md += (
                    f"| {r['trade_id']} | {r['side']} | {str(r['entry_time'])[:16]} | ₹{r['entry_price']:,.1f} | "
                    f"{str(r['exit_time'])[:16]} | ₹{r['exit_price']:,.1f} | {r['exit_reason']} | {r['bars_held']} | "
                    f"{pnl_color}₹{r['net_pnl']:,.2f}{pnl_color} | {r['return_pct']:+.2f}% | {r['r_multiple']:+.2f}R |\n"
                )

        return md

    @staticmethod
    def export_csv(result: Any, output_dir: Path) -> Tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sym_clean = result.config.symbol.replace(":", "_").replace("-", "_")
        trades_path = output_dir / f"backtest_trades_{sym_clean}_{result.config.timeframe}.csv"
        equity_path = output_dir / f"backtest_equity_{sym_clean}_{result.config.timeframe}.csv"

        result.trades_df.to_csv(trades_path, index=False)
        result.equity_curve.to_csv(equity_path, index=False)
        return trades_path, equity_path

    @staticmethod
    def export_json(result: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "symbol": result.config.symbol,
            "timeframe": result.config.timeframe,
            "candles_count": result.candles_count,
            "metrics": result.metrics,
            "trades_count": len(result.trades),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return output_path
