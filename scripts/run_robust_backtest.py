#!/usr/bin/env python
"""
Institutional Multi-Asset Portfolio Backtest CLI Runner.

Usage:
    python scripts/run_robust_backtest.py --capital 1000000 --risk 1.0 --min-prob 65
"""
import argparse
import logging
from datetime import date
from pathlib import Path
import pandas as pd

from trade_system.domains.analysis.application.backtesting.portfolio_backtester import (
    BacktestConfig,
    PortfolioBacktestEngine
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run Institutional Multi-Asset Portfolio Backtest")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Initial Capital (INR)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per Trade % (e.g. 1.0)")
    parser.add_argument("--max-positions", type=int, default=8, help="Max concurrent active positions")
    parser.add_argument("--max-sector", type=int, default=2, help="Max concurrent positions per sector")
    parser.add_argument("--min-prob", type=int, default=65, help="Minimum probability conviction (0-100)")
    parser.add_argument("--max-days", type=int, default=5, help="Max holding days per trade")
    parser.add_argument("--no-trailing", action="store_true", help="Disable trailing SL to breakeven")
    parser.add_argument("--output-dir", type=str, default="reports/backtest", help="Report output directory")
    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk,
        max_concurrent_positions=args.max_positions,
        max_positions_per_sector=args.max_sector,
        max_holding_days=args.max_days,
        min_probability=args.min_prob,
        trailing_stop=not args.no_trailing
    )

    print("\n" + "=" * 70)
    print("🚀 INSTITUTIONAL PORTFOLIO BACKTEST ENGINE (F&O UNIVERSE)")
    print(f"💰 Capital: ₹{config.initial_capital:,.2f} | 🛡️ Risk/Trade: {config.risk_per_trade_pct}% | 📊 Min Prob: {config.min_probability}%")
    print(f"🔒 Max Positions: {config.max_concurrent_positions} | 🧭 Max/Sector: {config.max_positions_per_sector} | ⏳ Max Hold: {config.max_holding_days}d")
    print("=" * 70 + "\n")

    engine = PortfolioBacktestEngine(config=config)
    report = engine.run_reversal_backtest()

    print("\n" + "-" * 70)
    print("📈 EXECUTIVE SUMMARY & CAPITAL GROWTH")
    print("-" * 70)
    print(f"Initial Capital:         ₹{report.initial_capital:,.2f}")
    print(f"Final Portfolio Equity:  ₹{report.final_equity:,.2f}")
    print(f"Net Realized PnL:        ₹{report.net_profit:+,.2f} ({report.total_return_pct:+.2f}%)")
    print(f"Annualized CAGR:         {report.cagr_pct:+.2f}%")
    print(f"Profit Factor:           {report.profit_factor:.2f}")
    print(f"Sharpe Ratio:            {report.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:           {report.sortino_ratio:.2f}")
    print(f"Calmar Ratio:            {report.calmar_ratio:.2f}")
    print(f"Max Portfolio Drawdown:  {report.max_drawdown_pct:.2f}% (₹{report.max_drawdown_inr:,.2f})")
    print(f"Max Drawdown Duration:   {report.max_drawdown_duration_days} trading days")

    print("\n" + "-" * 70)
    print("🎯 TRADE EXECUTION & WIN/LOSS STATS")
    print("-" * 70)
    print(f"Total Signals Scanned:   {report.total_signals}")
    print(f"Confirmed Executed:      {report.triggered_trades} ({report.trigger_rate_pct}% Execution Rate)")
    print(f"Wins / Losses:           {report.winning_trades} Wins / {report.losing_trades} Losses")
    print(f"Win Rate:                {report.win_rate_pct:.1f}% (Loss Rate: {report.loss_rate_pct:.1f}%)")
    print(f"Win / Loss Ratio:        {report.win_loss_ratio:.2f}")
    print(f"Average Win:             ₹{report.avg_win_pnl:+,.2f}")
    print(f"Average Loss:            ₹{report.avg_loss_pnl:+,.2f}")
    print(f"Expectancy:              ₹{report.expectancy_inr:+,.2f} per trade ({report.expectancy_r:+.2f}R)")
    print(f"Avg Holding Duration:    {report.avg_holding_days:.1f} days")
    print(f"Max Win / Loss Streaks:  {report.max_consecutive_wins} wins in a row / {report.max_consecutive_losses} losses in a row")

    if not report.confidence_attribution.empty:
        print("\n" + "-" * 70)
        print("🔥 PERFORMANCE BY CONFIDENCE TIER")
        print("-" * 70)
        print(report.confidence_attribution.to_string(index=False))

    if not report.sector_attribution.empty:
        print("\n" + "-" * 70)
        print("🧭 TOP SECTOR ALPHA ATTRIBUTION")
        print("-" * 70)
        print(report.sector_attribution.head(8).to_string(index=False))

    if not report.confluence_attribution.empty:
        print("\n" + "-" * 70)
        print("🔬 TOP INDICATOR & SMC CONFLUENCE EDGE")
        print("-" * 70)
        print(report.confluence_attribution.head(10).to_string(index=False))

    # Export Markdown and CSV Report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()

    if not report.trades_df.empty:
        csv_path = out_dir / f"backtest_trades_{today_str}.csv"
        report.trades_df.to_csv(csv_path, index=False)
        print(f"\n📁 Trade Log CSV saved to: {csv_path}")

    md_report_path = out_dir / f"backtest_summary_{today_str}.md"
    with open(md_report_path, "w") as f:
        f.write(f"# 📊 Institutional Portfolio Backtest Report ({today_str})\n\n")
        f.write(f"**Capital**: ₹{report.initial_capital:,.2f} | **Final Equity**: ₹{report.final_equity:,.2f} | **Total Return**: {report.total_return_pct:+.2f}%\n\n")
        f.write(f"### 📈 Core Metrics\n")
        f.write(f"- **CAGR**: `{report.cagr_pct:+.2f}%`\n")
        f.write(f"- **Profit Factor**: `{report.profit_factor:.2f}`\n")
        f.write(f"- **Sharpe Ratio**: `{report.sharpe_ratio:.2f}` | **Sortino**: `{report.sortino_ratio:.2f}`\n")
        f.write(f"- **Max Drawdown**: `{report.max_drawdown_pct:.2f}%` (₹{report.max_drawdown_inr:,.2f})\n")
        f.write(f"- **Win Rate**: `{report.win_rate_pct:.1f}%` ({report.winning_trades}W / {report.losing_trades}L)\n")
        f.write(f"- **Expectancy**: `₹{report.expectancy_inr:+,.2f}` per trade (`{report.expectancy_r:+.2f}R`)\n\n")

        if not report.confluence_attribution.empty:
            f.write("### 🔬 Confluence Edge Attribution\n\n")
            f.write(report.confluence_attribution.head(12).to_markdown(index=False))
            f.write("\n\n")

    print(f"📄 Markdown Report generated at: {md_report_path}\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
