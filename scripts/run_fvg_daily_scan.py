#!/usr/bin/env python3
"""
Run FVG Daily Scan — Post-Market Fair Value Gap Scanner for F&O Universe
=======================================================================
Usage:
    python scripts/run_fvg_daily_scan.py                      # Scan all F&O stocks
    python scripts/run_fvg_daily_scan.py --direction bullish  # Bullish BISI retests only
    python scripts/run_fvg_daily_scan.py --direction bearish  # Bearish SIBI retests only
    python scripts/run_fvg_daily_scan.py --top-n 15 --telegram # Post top setups to Telegram
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.advisory.application.agent.fvg_agent import FvgDailyScannerAgent

LOGGER = logging.getLogger("fvg_daily_scan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fair Value Gap (FVG) Daily Scanner for F&O Stocks")
    parser.add_argument("--direction", choices=["bullish", "bearish", "both"], default="both", help="Setup direction")
    parser.add_argument("--min-score", type=float, default=65.0, help="Minimum confluence score (0-100)")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top setups to display")
    parser.add_argument("--telegram", action="store_true", help="Send top setups to Telegram channel")
    parser.add_argument("--output-dir", type=str, default="reports/fvg", help="Directory to save report markdown")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    settings = Settings.load()
    agent = FvgDailyScannerAgent(settings=settings, min_score=args.min_score)

    LOGGER.info("=" * 70)
    LOGGER.info("⚡ FAIR VALUE GAP (FVG) DAILY F&O SCANNER")
    LOGGER.info("=" * 70)

    setups = agent.scan_universe(direction=args.direction, min_score=args.min_score)

    if not setups:
        print("\nNo active FVG setups found meeting the score threshold.")
        return 0

    print(f"\n🎯 FOUND {len(setups)} ACTIVE DAILY FVG SETUPS (Showing Top {args.top_n}):\n")
    print(f"{'#':<3} {'Symbol':<18} {'Action':<10} {'Type':<20} {'Score':<6} {'Spot/Entry':<12} {'Stop Loss':<12} {'Target 1':<12} {'50% CE':<12} {'RRR':<6}")
    print("-" * 115)

    for idx, s in enumerate(setups[:args.top_n], 1):
        sym = s.symbol.replace("NSE:", "").replace("-EQ", "")
        print(f"{idx:<3} {sym:<18} {s.action:<10} {s.setup_type:<20} {s.confluence_score:<6.0f} ₹{s.entry_price:<11.2f} ₹{s.stop_loss:<11.2f} ₹{s.target_1:<11.2f} ₹{s.consequent_encroachment:<11.2f} 1:{s.risk_reward_ratio:<4.1f}")

    print("\n" + "=" * 115)
    print("Top Setup Highlights:")
    for idx, s in enumerate(setups[:5], 1):
        print(f"\n{idx}. {s.format_summary()}")

    # Save Markdown report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = agent.generate_report_markdown(setups)
    report_file = out_dir / f"fvg_daily_scan.md"
    report_file.write_text(report_md)
    LOGGER.info("Report saved to %s", report_file)

    # Optional Telegram notification
    if args.telegram:
        LOGGER.info("Sending Telegram FVG notification...")
        sent = agent.send_telegram_report(setups, top_n=args.top_n)
        if sent:
            LOGGER.info("✓ Telegram FVG alert delivered successfully.")
        else:
            LOGGER.warning("✗ Failed to send Telegram FVG alert (check telegram credentials or sector conflicts).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
