#!/usr/bin/env python3
"""
Run Wyckoff Daily Scan — Post-Market Wyckoff & VSA Scanner for F&O Universe
===========================================================================
Usage:
    python scripts/run_wyckoff_daily_scan.py                      # Scan all F&O stocks
    python scripts/run_wyckoff_daily_scan.py --direction bullish  # Accumulation springs only
    python scripts/run_wyckoff_daily_scan.py --direction bearish  # Distribution UTADs only
    python scripts/run_wyckoff_daily_scan.py --top-n 15 --telegram # Post top setups to Telegram
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
from trade_system.domains.advisory.application.agent.wyckoff_agent import WyckoffDailyScannerAgent

LOGGER = logging.getLogger("wyckoff_daily_scan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wyckoff & VSA Daily Scanner for F&O Stocks")
    parser.add_argument("--direction", choices=["bullish", "bearish", "both"], default="both", help="Setup direction")
    parser.add_argument("--min-score", type=float, default=60.0, help="Minimum confluence score (0-100)")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top setups to display")
    parser.add_argument("--telegram", action="store_true", help="Send top setups to Telegram channel")
    parser.add_argument("--output-dir", type=str, default="reports/wyckoff", help="Directory to save report markdown")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    settings = Settings.load()
    agent = WyckoffDailyScannerAgent(settings=settings, min_score=args.min_score)

    LOGGER.info("=" * 70)
    LOGGER.info("📈 WYCKOFF METHOD & VSA DAILY F&O SCANNER")
    LOGGER.info("=" * 70)

    setups = agent.scan_universe(direction=args.direction, min_score=args.min_score)

    if not setups:
        print("\nNo active Wyckoff setups found meeting the score threshold.")
        return 0

    print(f"\n🎯 FOUND {len(setups)} ACTIVE WYCKOFF SETUPS (Showing Top {args.top_n}):\n")
    print(f"{'#':<3} {'Symbol':<18} {'Action':<10} {'Pattern':<25} {'Score':<6} {'Spot/Entry':<12} {'Stop Loss':<12} {'Target 1':<12} {'RRR':<6}")
    print("-" * 110)

    for idx, s in enumerate(setups[:args.top_n], 1):
        sym = s.symbol.replace("NSE:", "").replace("-EQ", "")
        print(f"{idx:<3} {sym:<18} {s.action:<10} {s.pattern_type:<25} {s.confluence_score:<6.0f} ₹{s.entry_price:<11.2f} ₹{s.stop_loss:<11.2f} ₹{s.target_1:<11.2f} 1:{s.risk_reward_ratio:<4.1f}")

    print("\n" + "=" * 110)
    print("Top Setup Highlights:")
    for idx, s in enumerate(setups[:5], 1):
        print(f"\n{idx}. {s.format_summary()}")

    # Save Markdown report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = agent.generate_report_markdown(setups)
    report_file = out_dir / f"wyckoff_daily_scan.md"
    report_file.write_text(report_md)
    LOGGER.info("Report saved to %s", report_file)

    # Optional Telegram notification
    if args.telegram:
        LOGGER.info("Sending Telegram Wyckoff notification...")
        sent = agent.send_telegram_report(setups, top_n=args.top_n)
        if sent:
            LOGGER.info("✓ Telegram Wyckoff alert delivered successfully.")
        else:
            LOGGER.warning("✗ Failed to send Telegram Wyckoff alert (check telegram credentials).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
