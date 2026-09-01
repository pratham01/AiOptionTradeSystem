#!/usr/bin/env python3
"""
Backtest Wyckoff & VSA Institutional Strategy over Multi-Year Historical Data.
=============================================================================
Usage:
    python scripts/backtest_wyckoff.py
    python scripts/backtest_wyckoff.py --max-symbols 100 --holding-bars 10
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

from trade_system.domains.analysis.application.backtesting.wyckoff_backtest import WyckoffBacktestEngine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

LOGGER = logging.getLogger("wyckoff_backtest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest Wyckoff Institutional Strategy")
    parser.add_argument("--max-symbols", type=int, default=80, help="Number of liquid F&O symbols to backtest")
    parser.add_argument("--holding-bars", type=int, default=12, help="Max trade holding bars")
    parser.add_argument("--output-dir", type=str, default="reports/wyckoff", help="Directory to save backtest report")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    LOGGER.info("=" * 70)
    LOGGER.info("🏛️ INSTITUTIONAL WYCKOFF & VSA STRATEGY BACKTEST")
    LOGGER.info("=" * 70)

    engine = WyckoffBacktestEngine(max_holding_bars=args.holding_bars)
    symbols = (["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"] + get_fo_universe())[:args.max_symbols]

    summary = engine.evaluate_universe(symbols=symbols)

    print(summary.summary_markdown())

    # Save Markdown report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "wyckoff_backtest_report.md"
    report_file.write_text(summary.summary_markdown())
    LOGGER.info("Detailed report written to %s", report_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
