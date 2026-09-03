#!/usr/bin/env python3
"""
Multi-Touch Support/Resistance Breakout & Breakdown Proximity Scanner.

Scans the entire F&O Universe (~200 stocks) on the Daily timeframe to find stocks
exhibiting the exact multi-touch compression pattern observed in stocks like KEI:
- Repeated tests (>= 3 touches) against a flat horizontal support or resistance.
- Descending highs into support (Breakdown) or Ascending lows into resistance (Breakout).
- Price currently in tight proximity (0.0% to 2.5%) or right at the trigger point.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from trade_system.domains.analysis.application.analysis.breakout_breakdown_proximity_screener import (
    BreakoutBreakdownProximityScreener,
    ProximitySetup,
)
from trade_system.domains.market_data.infrastructure.data.fo_universe import (
    get_fo_universe,
    get_sector_mapping,
)
from trade_system.domains.trading.infrastructure.brokers.legacy import (
    FyersBrokerClient,
    FyersAuthService,
)
from trade_system.shared.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("proximity_scanner")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan F&O Universe for Multi-Touch Breakout/Breakdown Proximity Setups."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Daily lookback period in sessions (default: 30).",
    )
    parser.add_argument(
        "--min-touches",
        type=int,
        default=3,
        help="Minimum times the level must have been tested (default: 3).",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=2.5,
        help="Max %% distance from current close to level (default: 2.5%%).",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["all", "breakdown", "breakout"],
        default="all",
        help="Filter setups by type: 'all', 'breakdown', or 'breakout'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/breakout_breakdown_proximity_report.md",
        help="Output markdown report file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max results to display per table (default: 20).",
    )
    return parser.parse_args()


def fetch_universe_daily_data(
    broker: FyersBrokerClient,
    symbols: List[str],
    days: int = 60,
) -> Dict[str, pd.DataFrame]:
    """Fetch daily OHLCV bars for all requested symbols."""
    today = date.today()
    start_date = today - timedelta(days=days + 30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    results: Dict[str, pd.DataFrame] = {}
    total = len(symbols)

    LOGGER.info("Fetching daily OHLCV data for %d F&O symbols (%s to %s)...", total, start_str, end_str)

    for i, sym in enumerate(symbols, 1):
        try:
            df = broker.get_historical_data(
                symbol=sym,
                resolution="D",
                range_from=start_str,
                range_to=end_str,
            )
            if df is not None and not df.empty and len(df) >= 20:
                results[sym] = df
            if i % 25 == 0 or i == total:
                LOGGER.info("Progress: %d/%d symbols fetched...", i, total)
        except Exception as exc:
            LOGGER.debug("Failed to fetch daily data for %s: %s", sym, exc)

    LOGGER.info("Successfully fetched daily data for %d/%d symbols.", len(results), total)
    return results


def format_table(setups: List[ProximitySetup], limit: int = 20) -> str:
    """Format a list of ProximitySetup into a clean markdown/text table."""
    if not setups:
        return "No setups detected matching criteria."

    rows = []
    for s in setups[:limit]:
        status_icon = (
            "🚨 AT TRIGGER" if "AT_" in s.setup_type else
            ("📉 NEAR LEVEL" if s.direction == "BEARISH" else "📈 NEAR LEVEL")
            if "NEAR_" in s.setup_type else "⚡ BROKEN"
        )
        dist_str = f"{s.distance_pct:+.2f}%"
        rows.append([
            s.clean_symbol,
            s.sector,
            f"₹{s.current_price:,.2f}",
            f"₹{s.key_level:,.2f}",
            dist_str,
            f"{s.touch_count} tests",
            f"{s.compression_score:.0f}/100",
            f"{s.atr_ratio:.2f}x",
            status_icon,
            s.setup_type.replace("_", " "),
        ])

    headers = [
        "Symbol",
        "Sector",
        "LTP",
        "Key Level",
        "Dist %",
        "Touches",
        "Compression",
        "ATR Ratio",
        "Alert State",
        "Setup Category",
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def generate_markdown_report(
    breakdowns: List[ProximitySetup],
    breakouts: List[ProximitySetup],
    args: argparse.Namespace,
) -> str:
    """Generate comprehensive markdown report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    total_found = len(breakdowns) + len(breakouts)

    report = [
        "# 🎯 Multi-Touch Support & Resistance Proximity Report",
        f"**Generated:** {now_str} | **Universe:** F&O Stocks (~200) | **Lookback:** {args.lookback} Daily Bars",
        f"**Criteria:** Minimum {args.min_touches} Touches | Within {args.max_distance:.1f}% of Critical Level",
        "",
        "---",
        "",
        f"## 🚨 1. Top Breakdown Candidates (Support Floor Tests — Pattern Like `KEI`)",
        "> **Anatomy:** Stocks that have repeatedly tested a horizontal support floor ($\ge 3$ times) with descending highs. A break below the floor triggers an explosive breakdown.",
        "",
        format_table(breakdowns, limit=args.limit),
        "",
        "---",
        "",
        f"## 🚀 2. Top Breakout Candidates (Resistance Ceiling Tests)",
        "> **Anatomy:** Stocks that have repeatedly tested a horizontal resistance ceiling ($\ge 3$ times) with ascending higher lows. A break above the ceiling triggers an explosive breakout.",
        "",
        format_table(breakouts, limit=args.limit),
        "",
        "---",
        "",
        "## 🔬 Detailed Case Forensics",
    ]

    # Add top 3 breakdowns and breakouts detailed analysis
    for cat_name, cat_list in [("Breakdown Setups", breakdowns[:3]), ("Breakout Setups", breakouts[:3])]:
        if not cat_list:
            continue
        report.append(f"### {cat_name}")
        for s in cat_list:
            report.append(f"#### **{s.clean_symbol}** ({s.sector}) — `{s.setup_type}`")
            report.append(f"- **Current Price:** ₹{s.current_price:.2f} | **Key Level:** ₹{s.key_level:.2f} (**{s.distance_pct:+.2f}%** away)")
            report.append(f"- **Touch Count:** {s.touch_count} distinct tests across last {s.lookback_days} sessions")
            report.append(f"- **Volatility Squeeze:** 5d/20d ATR Ratio = `{s.atr_ratio:.2f}` | BB Width = `{s.bollinger_bandwidth_pct:.2f}%` | Score = `{s.compression_score:.0f}/100`")
            report.append(f"- **Recent High/Low Range:** ₹{s.recent_range_low:.2f} — ₹{s.recent_range_high:.2f}")
            report.append(f"- **Summary:** _{s.analysis_summary}_")
            report.append("")

    return "\n".join(report)


def main() -> None:
    args = parse_arguments()

    LOGGER.info("Initializing Fyers Broker Client...")
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator,
    )

    fo_symbols = get_fo_universe()
    LOGGER.info("Loaded %d symbols from F&O Universe.", len(fo_symbols))

    # Fetch daily OHLCV
    symbols_data = fetch_universe_daily_data(broker, fo_symbols, days=args.lookback + 20)

    # Initialize Screener
    screener = BreakoutBreakdownProximityScreener(
        lookback_days=args.lookback,
        cluster_tolerance_pct=0.012,
        min_touches=args.min_touches,
        max_proximity_pct=args.max_distance,
    )

    LOGGER.info("Analyzing multi-touch support and resistance levels across all symbols...")
    scan_results = screener.scan_universe(symbols_data)
    breakdowns = scan_results["breakdown_setups"]
    breakouts = scan_results["breakout_setups"]

    # Filter by user preference
    if args.type == "breakdown":
        breakouts = []
    elif args.type == "breakout":
        breakdowns = []

    # Print to console
    print("\n" + "=" * 90)
    print("🎯 MULTI-TOUCH BREAKOUT & BREAKDOWN PROXIMITY SCANNER (DAILY TIMEFRAME)")
    print(f"📅 Lookback: {args.lookback} Bars | 🔍 Min Touches: {args.min_touches} | 📏 Max Proximity: {args.max_distance}%")
    print("=" * 90)

    if breakdowns:
        print("\n🚨 TOP F&O STOCKS NEAR/AT BREAKDOWN FLOOR (Like KEI Pattern):")
        print(format_table(breakdowns, limit=args.limit))

    if breakouts:
        print("\n🚀 TOP F&O STOCKS NEAR/AT BREAKOUT CEILING:")
        print(format_table(breakouts, limit=args.limit))

    print("\n" + "=" * 90)
    print(f"Total Breakdown Setups: {len(breakdowns)} | Total Breakout Setups: {len(breakouts)}")
    print("=" * 90)

    # Export report
    report_content = generate_markdown_report(breakdowns, breakouts, args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_content)
    LOGGER.info("Report exported to: %s", out_path)


if __name__ == "__main__":
    main()
