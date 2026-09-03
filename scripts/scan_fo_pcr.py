#!/usr/bin/env python3
"""
F&O Stock Put-Call Ratio (PCR) & Overbought / Oversold Scanner.

Scans the entire F&O Universe to calculate:
- Live Put-Call Ratio (OI based & Volume based)
- Identifies Extreme Overbought (PCR >= 1.30) and Extreme Oversold (PCR <= 0.55) stocks
- Max Pain Strikes & Highest OI Support/Resistance Walls
- Contrarian trade opportunities (Short squeeze vs Bearish reversal)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tabulate import tabulate

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from trade_system.domains.analysis.application.analysis.fo_pcr_screener import (
    FOPCRScreener,
    StockPCRInfo,
)
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.trading.infrastructure.brokers.legacy import (
    FyersBrokerClient,
    FyersAuthService,
)
from trade_system.shared.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("pcr_scanner")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan F&O Universe for Put-Call Ratio (PCR) & Overbought/Oversold Stocks."
    )
    parser.add_argument(
        "--overbought",
        type=float,
        default=0.85,
        help="PCR threshold for Overbought stocks (default: 0.85).",
    )
    parser.add_argument(
        "--oversold",
        type=float,
        default=0.55,
        help="PCR threshold for Oversold stocks (default: 0.55).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max results to display per table (default: 25).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/fo_pcr_overbought_oversold_report.md",
        help="Output markdown report file path.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of stocks to scan (e.g. 50 for quick run).",
    )
    return parser.parse_args()


def format_pcr_table(stocks: List[StockPCRInfo], limit: int = 25) -> str:
    """Format a list of StockPCRInfo into a clean markdown table."""
    if not stocks:
        return "No stocks found matching criteria."

    rows = []
    for s in stocks[:limit]:
        pcr_str = f"{s.pcr_oi:.2f}"
        pcr_vol_str = f"{s.pcr_volume:.2f}"
        state_icon = (
            "🔥 EXTREME OVERBOUGHT" if s.sentiment_state == "EXTREME_OVERBOUGHT" else
            ("🔴 OVERBOUGHT" if s.sentiment_state == "OVERBOUGHT" else
             ("🟢 OVERSOLD" if s.sentiment_state == "OVERSOLD" else
              ("💎 EXTREME OVERSOLD" if s.sentiment_state == "EXTREME_OVERSOLD" else "⚪ NEUTRAL")))
        )
        ce_oi_str = f"{s.total_call_oi:,.0f}"
        pe_oi_str = f"{s.total_put_oi:,.0f}"

        rows.append([
            s.clean_symbol,
            s.sector,
            f"₹{s.spot_price:,.2f}",
            pcr_str,
            pcr_vol_str,
            state_icon,
            ce_oi_str,
            pe_oi_str,
            f"₹{s.max_pain_strike:,.1f}",
            f"₹{s.highest_ce_oi_strike:,.1f}",
            f"₹{s.highest_pe_oi_strike:,.1f}",
            s.contrarian_bias.replace("_", " "),
        ])

    headers = [
        "Symbol",
        "Sector",
        "Spot LTP",
        "PCR (OI)",
        "PCR (Vol)",
        "Regime / State",
        "Call OI",
        "Put OI",
        "Max Pain",
        "CE Wall",
        "PE Wall",
        "Contrarian Bias",
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def generate_markdown_report(
    scan_results: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    overbought = scan_results.get("overbought", [])
    oversold = scan_results.get("oversold", [])
    all_stocks = scan_results.get("all", [])

    report = [
        "# 📊 F&O Stocks Put-Call Ratio (PCR) & Overbought / Oversold Report",
        f"**Generated:** {now_str} | **Universe:** F&O Stocks ({len(all_stocks)} scanned)",
        f"**Thresholds:** Overbought: $\ge {args.overbought:.2f}$ | Oversold: $\le {args.oversold:.2f}$",
        "",
        "---",
        "",
        f"## 🔴 1. Overbought F&O Stocks (High PCR $\ge {args.overbought:.2f}$ — Heavy Put Writing / Bullish Congestion)",
        "> **Institutional Interpretation:** Excessive Put writing indicates participants are heavily bullish. If price fails to make new highs or breaks support, it creates high risk of long liquidation / profit booking.",
        "",
        format_pcr_table(overbought, limit=args.limit),
        "",
        "---",
        "",
        f"## 🟢 2. Oversold F&O Stocks (Low PCR $\le {args.oversold:.2f}$ — Heavy Call Writing / Bearish Consensus)",
        "> **Institutional Interpretation:** Excessive Call writing indicates strong resistance walls. However, if the stock finds a base and begins to reverse upwards, it creates prime conditions for an explosive **Short Squeeze** as Call writers scramble to cover.",
        "",
        format_pcr_table(oversold, limit=args.limit),
        "",
        "---",
        "",
        "## 🔬 Top Contrarian Opportunities Forensics",
    ]

    # Add forensics for top 3 overbought & oversold
    if overbought:
        report.append("### ⚠️ Top Overbought Stocks (Reversal / Correction Risk)")
        for s in overbought[:3]:
            report.append(f"#### **{s.clean_symbol}** ({s.sector}) — `PCR: {s.pcr_oi:.2f}`")
            report.append(f"- **Spot Price:** ₹{s.spot_price:.2f} | **Max Pain:** ₹{s.max_pain_strike:.2f}")
            report.append(f"- **OI Distribution:** Put OI = `{s.total_put_oi:,}` vs Call OI = `{s.total_call_oi:,}`")
            report.append(f"- **Key Walls:** Resistance CE Wall = `₹{s.highest_ce_oi_strike:.1f}` | Support PE Wall = `₹{s.highest_pe_oi_strike:.1f}`")
            report.append(f"- **Forensics:** _{s.analysis_narrative}_")
            report.append("")

    if oversold:
        report.append("### 🚀 Top Oversold Stocks (Short Squeeze Candidates)")
        for s in oversold[:3]:
            report.append(f"#### **{s.clean_symbol}** ({s.sector}) — `PCR: {s.pcr_oi:.2f}`")
            report.append(f"- **Spot Price:** ₹{s.spot_price:.2f} | **Max Pain:** ₹{s.max_pain_strike:.2f}")
            report.append(f"- **OI Distribution:** Call OI = `{s.total_call_oi:,}` vs Put OI = `{s.total_put_oi:,}`")
            report.append(f"- **Key Walls:** Resistance CE Wall = `₹{s.highest_ce_oi_strike:.1f}` | Support PE Wall = `₹{s.highest_pe_oi_strike:.1f}`")
            report.append(f"- **Forensics:** _{s.analysis_narrative}_")
            report.append("")

    return "\n".join(report)


def main() -> None:
    args = parse_arguments()

    LOGGER.info("Authenticating with Fyers...")
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator,
    )
    broker.authenticate()

    screener = FOPCRScreener(
        broker=broker,
        overbought_threshold=args.overbought,
        oversold_threshold=args.oversold,
    )

    fo_symbols = get_fo_universe()
    if args.sample:
        fo_symbols = fo_symbols[:args.sample]

    scan_results = screener.scan_universe_pcr(symbols=fo_symbols)
    overbought = scan_results["overbought"]
    oversold = scan_results["oversold"]

    print("\n" + "=" * 90)
    print("📊 F&O STOCKS PUT-CALL RATIO (PCR) & OVERBOUGHT / OVERSOLD SCANNER")
    print(f"🔍 Scanned: {len(scan_results['all'])} F&O Stocks | Overbought: {len(overbought)} | Oversold: {len(oversold)}")
    print("=" * 90)

    if overbought:
        print(f"\n🔴 TOP OVERBOUGHT F&O STOCKS (PCR >= {args.overbought:.2f}):")
        print(format_pcr_table(overbought, limit=args.limit))

    if oversold:
        print(f"\n🟢 TOP OVERSOLD F&O STOCKS (PCR <= {args.oversold:.2f} — Short Squeeze Candidates):")
        print(format_pcr_table(oversold, limit=args.limit))

    print("\n" + "=" * 90)

    # Save report
    report_content = generate_markdown_report(scan_results, args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_content)
    LOGGER.info("Report exported to: %s", out_path)


if __name__ == "__main__":
    main()
