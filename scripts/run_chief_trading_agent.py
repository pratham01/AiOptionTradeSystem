#!/usr/bin/env python3
"""
Chief Trading Agent Runner — Evaluates multi-pillar confluence and daily governor.

Usage:
    # Evaluate NIFTY 50
    python3 scripts/run_chief_trading_agent.py --symbol NSE:NIFTY50-INDEX

    # Evaluate BANK NIFTY and dispatch Telegram alert if qualified
    python3 scripts/run_chief_trading_agent.py --symbol NSE:NIFTYBANK-INDEX --dispatch-telegram

    # Scan all major indices
    python3 scripts/run_chief_trading_agent.py --scan-indices

    # Scan selected F&O stock
    python3 scripts/run_chief_trading_agent.py --symbol NSE:HDFCBANK-EQ
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from trade_system.domains.advisory.application.agent.chief_trading_agent import (
    ChiefTradingAgent,
    ChiefTradeSignal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("run_chief_agent")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chief Trading Agent to evaluate institutional multi-pillar confluence."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="NSE:NIFTY50-INDEX",
        help="Symbol to evaluate (default: NSE:NIFTY50-INDEX).",
    )
    parser.add_argument(
        "--scan-indices",
        action="store_true",
        help="Scan all major indices (NIFTY50, BANKNIFTY, SENSEX).",
    )
    parser.add_argument(
        "--strikecount",
        type=int,
        default=15,
        help="Number of strikes to evaluate (default: 15).",
    )
    parser.add_argument(
        "--dispatch-telegram",
        action="store_true",
        help="Dispatch high-priority Telegram alert if signal qualifies.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass daily trade budget checks for diagnostics.",
    )
    return parser.parse_args()


def print_scorecard(scorecard: dict, signal: ChiefTradeSignal | None, budget_info: tuple) -> None:
    """Print beautifully formatted Chief Agent scorecard."""
    can_trade, count, max_trades, budget_msg = budget_info
    symbol = scorecard.get("symbol", "N/A")
    score = scorecard.get("total_confluence_score", 0)
    direction = scorecard.get("direction", "NEUTRAL")
    spot = scorecard.get("spot_price", 0.0)
    pillars = scorecard.get("pillar_scores", {})
    vetoes = scorecard.get("veto_reasons", [])

    print("\n" + "=" * 80)
    print(f"🎯 CHIEF TRADING AGENT: INSTITUTIONAL CONFLUENCE SCORECARD")
    print(f"📍 Underlying: {symbol} | Spot LTP: ₹{spot:,.2f}")
    print(f"🛡️ Daily Budget: {count} / {max_trades} Trades Taken Today | Status: {budget_msg}")
    print("=" * 80)

    print("\n🏛️ 1. FIVE-PILLAR CONFLUENCE BREAKDOWN:")
    p1 = pillars.get("Price Action & Momentum", 0)
    p2 = pillars.get("Smart Money Flow", 0)
    p3 = pillars.get("Order Flow Aggression", 0)
    p4 = pillars.get("Structural Walls", 0)
    p5 = pillars.get("Divergence & Traps", 0)

    print(f" • Pillar 1: Price Action & Momentum:     {p1:2d} / 25 pts  {'🟢' if p1 >= 18 else ('🟡' if p1 >= 10 else '🔴')}")
    print(f" • Pillar 2: Option Chain Smart Money:     {p2:2d} / 25 pts  {'🟢' if p2 >= 18 else ('🟡' if p2 >= 10 else '🔴')}")
    print(f" • Pillar 3: Order Flow & ATM Delta:       {p3:2d} / 20 pts  {'🟢' if p3 >= 14 else ('🟡' if p3 >= 8 else '🔴')}")
    print(f" • Pillar 4: Structural Liquidity Walls:   {p4:2d} / 15 pts  {'🟢' if p4 >= 10 else ('🟡' if p4 >= 5 else '🔴')}")
    print(f" • Pillar 5: Divergence & Trap Filter:     {p5:2d} / 15 pts  {'🟢' if p5 >= 10 else ('🟡' if p5 >= 5 else '🔴')}")
    print(f" --------------------------------------------------------")
    status_icon = "🔥 QUALIFIED" if score >= 80 and not vetoes else "⏳ STAND ASIDE"
    print(f" 🏆 TOTAL CONFLUENCE SCORE:               {score:2d} / 100 pts [{status_icon}]")
    print(f" 🧭 PRIMARY DIRECTIONAL BIAS:             {direction}")

    if vetoes:
        print("\n🚨 2. ACTIVE VETO & GATEKEEPING FILTERS:")
        for v in vetoes:
            print(f" • {v}")
    else:
        print("\n🚨 2. ACTIVE VETO FILTERS: None (All Gatekeeping Rules Cleared ✅)")

    if signal:
        print("\n" + "-" * 80)
        print(f"🎯 3. QUALIFIED SNIPER TRADE SIGNAL: [READY TO EXECUTE]")
        print("-" * 80)
        print(f" • Contract / Strike:    {signal.strike_name}")
        print(f" • Action:               {signal.direction}")
        print(f" • Spot Entry Trigger:   ₹{signal.spot_entry_trigger:,.2f}")
        print(f" • Spot Stop Loss:       ₹{signal.spot_stop_loss:,.2f}")
        print(f" • Target 1 / Target 2:  ₹{signal.target_1:,.2f} / ₹{signal.target_2:,.2f}")
        print(f" • Risk:Reward Ratio:    {signal.risk_reward}")
        print(f" • Master Thesis:        {signal.primary_thesis}")
        print("=" * 80 + "\n")
    else:
        print("\n" + "-" * 80)
        print("🛑 3. TRADE STATUS: NO SNIPER TRADE RECOMMENDED AT THIS TIME")
        print("     Agent maintains strict institutional discipline. Only enters on >= 80% confluence.")
        print("-" * 80 + "\n")


def main() -> None:
    args = parse_arguments()
    agent = ChiefTradingAgent()

    symbols_to_scan = []
    if args.scan_indices:
        symbols_to_scan = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
    else:
        symbols_to_scan = [args.symbol]

    for sym in symbols_to_scan:
        LOGGER.info("Chief Agent evaluating %s...", sym)
        asset_type = "INDEX" if "INDEX" in sym else "STOCK"
        budget_info = agent.check_daily_trade_budget(asset_type)

        signal, scorecard = agent.evaluate_symbol(
            sym,
            strikecount=args.strikecount,
            force_evaluation=args.force,
        )

        print_scorecard(scorecard, signal, budget_info)

        if signal and args.dispatch_telegram:
            LOGGER.info("Dispatching Telegram sniper alert for %s...", sym)
            success = agent.dispatch_telegram_alert(signal)
            print(f"📢 Telegram Alert Dispatched: {'SUCCESS ✅' if success else 'FAILED ❌'}")


if __name__ == "__main__":
    main()
