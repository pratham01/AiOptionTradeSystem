#!/usr/bin/env python3
"""
Continuous Chief Trading Agent Daemon — Autonomous Market Structure & Confluence Monitor.

Runs continuously during market hours:
- Tracks India VIX dynamics and volatility regimes
- Detects Market Structure: BOS (Break of Structure), Liquidity Sweeps, and Supertrend
- Evaluates Option Chain Wall shifts and dynamic OI velocity
- Fuses all data streams into the 5-Pillar Confluence Engine (0-100)
- Enforces strict daily budget (max 2-3 trades on Indices)
- Automatically dispatches high-conviction sniper trade signals to Telegram!

Usage:
    # Run continuous daemon (every 2 minutes)
    python3 scripts/run_continuous_chief_agent.py

    # Run single cycle check
    python3 scripts/run_continuous_chief_agent.py --run-once
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
    ContinuousChiefAgent,
)

logs_dir = root_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "chief_agent.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("continuous_chief_agent")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Continuous Chief Trading Agent daemon."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Evaluation interval in seconds (default: 120).",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single evaluation cycle and exit.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="NSE:NIFTY50-INDEX,NSE:NIFTYBANK-INDEX",
        help="Comma-separated symbols to monitor (default: NSE:NIFTY50-INDEX,NSE:NIFTYBANK-INDEX).",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable automatic Telegram signal dispatch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    target_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    dispatch = not args.no_telegram

    LOGGER.info("Initializing Continuous Chief Trading Agent...")
    LOGGER.info("Monitoring Symbols: %s", target_symbols)
    LOGGER.info("Telegram Dispatch: %s", "ENABLED" if dispatch else "DISABLED")

    daemon = ContinuousChiefAgent()

    if args.run_once:
        LOGGER.info("Running single evaluation cycle...")
        result = daemon.run_cycle(symbols=target_symbols, dispatch_telegram=dispatch)
        print("\n" + "=" * 70)
        print("🎯 CONTINUOUS CHIEF AGENT: CYCLE COMPLETED")
        print("=" * 70)
        vix = result.get("vix", {})
        print(f"📊 India VIX: {vix.get('vix_level', 'N/A')} ({vix.get('vix_change_pct', 0.0):+.2f}%) | Regime: {vix.get('vix_regime', 'N/A')}")
        for sym, sc in result.get("indices_status", {}).items():
            ms = sc.get("market_structure", {})
            print(f"\n📍 {sym} (Spot: ₹{sc.get('spot_price', 0):,.2f}):")
            print(f" • Confluence Score: {sc.get('total_confluence_score', 0)}/100 [{sc.get('direction', 'NEUTRAL')}]")
            print(f" • Structure State:  {ms.get('structure_state', 'N/A')}")
            print(f" • Wall Migration:   {ms.get('wall_migration', 'STABLE')}")
            print(f" • VIX Divergence:   {ms.get('vix_divergence', 'NONE')}")
        signals = result.get("qualified_signals", [])
        if signals:
            print(f"\n🔥 QUALIFIED SIGNALS TRIGGERED: {len(signals)}")
            for s in signals:
                print(f" • {s.get('symbol')} {s.get('direction')} ({s.get('strike_name')}) Trigger: ₹{s.get('spot_entry_trigger')}")
        else:
            print("\n🛑 NO QUALIFIED TRADES TRIGGERED (Criteria >= 80% confluence not met).")
        print("=" * 70 + "\n")
        return

    LOGGER.info("Entering continuous autonomous monitoring loop (Interval: %ss)...", args.interval)
    daemon.run_forever(interval_seconds=args.interval, dispatch_telegram=dispatch)


if __name__ == "__main__":
    main()
