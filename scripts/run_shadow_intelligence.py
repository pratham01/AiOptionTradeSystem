"""
Shadow Intelligence Runner — Runs the StockIntelligenceEnricher in shadow mode.

Fetches the current top gainers/losers from the F&O universe via live quotes,
runs the 8-layer intelligence enricher, and writes results to JSON files
for validation. Does NOT modify any existing dashboard or trading flow.

Usage:
  python scripts/run_shadow_intelligence.py          # Run once
  python scripts/run_shadow_intelligence.py --loop   # Run every 15 minutes during market hours
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is on path
root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.analysis.application.analysis.stock_intelligence_enricher import (
    StockIntelligenceEnricher,
)

IST = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger("shadow_intelligence")

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)


def _is_market_hours() -> bool:
    """Check if current time is within market hours (IST)."""
    now = datetime.now(IST).time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def _fetch_top_movers(settings: Settings, top_n: int = 20) -> tuple[list[str], list[str]]:
    """
    Fetch the top N gainers and losers from the F&O universe using live quotes.
    Returns: (gainer_symbols, loser_symbols)
    """
    auth_service = FyersAuthService(settings)
    token = auth_service.get_valid_token()
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth_service.authenticator,
    )

    fo_universe = get_fo_universe()
    all_symbols = [s for s in fo_universe if "INDEX" not in s]

    # Fetch quotes in batches
    all_quotes = {}
    batch_size = 50
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i + batch_size]
        try:
            quotes = broker.get_quotes(batch)
            if quotes:
                all_quotes.update(quotes)
        except Exception as e:
            LOGGER.warning(f"Quote fetch failed for batch {i}: {e}")

    # Calculate pChange and sort
    movers = []
    for sym, q in all_quotes.items():
        if isinstance(q, dict):
            ltp = q.get("lp", 0) or q.get("last_price", 0)
            prev = q.get("prev_close_price", 0) or q.get("pc", 0)
        else:
            ltp = getattr(q, "ltp", 0) or getattr(q, "last_price", 0)
            prev = getattr(q, "prev_close", 0)

        if prev and prev > 0:
            pchange = ((ltp - prev) / prev) * 100
            movers.append((sym, pchange))

    movers.sort(key=lambda x: x[1], reverse=True)
    gainers = [sym for sym, _ in movers[:top_n]]
    losers = [sym for sym, _ in movers[-top_n:]]

    return gainers, losers


def run_once(settings: Settings) -> None:
    """Run a single shadow intelligence cycle."""
    now = datetime.now(IST)
    LOGGER.info(f"=== Shadow Intelligence Run @ {now.strftime('%H:%M:%S')} ===")

    # 1. Fetch top movers
    LOGGER.info("Fetching top 20 gainers and losers from F&O universe...")
    try:
        gainers, losers = _fetch_top_movers(settings, top_n=20)
    except Exception as e:
        LOGGER.error(f"Failed to fetch top movers: {e}")
        return

    all_symbols = list(set(gainers + losers))
    LOGGER.info(f"Processing {len(all_symbols)} unique symbols ({len(gainers)} gainers + {len(losers)} losers)")

    # 2. Enrich
    enricher = StockIntelligenceEnricher()
    results = enricher.enrich_symbols(all_symbols)

    # 3. Write to JSON
    output_dir = Path(settings.data_dir) / "shadow_intelligence" / now.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"intel_{now.strftime('%H%M')}.json"

    output_data = {
        "timestamp": now.isoformat(),
        "gainers": gainers,
        "losers": losers,
        "results": [r.to_dict() for r in results],
        "summary": {
            "total_symbols": len(results),
            "strong_momentum": sum(1 for r in results if "STRONG" in r.composite_verdict),
            "building": sum(1 for r in results if "BUILDING" in r.composite_verdict),
            "reversal_warning": sum(1 for r in results if "REVERSAL" in r.composite_verdict),
            "no_edge": sum(1 for r in results if "NO EDGE" in r.composite_verdict),
        }
    }

    output_file.write_text(json.dumps(output_data, indent=2, default=str))
    LOGGER.info(f"Results written to {output_file}")

    # 4. Print summary
    print(f"\n{'='*60}")
    print(f"Shadow Intelligence Summary @ {now.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Total symbols analyzed: {len(results)}")
    print(f"  🔥 Strong Momentum:    {output_data['summary']['strong_momentum']}")
    print(f"  ⚡ Building:           {output_data['summary']['building']}")
    print(f"  ⚠️  Reversal Warning:   {output_data['summary']['reversal_warning']}")
    print(f"  💤 No Edge:            {output_data['summary']['no_edge']}")
    print()

    # Print top signals
    strong = [r for r in results if "STRONG" in r.composite_verdict or "REVERSAL" in r.composite_verdict]
    if strong:
        print("Key Signals:")
        for r in strong:
            direction_emoji = "🟢" if r.composite_direction == "BULL" else ("🔴" if r.composite_direction == "BEAR" else "⚪")
            clean = r.symbol.replace("NSE:", "").replace("-EQ", "")
            print(f"  {direction_emoji} {clean:15s} | {r.composite_verdict:25s} | Score: {r.composite_score:.0f} | Bull:{r.bull_count} Bear:{r.bear_count}")
            for layer in r.layers:
                if layer.direction != "NEUTRAL" or layer.strength > 0.3:
                    print(f"      → {layer.label}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Shadow Intelligence Runner")
    parser.add_argument("--loop", action="store_true", help="Run every 15 minutes during market hours")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    settings = Settings.load()

    if args.loop:
        LOGGER.info("Starting shadow intelligence loop (every 15 minutes during market hours)...")
        while True:
            if _is_market_hours():
                try:
                    run_once(settings)
                except Exception as e:
                    LOGGER.error(f"Shadow intelligence run failed: {e}", exc_info=True)
                LOGGER.info("Sleeping for 15 minutes...")
                time.sleep(900)  # 15 minutes
            else:
                LOGGER.info("Outside market hours. Sleeping for 5 minutes...")
                time.sleep(300)
    else:
        run_once(settings)


if __name__ == "__main__":
    main()
