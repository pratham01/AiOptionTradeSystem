"""
Fetch historical data from Fyers and store directly into timeframe-specific database tables.
Backfills data from 2020-01-01 for the F&O universe.
"""

import sys
import logging
import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session

# Add src to path
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.infrastructure.data.history import HistoricalDataService
from trade_system.infrastructure.data.storage import CsvDataCatalog
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.repository import save_market_data_batch
from trade_system.infrastructure.data.fo_universe import get_fo_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

def _require_token(settings):
    token_file = Path(".secrets/fyers_token.json")
    if not token_file.exists():
        raise FileNotFoundError("Fyers token not found. Please run 'python scripts/authenticate_fyers.py' first.")
    import json
    return json.loads(token_file.read_text())["access_token"]

async def backfill_data(args):
    settings = Settings.load()
    token = _require_token(settings)
    
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
    )
    
    catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
    service = HistoricalDataService(broker, catalog)
    engine = get_engine()
    
    symbols = args.symbols
    if "FO" in symbols:
        symbols.remove("FO")
        symbols.extend(get_fo_universe())
    if "INDEX" in symbols:
        symbols.remove("INDEX")
        symbols.extend(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"])
        
    resolutions = args.resolutions
    from_date = date.fromisoformat(args.from_date)
    to_date = date.today() - timedelta(days=1)
    
    LOGGER.info(f"Starting backfill for {len(symbols)} symbols and {resolutions} resolutions...")
    
    for symbol in symbols:
        for res in resolutions:
            try:
                LOGGER.info(f"Processing {symbol} ({res})...")
                # HistoricalDataService handles chunking internally
                df = service.collect(
                    symbol=symbol,
                    resolution=res,
                    from_date=from_date,
                    to_date=to_date,
                    chunk_days=args.chunk_days,
                    sleep_seconds=args.sleep
                )
                
                if not df.empty:
                    with Session(engine) as session:
                        count = save_market_data_batch(session, symbol, res, df.to_dict('records'))
                        LOGGER.info(f"Successfully saved {count} rows for {symbol} ({res}) to DB.")
                else:
                    LOGGER.warning(f"No data returned for {symbol} ({res}).")
                    
            except Exception as e:
                LOGGER.error(f"Failed to backfill {symbol} ({res}): {e}")
                time.sleep(1) # Grace period after error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical data to SQLite.")
    parser.add_argument("--symbols", nargs="+", default=["INDEX"], help="List of symbols or 'FO'/'INDEX'.")
    parser.add_argument("--resolutions", nargs="+", default=["D", "15", "5", "3"], help="Resolutions to fetch.")
    parser.add_argument("--from-date", default="2020-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--chunk-days", type=int, default=30, help="Days per request chunk.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between chunks.")
    
    args = parser.parse_args()
    
    import asyncio
    asyncio.run(backfill_data(args))
