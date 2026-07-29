"""
Heal missing F&O historical data in SQLite DB for the period 2026-06-12 to 2026-06-21.
Identifies symbols with missing/incomplete daily (D) or 15-minute (15) candles and fetches them.
"""

import sys
import logging
import asyncio
import time
from pathlib import Path
from datetime import date, datetime, timedelta
from sqlalchemy import text
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.domains.market_data.infrastructure.data.history import HistoricalDataService
from trade_system.domains.market_data.infrastructure.data.storage import CsvDataCatalog
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import save_market_data_batch
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(root_path / "logs/heal_missing_fo_data.log")
    ]
)
LOGGER = logging.getLogger("HealMissingFoData")

# Constants
START_DATE_STR = "2026-06-12"
END_DATE_STR = "2026-06-21"
EXPECTED_DAILY_CANDLES = 6       # 6 trading days: June 12, 15, 16, 17, 18, 19
EXPECTED_15M_CANDLES = 150       # 6 days * 25 candles/day

async def fetch_with_retry(service, symbol, res, from_date, to_date, retries=3):
    for attempt in range(1, retries + 1):
        try:
            # HistoricalDataService collects and also writes to CSV catalog
            df = service.collect(
                symbol=symbol,
                resolution=res,
                from_date=from_date,
                to_date=to_date,
                chunk_days=30,
                sleep_seconds=0.5
            )
            return df
        except Exception as e:
            LOGGER.warning(f"Attempt {attempt}/{retries} failed for {symbol} ({res}): {e}")
            if attempt == retries:
                raise e
            # Exponential backoff
            await asyncio.sleep(attempt * 2.5)

async def main():
    LOGGER.info("Starting audit of F&O database tables...")
    settings = Settings.load()
    engine = get_engine()
    universe = get_fo_universe()
    
    # Audit phase
    todo_daily = []
    todo_15m = []
    
    with Session(engine) as session:
        for idx, symbol in enumerate(universe):
            # Check Daily
            daily_count = session.execute(
                text("""
                    SELECT COUNT(*) FROM ohlcv_daily 
                    WHERE symbol = :symbol 
                      AND timestamp >= :start 
                      AND timestamp <= :end
                """),
                {"symbol": symbol, "start": f"{START_DATE_STR} 00:00:00", "end": f"{END_DATE_STR} 23:59:59"}
            ).scalar()
            
            if daily_count < EXPECTED_DAILY_CANDLES:
                todo_daily.append((symbol, daily_count))
                
            # Check 15m
            m15_count = session.execute(
                text("""
                    SELECT COUNT(*) FROM ohlcv_15m 
                    WHERE symbol = :symbol 
                      AND timestamp >= :start 
                      AND timestamp <= :end
                """),
                {"symbol": symbol, "start": f"{START_DATE_STR} 00:00:00", "end": f"{END_DATE_STR} 23:59:59"}
            ).scalar()
            
            if m15_count < EXPECTED_15M_CANDLES:
                todo_15m.append((symbol, m15_count))
                
    LOGGER.info(f"Audit Complete.")
    LOGGER.info(f"  - Daily todo list: {len(todo_daily)} symbols need healing.")
    LOGGER.info(f"  - 15-minute todo list: {len(todo_15m)} symbols need healing.")
    
    if len(todo_daily) > 0:
        LOGGER.info(f"Sample daily missing/incomplete: {todo_daily[:15]}")
    if len(todo_15m) > 0:
        LOGGER.info(f"Sample 15m missing/incomplete: {todo_15m[:15]}")
        
    if not todo_daily and not todo_15m:
        LOGGER.info("✅ All F&O symbols have complete daily and 15m data. No healing required!")
        return

    # Healing phase
    LOGGER.info("Initializing Fyers Client for healing...")
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    if not token:
        LOGGER.error("Could not obtain a valid Fyers access token. Aborting.")
        return
        
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
    )
    
    catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
    service = HistoricalDataService(broker, catalog)
    
    from_date = date.fromisoformat(START_DATE_STR)
    to_date = date.fromisoformat(END_DATE_STR)
    
    # Heal Daily Data
    if todo_daily:
        LOGGER.info(f"Healing daily data for {len(todo_daily)} symbols...")
        for i, (symbol, current_count) in enumerate(todo_daily):
            LOGGER.info(f"[{i+1}/{len(todo_daily)}] Healing daily for {symbol} (current count: {current_count})")
            try:
                df = await fetch_with_retry(service, symbol, "D", from_date, to_date)
                if not df.empty:
                    with Session(engine) as session:
                        saved = save_market_data_batch(session, symbol, "D", df.to_dict('records'))
                        LOGGER.info(f"  Successfully saved {saved} rows for {symbol} to DB.")
                else:
                    LOGGER.warning(f"  No daily data returned for {symbol}.")
            except Exception as e:
                LOGGER.error(f"  Failed to heal daily for {symbol}: {e}")
            await asyncio.sleep(0.5) # rate limit prevention

    # Heal 15m Data
    if todo_15m:
        LOGGER.info(f"Healing 15-minute data for {len(todo_15m)} symbols...")
        for i, (symbol, current_count) in enumerate(todo_15m):
            LOGGER.info(f"[{i+1}/{len(todo_15m)}] Healing 15m for {symbol} (current count: {current_count})")
            try:
                df = await fetch_with_retry(service, symbol, "15", from_date, to_date)
                if not df.empty:
                    with Session(engine) as session:
                        saved = save_market_data_batch(session, symbol, "15", df.to_dict('records'))
                        LOGGER.info(f"  Successfully saved {saved} rows for {symbol} to DB.")
                else:
                    LOGGER.warning(f"  No 15m data returned for {symbol}.")
            except Exception as e:
                LOGGER.error(f"  Failed to heal 15m for {symbol}: {e}")
            await asyncio.sleep(0.5) # rate limit prevention

    LOGGER.info("🎉 All healing operations completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
