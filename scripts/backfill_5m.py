import os
import sys
import logging
import time
from datetime import date, datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
import pandas as pd
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import Ohlcv5m
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

def main():
    settings = Settings.load()
    engine = get_engine()
    
    LOGGER.info("Initializing FyersBrokerClient...")
    broker = FyersBrokerClient(
        client_id=settings.fyers_client_id,
        access_token=settings.fyers_access_token,
    )
    
    symbols = get_fo_universe()
    LOGGER.info(f"Loaded {len(symbols)} F&O symbols.")
    
    end_date = date.today()
    start_date = end_date - timedelta(days=60)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    LOGGER.info(f"Backfill date range: {start_str} to {end_str}")
    
    total = len(symbols)
    success_count = 0
    total_records = 0
    
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    
    for i, symbol in enumerate(symbols):
        LOGGER.info(f"[{i+1}/{total}] Fetching 5m data for {symbol}...")
        try:
            df = broker.fetch_history(
                symbol=symbol,
                resolution="5",
                range_from=start_str,
                range_to=end_str
            )
            
            if df is None or df.empty:
                LOGGER.warning(f"No data returned for {symbol}.")
                continue
            
            # Sanitize data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Localize to IST if naive, or convert if tz-aware
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(IST)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(IST)
                
            # Filter weekdays
            df = df[df['timestamp'].dt.dayofweek < 5]
            
            # Filter market hours
            time_series = df['timestamp'].dt.time
            df = df[(time_series >= market_open) & (time_series <= market_close)]
            
            if df.empty:
                LOGGER.warning(f"No valid market-hour data for {symbol}.")
                continue
                
            # Convert timestamp back to naive for sqlite
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Prepare records
            records = df.to_dict('records')
            
            valid_records = []
            for r in records:
                # Add symbol column
                r['symbol'] = symbol
                try:
                    # ensure numeric types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        r[col] = float(r[col])
                    valid_records.append(r)
                except (ValueError, TypeError):
                    continue
            
            if not valid_records:
                continue
                
            # Upsert
            with Session(engine) as session:
                stmt = insert(Ohlcv5m).values(valid_records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "timestamp"],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                    }
                )
                session.execute(stmt)
                session.commit()
                
            inserted = len(valid_records)
            total_records += inserted
            success_count += 1
            LOGGER.info(f"Saved {inserted} records for {symbol}.")
            
        except Exception as e:
            LOGGER.error(f"Failed processing {symbol}: {e}")
            
        time.sleep(0.15)  # Avoid rate limits
        
    LOGGER.info(f"Backfill complete. Processed {success_count}/{total} symbols. Total records upserted: {total_records}")

if __name__ == "__main__":
    main()
