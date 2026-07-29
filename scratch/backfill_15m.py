import sys
import time
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
import logging

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy import FyersBrokerClient, FyersAuthService
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import Ohlcv15m

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def backfill():
    settings = Settings.load()
    engine = get_engine()
    all_symbols = get_fo_universe()
    
    # Identify which symbols are missing 15m data for any of the target dates
    # Target dates: 2026-06-08, 2026-06-09, 2026-06-10, 2026-06-11, 2026-06-12 (5 days)
    # Expected number of 15m candles per day = 25 (9:15 to 15:15 is 25 candles)
    # So 5 days * 25 candles = 125 candles expected.
    # Let's see which symbols have fewer than 100 candles.
    query = text("""
        SELECT symbol, COUNT(*) as c 
        FROM ohlcv_15m 
        WHERE timestamp >= '2026-06-08 00:00:00' 
          AND timestamp <= '2026-06-12 23:59:59' 
        GROUP BY symbol
    """)
    
    with engine.connect() as conn:
        counts = conn.execute(query).fetchall()
        
    counts_dict = {row[0]: row[1] for row in counts}
    symbols_to_fetch = []
    
    for symbol in all_symbols:
        count = counts_dict.get(symbol, 0)
        if count < 100: # Threshold of 100 candles for the 5 days
            symbols_to_fetch.append((symbol, count))
            
    if not symbols_to_fetch:
        logging.info("All symbols have sufficient 15m data. No backfill needed.")
        return
        
    logging.info(f"Identified {len(symbols_to_fetch)} symbols needing 15m backfill: {symbols_to_fetch}")
    
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    if not token:
        logging.error("Failed to retrieve a valid Fyers token.")
        return

    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id
    )

    start_date = "2026-06-08"
    end_date = "2026-06-12"
    
    total_added = 0
    start_time = time.time()
    
    for idx, (symbol, current_count) in enumerate(symbols_to_fetch):
        logging.info(f"[{idx+1}/{len(symbols_to_fetch)}] Healing {symbol} (has {current_count} candles)...")
        try:
            df = broker.fetch_history(
                symbol=symbol,
                resolution="15",
                range_from=start_date,
                range_to=end_date,
                date_format="1"
            )
            if df.empty:
                logging.warning(f"Empty data returned for {symbol}")
                continue
                
            records = []
            for _, row in df.iterrows():
                dt = pd.to_datetime(row['timestamp']).to_pydatetime().replace(tzinfo=None)
                records.append({
                    "symbol": symbol,
                    "timestamp": dt,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                })
                
            if not records:
                continue
                
            with Session(engine) as session:
                stmt = insert(Ohlcv15m).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'timestamp'],
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'volume': stmt.excluded.volume
                    }
                )
                session.execute(stmt)
                session.commit()
                
            total_added += len(records)
            time.sleep(0.5) # Sleep 500ms to avoid request rate limit
            
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            time.sleep(1.0)
            
    duration = time.time() - start_time
    logging.info(f"Backfill completed! Upserted {total_added} rows across {len(symbols_to_fetch)} symbols in {duration:.1f} seconds.")

if __name__ == "__main__":
    backfill()
