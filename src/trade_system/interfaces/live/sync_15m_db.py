import time
import pandas as pd
from datetime import datetime, date
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
import logging

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy import FyersBrokerClient, FyersAuthService
from trade_system.infrastructure.data.fo_universe import get_fo_universe
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import Ohlcv15m

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_live_loop():
    settings = Settings.load()
    engine = get_engine()
    symbols = get_fo_universe()
    
    logging.info(f"Starting continuous live 15m fetcher for {len(symbols)} F&O symbols...")

    while True:
        auth = FyersAuthService(settings)
        token = auth.read_cached_token() or settings.fyers.access_token

        broker = FyersBrokerClient(
            client_id=settings.fyers.client_id,
            access_token=token,
            user_id=settings.fyers.user_id
        )

        now = datetime.now()
        # Only run during market hours (09:15 to 15:30)
        if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour >= 16:
            logging.info("Outside market hours. Sleeping for 5 minutes...")
            time.sleep(300)
            continue
            
        today_str = now.strftime("%Y-%m-%d")
        logging.info(f"Fetching latest 15m data for {today_str}...")
        
        total_added = 0
        start_time = time.time()
        
        for idx, symbol in enumerate(symbols):
            if idx > 0 and idx % 20 == 0:
                logging.info(f"Progress: {idx}/{len(symbols)} symbols processed.")
                
            try:
                df = broker.fetch_history(
                    symbol=symbol,
                    resolution="15",
                    range_from=today_str,
                    range_to=today_str,
                    date_format="1"
                )
                if df.empty:
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
                time.sleep(0.1) # Respect broker rate limits
                
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                time.sleep(0.5)
                
        duration = time.time() - start_time
        logging.info(f"Loop completed! Upserted {total_added} rows across {len(symbols)} symbols in {duration:.1f} seconds.")
        
        # Sleep until the next 5-minute interval to keep data relatively fresh without aggressive rate limiting
        logging.info("Sleeping for 3 minutes before next sync...")
        time.sleep(180)

if __name__ == "__main__":
    fetch_live_loop()
