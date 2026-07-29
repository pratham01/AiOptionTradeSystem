import time
import pandas as pd
from datetime import datetime, date
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
import logging

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy import FyersBrokerClient, FyersAuthService
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import Ohlcv15m

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_live_loop():
    settings = Settings.load()
    engine = get_engine()
    # Sync both F&O universe and indices
    symbols = get_fo_universe() + settings.index_symbols
    
    logging.info(f"Starting continuous live 15m fetcher for {len(symbols)} F&O symbols...")

    while True:
        # Reload settings dynamically to pick up any updates (e.g. from other active processes)
        settings = Settings.load()
        auth = FyersAuthService(settings)
        token = auth.get_valid_token()

        if not token:
            logging.error("Failed to retrieve a valid Fyers token. Retrying in 60 seconds...")
            time.sleep(60)
            continue

        broker = FyersBrokerClient(
            client_id=settings.fyers.client_id,
            access_token=token,
            user_id=settings.fyers.user_id
        )

        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        # Only run during market hours (09:15 to 15:30)
        if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour >= 16:
            logging.info("Outside market hours. Sleeping for 5 minutes...")
            time.sleep(300)
            continue
            
        today_str = now.strftime("%Y-%m-%d")
        logging.info(f"Fetching latest 15m data for {today_str}...")
        
        total_added = 0
        total_skipped = 0
        start_time = time.time()
        rate_limited = False
        
        for idx, symbol in enumerate(symbols):
            if rate_limited:
                logging.warning(f"⚠️  Rate limit hit — skipping remaining {len(symbols)-idx} symbols. Will retry next cycle.")
                break
                
            if idx > 0 and idx % 20 == 0:
                logging.info(f"Progress: {idx}/{len(symbols)} symbols processed ({total_added} rows written).")
                
            try:
                df = broker.fetch_history(
                    symbol=symbol,
                    resolution="15",
                    range_from=today_str,
                    range_to=today_str,
                    date_format="1"
                )
                if df is None or df.empty:
                    total_skipped += 1
                    continue
                    
                records = []
                for _, row in df.iterrows():
                    dt = pd.to_datetime(row['timestamp'], format="mixed").to_pydatetime().replace(tzinfo=None)
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
                    total_skipped += 1
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
                # ─── Rate-limit protection ───────────────────────────────────
                # 0.6s per symbol = ~2 calls/sec, well within Fyers 10 calls/sec limit
                time.sleep(0.6)
                
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(kw in err_str for kw in [
                    "request limit", "rate limit", "too many request", "429", "limit reached"
                ])
                if is_rate_limit:
                    logging.warning(
                        f"🔴 Fyers rate limit hit on {symbol}. Pausing 60s before next cycle to let the API window reset."
                    )
                    rate_limited = True
                    time.sleep(60)  # Give API window time to reset
                else:
                    logging.error(f"Error processing {symbol}: {e}")
                    time.sleep(1.0)  # Small back-off on other errors
                
        duration = time.time() - start_time
        if rate_limited:
            logging.warning(
                f"⚠️  Sync cycle aborted early due to rate limiting. "
                f"Upserted {total_added} rows before limit hit. Sleeping 3 min before retry."
            )
        else:
            logging.info(
                f"✅ Loop completed! Upserted {total_added} rows across {len(symbols)} symbols "
                f"({total_skipped} empty). Took {duration:.1f}s."
            )
        
        logging.info("Sleeping for 3 minutes before next sync...")
        time.sleep(180)

if __name__ == "__main__":
    fetch_live_loop()
