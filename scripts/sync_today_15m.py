import os
import sys
import time
from datetime import date
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager

def sync_fo_15m_candles(today_str: str = None) -> int:
    engine = get_engine()
    mgr = get_broker_manager()
    today_str = today_str or date.today().strftime("%Y-%m-%d")
    print(f"Syncing 15m data for today: {today_str}...")

    # Get symbols from ohlcv_15m
    with engine.connect() as conn:
        symbols = pd.read_sql(text("SELECT DISTINCT symbol FROM ohlcv_15m WHERE symbol NOT LIKE '%INDEX%'"), conn)['symbol'].tolist()
    print(f"Total symbols to sync: {len(symbols)}")

    success = 0
    total_rows = 0
    batch_rows = []

    for i, symbol in enumerate(symbols):
        for attempt in range(2):
            try:
                df = mgr.get_historical_data(symbol, "15", today_str, today_str)
                if df is not None and not df.empty:
                    df['symbol'] = symbol
                    # ensure proper columns: symbol, timestamp, open, high, low, close, volume
                    cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
                    for c in cols:
                        if c not in df.columns:
                            df[c] = 0
                    df = df[cols]
                    batch_rows.append(df)
                    success += 1
                    total_rows += len(df)
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "request limit" in err_msg or "429" in err_msg:
                    time.sleep(2.0)
                else:
                    break
        time.sleep(0.28)

        if (i + 1) % 25 == 0 or (i + 1) == len(symbols):
            print(f"[{i+1}/{len(symbols)}] Fetched {success} symbols, {total_rows} total rows...")

    if batch_rows:
        all_df = pd.concat(batch_rows, ignore_index=True)
        all_df['timestamp'] = pd.to_datetime(all_df['timestamp'])
        
        print(f"Writing {len(all_df)} rows to ohlcv_15m...")
        with engine.connect() as conn:
            # Delete today's partial stock data first to avoid duplicates
            conn.execute(text("DELETE FROM ohlcv_15m WHERE date(timestamp) = :t_date AND symbol NOT LIKE '%INDEX%'"), {"t_date": today_str})
            conn.commit()
            
            all_df.to_sql('ohlcv_15m', conn, if_exists='append', index=False)
            conn.commit()
        print(f"Successfully synced {len(all_df)} 15m candles into database!")
        return len(all_df)
    return 0

def main():
    sync_fo_15m_candles()

if __name__ == "__main__":
    main()
