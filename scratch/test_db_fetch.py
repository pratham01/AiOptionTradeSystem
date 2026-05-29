import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import date
from trade_system.infrastructure.database.connection import get_engine
from sqlalchemy import text
import pandas as pd

def test_fetch():
    engine = get_engine()
    selected_date_str = date.today().strftime("%Y-%m-%d")
    print(f"Testing fetch for date: {selected_date_str}")
    
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date(:latest_date, '-10 days') AND timestamp <= date(:latest_date, '+1 day')
        ORDER BY symbol, timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"latest_date": selected_date_str})
        
    print(f"DataFrame shape: {df.shape}")
    if not df.empty:
        print(f"Min timestamp: {df['timestamp'].min()}")
        print(f"Max timestamp: {df['timestamp'].max()}")
        print(f"Number of unique symbols: {df['symbol'].nunique()}")
    else:
        print("DataFrame is empty!")

if __name__ == "__main__":
    test_fetch()
