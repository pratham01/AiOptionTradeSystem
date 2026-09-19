import sys
import os
import pandas as pd
from sqlalchemy import text
from datetime import date

# Fix python path
sys.path.append(os.path.abspath('src'))

from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

fo_metadata = get_sector_mapping()
print(f"1. F&O Universe JSON count: {len(fo_metadata)}")

engine = get_engine()
with engine.connect() as conn:
    daily_count = conn.execute(text("SELECT count(distinct symbol) FROM ohlcv_daily")).scalar()
    print(f"2. distinct symbols in ohlcv_daily: {daily_count}")
    
    m15_count = conn.execute(text("SELECT count(distinct symbol) FROM ohlcv_15m")).scalar()
    print(f"3. distinct symbols in ohlcv_15m: {m15_count}")

# Simulate Sector Drill Down logic
df = pd.read_sql(
    text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date('2026-09-18', '-10 days') 
          AND timestamp < date('2026-09-18')
        ORDER BY symbol, timestamp ASC
    """),
    conn
)

df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))
print(f"4. Total rows fetched for Sector Scope (15m historical): {len(df)}")
print(f"5. Unique symbols fetched for Sector Scope (15m historical): {df['symbol'].nunique()}")

# Grouped logic
grouped = df[~df['symbol'].str.contains("INDEX")].groupby('symbol')
print(f"6. Unique symbols passing ~INDEX filter: {len(grouped)}")

