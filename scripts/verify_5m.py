import os
import sys
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine

engine = get_engine()

query = text("""
    SELECT symbol, date(timestamp) as trade_date, count(*) as num_candles, min(time(timestamp)) as first_candle, max(time(timestamp)) as last_candle
    FROM ohlcv_5m
    WHERE symbol LIKE 'NSE:%-EQ'
    GROUP BY symbol, date(timestamp)
    ORDER BY symbol, trade_date DESC
    LIMIT 20
""")

with engine.connect() as conn:
    df = pd.read_sql(query, conn)

print(df.to_string())
