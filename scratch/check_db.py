from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('sqlite:///data/trade_system.db')

try:
    with engine.connect() as conn:
        df = pd.read_sql("SELECT DISTINCT date(timestamp) as dt FROM ohlcv_15m ORDER BY dt DESC LIMIT 15", conn)
        print("Dates in ohlcv_15m:")
        print(df)
except Exception as e:
    print(e)




