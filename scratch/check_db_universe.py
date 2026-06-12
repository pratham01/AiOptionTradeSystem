from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///data/trade_system.db')
try:
    df = pd.read_sql("SELECT COUNT(DISTINCT symbol) as c FROM ohlcv_daily WHERE timestamp >= date('now', '-7 days')", engine)
    print(df)
except Exception as e:
    print(e)
