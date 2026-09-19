import sys
import os
import pandas as pd
from datetime import date
from sqlalchemy import text

sys.path.append(os.path.abspath('src'))
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

target_date = date.today()
today_str = target_date.strftime("%Y-%m-%d")

engine = get_engine()
with engine.connect() as conn:
    df_15m = pd.read_sql(
        text("SELECT symbol, timestamp, close FROM ohlcv_15m WHERE date(timestamp) = :d AND symbol IN ('NSE:UNOMINDA-EQ', 'NSE:CGPOWER-EQ')"),
        conn, params={"d": today_str}
    )

print("DB 15m data for UNOMINDA and CGPOWER today:")
print(df_15m)

