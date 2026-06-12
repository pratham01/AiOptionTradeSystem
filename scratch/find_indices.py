import sys
from pathlib import Path
sys.path.append(str(Path("src").absolute()))

from trade_system.infrastructure.database.connection import get_engine
from sqlalchemy import text

engine = get_engine()

with engine.connect() as conn:
    # Find all unique symbols in ohlcv_15m containing 'INDEX'
    result = conn.execute(text("SELECT DISTINCT symbol FROM ohlcv_15m WHERE symbol LIKE '%INDEX%'")).fetchall()
    print("Indices in ohlcv_15m:", [row[0] for row in result])
    
    # Let's check ohlcv_daily too
    result_d = conn.execute(text("SELECT DISTINCT symbol FROM ohlcv_daily WHERE symbol LIKE '%INDEX%'")).fetchall()
    print("Indices in ohlcv_daily:", [row[0] for row in result_d])
