import sys
from pathlib import Path
from sqlalchemy.orm import Session
import pandas as pd

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import OhlcvDaily

def check_dates():
    engine = get_engine()
    symbol = "NSE:CUMMINSIND-EQ"
    
    with Session(engine) as session:
        # Fetch all records without limit
        candles = session.query(OhlcvDaily).filter(OhlcvDaily.symbol == symbol).order_by(OhlcvDaily.timestamp.asc()).all()
        print(f"Total daily candles in database for {symbol}: {len(candles)}")
        if candles:
            print(f"Earliest candle timestamp: {candles[0].timestamp}")
            print(f"Latest candle timestamp: {candles[-1].timestamp}")
            
            # Print the last 5 candles
            print("\nLast 5 candles in database:")
            for c in candles[-5:]:
                print(f"  {c.timestamp} | Open: {c.open} | High: {c.high} | Low: {c.low} | Close: {c.close} | Vol: {c.volume}")

if __name__ == "__main__":
    check_dates()
