import sys
from pathlib import Path
from sqlalchemy import select, func
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import OhlcvDaily, Ohlcv5m, Ohlcv1m
from trade_system.config import Settings

def check_db():
    engine = get_engine()
    
    with Session(engine) as session:
        # Check total rows in Daily table
        daily_count = session.query(func.count(OhlcvDaily.id)).scalar()
        print(f"Total rows in OhlcvDaily: {daily_count}")
        
        # Check total rows in 5m table
        m5_count = session.query(func.count(Ohlcv5m.id)).scalar()
        print(f"Total rows in Ohlcv5m: {m5_count}")
        
        # Get some unique symbols in OhlcvDaily
        symbols_daily = session.query(OhlcvDaily.symbol).distinct().limit(20).all()
        print(f"Sample symbols in OhlcvDaily: {[s[0] for s in symbols_daily]}")
        
        # Search for CUMMINSIND specifically in all tables
        for model_name, model in [("Daily", OhlcvDaily), ("5m", Ohlcv5m), ("1m", Ohlcv1m)]:
            count = session.query(func.count(model.id)).filter(model.symbol.like("%CUMMINS%")).scalar()
            print(f"Rows matching 'CUMMINS' in {model_name}: {count}")
            if count > 0:
                sample = session.query(model.symbol).filter(model.symbol.like("%CUMMINS%")).distinct().all()
                print(f"  Distinct symbols matching 'CUMMINS': {[s[0] for s in sample]}")

if __name__ == "__main__":
    check_db()
