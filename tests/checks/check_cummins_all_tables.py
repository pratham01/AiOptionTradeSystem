import sys
from pathlib import Path
from sqlalchemy import func
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import OhlcvDaily, Ohlcv15m, Ohlcv5m, Ohlcv3m, Ohlcv1m
from trade_system.shared.config import Settings

def check_all_tables():
    engine = get_engine()
    symbol = "NSE:CUMMINSIND-EQ"
    print(f"Checking data for {symbol} across all database tables...")
    
    with Session(engine) as session:
        for name, model in [
            ("OhlcvDaily (D)", OhlcvDaily),
            ("Ohlcv15m (15m)", Ohlcv15m),
            ("Ohlcv5m (5m)", Ohlcv5m),
            ("Ohlcv3m (3m)", Ohlcv3m),
            ("Ohlcv1m (1m)", Ohlcv1m)
        ]:
            count = session.query(func.count(model.id)).filter(model.symbol == symbol).scalar()
            print(f"  Table {name}: {count} rows")
            if count > 0:
                first = session.query(model.timestamp).filter(model.symbol == symbol).order_by(model.timestamp.asc()).first()
                last = session.query(model.timestamp).filter(model.symbol == symbol).order_by(model.timestamp.desc()).first()
                print(f"    Timestamp range: from {first[0]} to {last[0]}")

if __name__ == "__main__":
    check_all_tables()
