import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import OhlcvDaily, Ohlcv15m

def search_db():
    engine = get_engine()
    
    with Session(engine) as session:
        # Search distinct symbols containing 'TATA' or 'GMR' or 'IDFC' in Daily table
        for pattern in ['%TATA%', '%GMR%', '%IDFC%', '%MCDOWELL%', '%L&T%', '%LT%']:
            daily_matches = session.query(OhlcvDaily.symbol).filter(OhlcvDaily.symbol.like(pattern)).distinct().all()
            m15_matches = session.query(Ohlcv15m.symbol).filter(Ohlcv15m.symbol.like(pattern)).distinct().all()
            
            print(f"Pattern '{pattern}':")
            print(f"  Daily matches: {[m[0] for m in daily_matches]}")
            print(f"  15m matches: {[m[0] for m in m15_matches]}")

if __name__ == "__main__":
    search_db()
