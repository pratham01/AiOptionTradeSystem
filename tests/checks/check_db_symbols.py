import sys
from pathlib import Path
from sqlalchemy import func
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import OhlcvDaily
from trade_system.config import Settings
from trade_system.infrastructure.data.fo_universe import get_fo_universe

def check_symbols():
    engine = get_engine()
    universe = get_fo_universe()
    print(f"Total F&O Universe Size: {len(universe)}")
    
    with Session(engine) as session:
        # Group by symbol and count
        results = session.query(
            OhlcvDaily.symbol,
            func.count(OhlcvDaily.id),
            func.min(OhlcvDaily.timestamp),
            func.max(OhlcvDaily.timestamp)
        ).group_by(OhlcvDaily.symbol).all()
        
        print(f"Number of distinct symbols in OhlcvDaily: {len(results)}")
        
        # Check which universe symbols are missing
        present_symbols = {r[0] for r in results}
        missing_symbols = [s for s in universe if s not in present_symbols]
        print(f"Number of F&O universe symbols missing from OhlcvDaily: {len(missing_symbols)}")
        print(f"Sample missing symbols: {missing_symbols[:15]}")
        
        print("\nSymbols in OhlcvDaily with their candle counts:")
        for symbol, count, min_ts, max_ts in sorted(results, key=lambda x: x[1], reverse=True)[:30]:
            print(f"  {symbol}: {count} candles (from {min_ts} to {max_ts})")

if __name__ == "__main__":
    check_symbols()
