"""
Script to migrate historical market data from CSV files to SQLite database.
"""

import sys
import pandas as pd
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime

# Add src to path
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import Base
from trade_system.infrastructure.database.repository import save_market_data_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

def cleanup_old_table(engine):
    """Drop the legacy consolidated market_data_candles table."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS market_data_candles;"))
            conn.commit()
            LOGGER.info("Dropped legacy market_data_candles table.")
    except Exception as e:
        LOGGER.warning(f"Could not drop legacy table: {e}")

def migrate_nifty_vix(csv_path: Path, session: Session):
    """Specific migration for Nifty with VIX data."""
    LOGGER.info(f"Migrating Nifty VIX from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Structure: date,open,high,low,close,volume,vix_close
    candles = []
    for _, row in df.iterrows():
        candles.append({
            "timestamp": pd.to_datetime(row["date"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "vix": float(row["vix_close"])
        })
        
        # Batch commit every 1000 rows for efficiency
        if len(candles) >= 1000:
            save_market_data_batch(session, "NSE:NIFTY50-INDEX", "D", candles)
            candles = []
            
    if candles:
        save_market_data_batch(session, "NSE:NIFTY50-INDEX", "D", candles)
    
    LOGGER.info("Nifty VIX migration complete.")

def migrate_fo_historical(base_dir: Path, session: Session):
    """Migrate standard F&O historical CSVs."""
    LOGGER.info(f"Scanning {base_dir} for F&O historical data...")
    csv_files = list(base_dir.glob("*.csv"))
    
    for csv_path in csv_files:
        try:
            # Filename pattern: NSE_ABB-EQ_d_historical.csv
            parts = csv_path.stem.split("_")
            if len(parts) < 3:
                continue
                
            symbol = parts[0] + ":" + parts[1] # e.g. NSE:ABB-EQ
            res_str = parts[2].upper()      # e.g. D or 3MIN
            
            # Normalize resolution for repository mapping
            if "MIN" in res_str:
                resolution = res_str.replace("MIN", "")
            else:
                resolution = res_str
            
            LOGGER.info(f"Migrating {symbol} ({resolution})...")
            df = pd.read_csv(csv_path)
            
            # Standard columns: timestamp,open,high,low,close,volume
            # Note: some files might have 'date' instead of 'timestamp'
            ts_col = "timestamp" if "timestamp" in df.columns else "date"
            
            candles = []
            for _, row in df.iterrows():
                try:
                    ts = row[ts_col]
                    if isinstance(ts, str):
                        ts = pd.to_datetime(ts)
                        
                    candles.append({
                        "timestamp": ts,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0.0))
                    })
                except Exception as e:
                    LOGGER.warning(f"Error parsing row in {csv_path}: {e}")
                    continue
                
                if len(candles) >= 2000:
                    save_market_data_batch(session, symbol, resolution, candles)
                    candles = []
            
            if candles:
                save_market_data_batch(session, symbol, resolution, candles)
                
        except Exception as e:
            LOGGER.error(f"Failed to migrate {csv_path}: {e}")

def main():
    engine = get_engine()
    
    # 0. Cleanup
    cleanup_old_table(engine)
    
    # Create tables if not exist
    Base.metadata.create_all(engine)
    
    with Session(engine) as session:
        # 1. Migrate specific Nifty VIX file if provided
        nifty_vix_file = root_path.parent / "nifty50_with_vix_column_20170801_to_20260317.csv"
        if nifty_vix_file.exists():
            migrate_nifty_vix(nifty_vix_file, session)
        else:
            LOGGER.warning(f"Nifty VIX file not found at {nifty_vix_file}")
            
        # 2. Migrate F&O directory
        fo_dir = root_path / "data" / "fo_historical"
        if fo_dir.exists():
            migrate_fo_historical(fo_dir, session)
            
    LOGGER.info("All migrations completed successfully.")

if __name__ == "__main__":
    main()
