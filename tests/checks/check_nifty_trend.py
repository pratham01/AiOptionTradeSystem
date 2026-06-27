import sys
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import OhlcvDaily, Ohlcv1m, Ohlcv3m, Ohlcv15m
from trade_system.application.indicators.supertrend import calculate_supertrend
from trade_system.config import Settings

def check_nifty_trend():
    engine = get_engine()
    symbol = "NSE:NIFTY50-INDEX"
    today = "2026-05-27"
    
    print(f"Retrieving 3-minute bars for {symbol} on {today}...")
    
    with Session(engine) as session:
        # Let's query 1m data and resample it to 3m to see what we have
        from_dt = f"{today} 09:15:00"
        to_dt = f"{today} 15:30:00"
        
        # Query Ohlcv1m
        candles_1m = session.query(Ohlcv1m).filter(
            Ohlcv1m.symbol == symbol,
            Ohlcv1m.timestamp >= from_dt,
            Ohlcv1m.timestamp <= to_dt
        ).order_by(Ohlcv1m.timestamp.asc()).all()
        
        print(f"Number of 1m candles retrieved from database: {len(candles_1m)}")
        
        if not candles_1m:
            # Let's check Ohlcv3m directly
            candles_3m = session.query(Ohlcv3m).filter(
                Ohlcv3m.symbol == symbol,
                Ohlcv3m.timestamp >= from_dt,
                Ohlcv3m.timestamp <= to_dt
            ).order_by(Ohlcv3m.timestamp.asc()).all()
            print(f"Number of 3m candles retrieved from Ohlcv3m table: {len(candles_3m)}")
            if not candles_3m:
                print("No 3m candles found in the database!")
                return
            df_3m = pd.DataFrame([
                {"timestamp": c.timestamp, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
                for c in candles_3m
            ])
        else:
            df_1m = pd.DataFrame([
                {"timestamp": c.timestamp, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
                for c in candles_1m
            ])
            df_1m.set_index("timestamp", inplace=True)
            
            # Resample to 3m
            # open = first, high = max, low = min, close = last, volume = sum
            df_3m = df_1m.resample("3min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna().reset_index()
            
        print(f"Total 3-minute bars generated: {len(df_3m)}")
        
        # Calculate Supertrend
        df_3m.set_index("timestamp", inplace=True)
        # Period = 7, Multiplier = 3
        st_df = calculate_supertrend(df_3m, period=7, multiplier=3)
        
        if st_df.empty:
            print("Failed to calculate Supertrend (possibly insufficient bars)")
            return
            
        print("\nChecking for Supertrend flips:")
        prev_dir = None
        flips = []
        for idx, row in st_df.iterrows():
            curr_dir = row.get("supertrend_direction")
            if curr_dir is None or pd.isna(curr_dir):
                continue
            curr_dir = int(curr_dir)
            if prev_dir is not None and curr_dir != prev_dir:
                flips.append((idx, curr_dir, row["close"], row["supertrend"]))
            prev_dir = curr_dir
            
        print(f"Found {len(flips)} Supertrend flips today:")
        for idx, direction, close, st in flips:
            dir_str = "UP (GREEN)" if direction == 1 else "DOWN (RED)"
            print(f"  Time: {idx.strftime('%H:%M')} | Direction: {dir_str:<12} | Close: {close:.2f} | ST Value: {st:.2f}")
            
        # Print the last few bars
        print("\nLast 5 3-minute bars:")
        print(st_df[["open", "high", "low", "close", "supertrend", "supertrend_direction"]].tail(5))

if __name__ == "__main__":
    check_nifty_trend()
