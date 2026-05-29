import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from datetime import date
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.data.fo_universe import get_sector_mapping
from sqlalchemy import text

def test_lookbacks():
    print("=== Testing Lookback Calculations ===")
    
    # 1. Fetch available dates
    engine = get_engine()
    query = text("SELECT DISTINCT date(timestamp) as d FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX' ORDER BY d DESC")
    with engine.connect() as conn:
        result = conn.execute(query).fetchall()
    available_dates = [row[0] for row in result if row[0] is not None]
    
    # Use the latest actual date in DB as target
    target_date = pd.to_datetime(available_dates[0]).date()
    print(f"Target Date: {target_date}")
    
    # Fetch last 10 days of data around target_date
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date(:latest_date, '-10 days') AND timestamp <= date(:latest_date, '+1 day')
        ORDER BY symbol, timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"latest_date": target_date.strftime("%Y-%m-%d")})
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    fo_metadata = get_sector_mapping()
    df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))
    
    lookbacks = ["Daily (Today)", "Last 15 Mins", "Last 1 Hour", "Last 2 Hours"]
    
    for lookback in lookbacks:
        print(f"\n--- Lookback: {lookback} ---")
        if lookback == "Last 15 Mins":
            step = 1
        elif lookback == "Last 1 Hour":
            step = 4
        elif lookback == "Last 2 Hours":
            step = 8
        else:
            step = None
            
        df_filtered = df[df['timestamp'].dt.date <= target_date]
        rows = []
        
        grouped = df_filtered[~df_filtered['symbol'].str.contains("INDEX")].groupby('symbol')
        for symbol, grp in grouped:
            grp_sorted = grp.sort_values('timestamp')
            if grp_sorted.empty:
                continue
                
            sector = fo_metadata.get(symbol, "UNKNOWN")
            
            # Latest close
            latest_row = grp_sorted.iloc[-1]
            close_last = float(latest_row['close'])
            
            # Reference close
            if step is not None:
                if len(grp_sorted) > step:
                    ref_idx = -step
                    close_prev = float(grp_sorted.iloc[ref_idx]['close'])
                else:
                    close_prev = float(grp_sorted.iloc[0]['close'])
            else:
                # Daily return
                prev_candles = grp_sorted[grp_sorted['timestamp'].dt.date < target_date]
                if not prev_candles.empty:
                    close_prev = float(prev_candles.iloc[-1]['close'])
                else:
                    close_prev = float(grp_sorted.iloc[0]['close'])
                    
            if close_prev > 0 and close_last > 0:
                pchange = ((close_last - close_prev) / close_prev) * 100
                rows.append({
                    "symbol": symbol,
                    "sector": sector,
                    "close_last": close_last,
                    "close_prev": close_prev,
                    "pChange": pchange
                })
                
        if rows:
            merged = pd.DataFrame(rows)
            sector_perf = merged.groupby('sector')['pChange'].mean().reset_index()
            sector_perf = sector_perf[sector_perf['sector'] != 'UNKNOWN']
            sector_perf = sector_perf.sort_values(by='pChange', ascending=False)
            
            print("Top 3 Sectors:")
            for idx, r in sector_perf.head(3).iterrows():
                print(f" - {r['sector']}: {r['pChange']:+.2f}%")
            print("Worst 3 Sectors:")
            for idx, r in sector_perf.tail(3).iloc[::-1].iterrows():
                print(f" - {r['sector']}: {r['pChange']:+.2f}%")
        else:
            print("No data available.")

if __name__ == "__main__":
    test_lookbacks()
