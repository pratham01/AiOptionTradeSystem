import sys
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import text

# Add src folder to system path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.strategy.application.indicators.compression import CompressionIndicator

def run_compression_analysis():
    engine = get_engine()
    
    # 1. Fetch recent 15m candles for a heavily traded stock like HDFCBANK or SBI
    # We will pick a few liquid symbols to see compression periods
    symbols_to_test = ["NSE:HDFCBANK-EQ", "NSE:SBIN-EQ", "NSE:RELIANCE-EQ"]
    
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv_15m
        WHERE symbol IN ('NSE:HDFCBANK-EQ', 'NSE:SBIN-EQ', 'NSE:RELIANCE-EQ')
        ORDER BY symbol, timestamp ASC
    """)
    
    print("Fetching 15m candle data from database...")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    if df.empty:
        print("❌ No 15-minute candle data found in database.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    indicator = CompressionIndicator(atr_period=14, lookback=4)
    
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        if len(group) < 30:
            print(f"Skipping {symbol}: Insufficient historical bars ({len(group)}).")
            continue
            
        print(f"\n==========================================")
        print(f"📊 Analyzing Volatility Compression for: {symbol}")
        print(f"==========================================")
        
        # Calculate compression
        calc_df = indicator.calculate(group)
        
        # Find compressed bars
        compressed_bars = calc_df[calc_df['is_compressed'] == True]
        print(f"Found {len(compressed_bars)} compressed bars out of {len(calc_df)} total bars.")
        
        # Print the latest 5 compressed instances and the subsequent price moves
        latest_compressed = compressed_bars.tail(5)
        
        if latest_compressed.empty:
            print("No compression setups detected for this stock in recent data.")
            continue
            
        for idx, row in latest_compressed.iterrows():
            ts = row['timestamp']
            close = row['close']
            atr = row['atr']
            nr7 = "NR7" if row['nr7'] else ""
            inside = "InsideBar" if row['inside_bar'] else ""
            comp_score = row['range_compression']
            
            # Look at the next 4 bars to see the subsequent movement (expansion/spike)
            next_bars = calc_df.iloc[idx+1 : idx+5]
            if not next_bars.empty:
                max_high = next_bars['high'].max()
                min_low = next_bars['low'].min()
                latest_close = next_bars.iloc[-1]['close']
                
                max_run_pct = ((max_high - close) / close) * 100
                max_drop_pct = ((min_low - close) / close) * 100
                net_move_pct = ((latest_close - close) / close) * 100
                
                details = f"[{inside} {nr7}]".strip(" []")
                print(f"🔹 Compressed Bar @ {ts} | Close: ₹{close:.2f} | ATR: ₹{atr:.2f} | Compression Score: {comp_score:.1f}% | Setup: {details}")
                print(f"  └─ Next 4 bars move -> Max High Run: +{max_run_pct:.2f}% | Max Low Drop: {max_drop_pct:.2f}% | Net Move: {net_move_pct:.2f}%")

if __name__ == "__main__":
    run_compression_analysis()
