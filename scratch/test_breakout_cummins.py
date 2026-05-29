import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.infrastructure.database.connection import get_engine
from trade_system.application.analysis.breakout_screener import BreakoutScreener

def test_breakout():
    engine = get_engine()
    symbol = "NSE:CUMMINSIND-EQ"
    target_date = "2026-05-27"
    
    screener = BreakoutScreener()
    
    # Let's first run the screener's query to get the raw 15m data for Cummins
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE symbol = :symbol AND timestamp >= date(:target_date, '-10 days') AND timestamp <= datetime(:target_date || ' 15:30:00')
        ORDER BY timestamp ASC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol, "target_date": target_date})
        
    print(f"Retrieved {len(df)} candles for {symbol} up to {target_date} 15:30:00")
    if df.empty:
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['sector'] = df['symbol'].apply(lambda s: screener.fo_metadata.get(s, "UNKNOWN"))
    
    # 20-period Volume SMA
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    
    # Today's candles
    today = pd.to_datetime(target_date).date()
    today_candles = df[df['timestamp'].dt.date == today].copy()
    print(f"Number of candles today ({today}): {len(today_candles)}")
    if len(today_candles) == 0:
        return
        
    # Calculate session VWAP
    cum_pv = (today_candles['close'] * today_candles['volume']).cumsum()
    cum_vol = today_candles['volume'].cumsum()
    today_candles['vwap'] = cum_pv / cum_vol
    
    # Opening Range (First 4 candles of the day)
    orb_candles = today_candles.head(4)
    orb_high = orb_candles['high'].max()
    orb_low = orb_candles['low'].min()
    
    print("\nORB Levels (first 60 minutes):")
    print(f"  ORB High: {orb_high:.2f}")
    print(f"  ORB Low: {orb_low:.2f}")
    
    # Let's inspect each candle of today to see if any candle met the breakout condition
    print("\nIntraday Candle-by-Candle analysis for today:")
    print(f"{'Time':<9} | {'Close':<8} | {'Volume':<7} | {'Vol SMA':<7} | {'Vol Ratio':<9} | {'VWAP':<8} | {'High':<8} | {'Wick%':<5} | {'pChange':<8} | {'ORB Break':<9}")
    print("-" * 95)
    
    prev_day_candles = df[df['timestamp'].dt.date < today]
    prev_close = prev_day_candles.iloc[-1]['close'] if not prev_day_candles.empty else today_candles.iloc[0]['open']
    
    for i in range(len(today_candles)):
        row = today_candles.iloc[i]
        t_str = row['timestamp'].strftime('%H:%M')
        
        # Volume SMA from the previous candle
        if i == 0:
            prev_vol_sma = prev_day_candles.iloc[-1]['vol_sma_20'] if not prev_day_candles.empty else row['vol_sma_20']
        else:
            prev_vol_sma = today_candles.iloc[i-1]['vol_sma_20']
            
        vol_ratio = row['volume'] / prev_vol_sma if pd.notna(prev_vol_sma) and prev_vol_sma > 0 else 0.0
        pchange = ((row['close'] - prev_close) / prev_close) * 100
        
        # Wick rejection
        candle_range = row['high'] - row['low']
        wick_pct = 0.0
        if candle_range > 0:
            wick_pct = (row['high'] - row['close']) / candle_range * 100
            
        orb_break = "NO"
        if row['close'] > orb_high:
            orb_break = "LONG"
        elif row['close'] < orb_low:
            orb_break = "SHORT"
            
        print(f"{t_str:<9} | {row['close']:8.2f} | {row['volume']:7.0f} | {prev_vol_sma:7.0f} | {vol_ratio:8.2f}x | {row['vwap']:8.2f} | {row['high']:8.2f} | {wick_pct:4.1f}% | {pchange:7.2f}% | {orb_break:<9}")

    # Let's run the full scan to see the sector performance and leading sectors today
    print("\nRunning full scan for sector performance analysis on May 27...")
    all_alerts = screener.scan_for_breakouts(target_date=today)
    
    cummins_alerts = [a for a in all_alerts if a['symbol'] == symbol]
    if cummins_alerts:
        print(f"\nSUCCESS! Cummins alert found: {cummins_alerts}")
    else:
        print("\nCummins was not in the triggered breakout alerts.")
        
if __name__ == "__main__":
    test_breakout()
