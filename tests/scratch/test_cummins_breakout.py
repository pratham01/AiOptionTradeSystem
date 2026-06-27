import pandas as pd
from sqlalchemy import text
from datetime import date
from trade_system.infrastructure.database.connection import get_engine

engine = get_engine()

# Fetch Cummins data for today and past 10 days
query = text("""
    SELECT symbol, timestamp, open, high, low, close, volume 
    FROM ohlcv_15m 
    WHERE symbol='NSE:CUMMINSIND-EQ' 
      AND timestamp >= date('2026-05-27', '-10 days') 
      AND timestamp <= '2026-05-27 23:59:59'
    ORDER BY timestamp ASC
""")

with engine.connect() as conn:
    df = pd.read_sql(query, conn)

df['timestamp'] = pd.to_datetime(df['timestamp'])
today = date(2026, 5, 27)

# Calculate indicators
df['vol_sma_20'] = df['volume'].rolling(window=20).mean()

today_candles = df[df['timestamp'].dt.date == today].copy()
prev_candles = df[df['timestamp'].dt.date < today].copy()

print(f"Total candles today: {len(today_candles)}")
print(f"Total candles previous: {len(prev_candles)}")

if not today_candles.empty:
    # Session VWAP
    cum_pv = (today_candles['close'] * today_candles['volume']).cumsum()
    cum_vol = today_candles['volume'].cumsum()
    today_candles['vwap'] = cum_pv / cum_vol

    # ORB (First 4 candles of the day = 60 mins)
    orb_candles = today_candles.head(4)
    orb_high = orb_candles['high'].max()
    orb_low = orb_candles['low'].min()
    print(f"ORB High: {orb_high}, ORB Low: {orb_low}")

    prev_close = prev_candles.iloc[-1]['close'] if not prev_candles.empty else today_candles.iloc[0]['open']

    # Trace each candle today to see if it triggered and why it might have failed filters
    for i in range(len(today_candles)):
        candle = today_candles.iloc[i]
        ts = candle['timestamp']
        close = candle['close']
        vol = candle['volume']
        vwap = candle['vwap']
        
        # Get volume SMA from previous candle (current index - 1 in df)
        idx_in_df = df[df['timestamp'] == ts].index[0]
        prev_vol_sma = df.iloc[idx_in_df - 1]['vol_sma_20']
        
        vol_ratio = vol / prev_vol_sma if prev_vol_sma > 0 else 0
        pchange = ((close - prev_close) / prev_close) * 100
        
        is_triggered = close > orb_high
        is_vol_surge = vol > (2.0 * prev_vol_sma)
        is_vwap_align = close > vwap
        
        # Wick check
        candle_range = candle['high'] - candle['low']
        upper_wick = candle['high'] - close
        wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        is_wick_ok = wick_ratio <= 0.35
        
        # Let's print details
        print(f"\n--- Candle at {ts.strftime('%H:%M')} ---")
        print(f"  Price: Close={close}, High={candle['high']}, Low={candle['low']}, Volume={vol}")
        print(f"  VWAP: {vwap:.2f} (Align: {is_vwap_align})")
        print(f"  Vol SMA 20: {prev_vol_sma:.1f} | Ratio: {vol_ratio:.2f}x (Surge: {is_vol_surge})")
        print(f"  Change vs Prev Close ({prev_close:.2f}): {pchange:+.2f}%")
        print(f"  Triggered ORB Breakout (> {orb_high}): {is_triggered}")
        print(f"  Wick Shadow: {wick_ratio*100:.1f}% (OK: {is_wick_ok})")
        
        # Check if it would be bypass
        is_bypass = (pchange >= 3.0 and vol_ratio >= 2.5) or vol_ratio >= 4.0
        print(f"  Bypass Sector Check: {is_bypass}")
