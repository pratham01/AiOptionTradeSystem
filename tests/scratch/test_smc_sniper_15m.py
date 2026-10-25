import os, sys, sqlite3
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd

conn = sqlite3.connect('data/trade_system.db')
query = """
    SELECT * FROM ohlcv_15m 
    ORDER BY timestamp ASC
"""
df = pd.read_sql(query, conn)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

from trade_system.domains.strategy.application.indicators.smc_structure import SMCStructureEngine
from trade_system.domains.strategy.application.indicators.order_blocks import identify_order_blocks
engine = SMCStructureEngine(swing_length=5, internal_length=2)

symbols = df['symbol'].unique()

print("Testing SMC Sniper logic on 15m data...")
for symbol in symbols:
    sym_df = df[df['symbol'] == symbol].copy()
    if len(sym_df) < 20:
        continue
        
    last_vol = sym_df['volume'].iloc[-1]
    avg_vol = sym_df['volume'].iloc[-21:-1].mean()
    vol_surge = last_vol / avg_vol if avg_vol > 0 else 0
    
    if vol_surge < 1.5:
        continue
        
    try:
        state = engine.analyze(sym_df)
        
        struct_valid = False
        dist_to_ob = None
        close_last = sym_df['close'].iloc[-1]
        if state.is_internal_realigned or state.is_pullback:
            if state.strong_protected_level:
                dist_to_ob = abs((close_last - state.strong_protected_level) / close_last) * 100
            if dist_to_ob is not None and dist_to_ob <= 2.0:
                struct_valid = True
                
        obs = identify_order_blocks(sym_df, atr_multiplier=1.2)
        bullish_obs = obs.get('bullish', [])
        bearish_obs = obs.get('bearish', [])
        
        last_candle = sym_df.iloc[-1]
        c_open = last_candle['open']
        c_high = last_candle['high']
        c_low = last_candle['low']
        c_close = last_candle['close']
        c_body = abs(c_close - c_open)
        c_upper_wick = c_high - max(c_close, c_open)
        c_lower_wick = min(c_close, c_open) - c_low
        
        signal_type = None
        for ob in bullish_obs:
            if c_low <= ob['top'] and c_close >= ob['bottom'] and ob['start_date'] < last_candle['timestamp']:
                signal_type = "🟢 Daily Bullish OB Tap"
                break
        if not signal_type:
            for ob in bearish_obs:
                if c_high >= ob['bottom'] and c_close <= ob['top'] and ob['start_date'] < last_candle['timestamp']:
                    signal_type = "🔴 Daily Bearish OB Tap"
                    break
        if not signal_type:
            recent_low = float(sym_df['low'].tail(20).min())
            if c_low <= recent_low and c_lower_wick > (c_body * 2) and c_lower_wick > c_upper_wick:
                signal_type = "🟢 Daily Bullish Sweep"
            recent_high = float(sym_df['high'].tail(20).max())
            if c_high >= recent_high and c_upper_wick > (c_body * 2) and c_upper_wick > c_lower_wick:
                signal_type = "🔴 Daily Bearish Sweep"
    except Exception as e:
        print(f"ERROR on {symbol}: {e}")
print(f"Finished.")
