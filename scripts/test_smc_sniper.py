import os, sys, sqlite3
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd

conn = sqlite3.connect('data/trade_system.db')
query = """
    SELECT * FROM ohlcv_daily 
    WHERE symbol IN ('NSE:TCS-EQ', 'NSE:TATAELXSI-EQ')
    ORDER BY timestamp ASC
"""
df = pd.read_sql(query, conn)

from trade_system.domains.strategy.application.indicators.smc_structure import SMCStructureEngine
engine = SMCStructureEngine(swing_length=5, internal_length=2)

for symbol in df['symbol'].unique():
    sym_df = df[df['symbol'] == symbol].copy()
    sym_df['timestamp'] = pd.to_datetime(sym_df['timestamp'])
    
    last_vol = sym_df['volume'].iloc[-1]
    avg_vol = sym_df['volume'].iloc[-21:-1].mean()
    vol_surge = last_vol / avg_vol if avg_vol > 0 else 0
    
    state = engine.analyze(sym_df)
    
    print(f"\n--- {symbol} ---")
    print(f"Volume Surge: {vol_surge:.2f}x")
    print(f"Internal Realigned: {state.is_internal_realigned}")
    print(f"Is Pullback: {state.is_pullback}")
    
    if state.strong_protected_level:
        close_last = sym_df['close'].iloc[-1]
        dist_to_ob = abs((close_last - state.strong_protected_level) / close_last) * 100
        print(f"Strong Protected Level: {state.strong_protected_level:.2f} (Dist: {dist_to_ob:.2f}%)")
    else:
        print("No Strong Protected Level found.")
