import os, sys, sqlite3
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from trade_system.domains.strategy.application.indicators.smc_structure import SMCStructureEngine

def main():
    conn = sqlite3.connect('data/trade_system.db')
    query = """
        SELECT * FROM ohlcv_daily 
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    engine = SMCStructureEngine(swing_length=5, internal_length=2)
    
    symbols = df['symbol'].unique()
    print(f"Validating SMC Sniper logic for {len(symbols)} stocks...\n")
    
    passed_structure = []
    passed_volume = []
    passed_all = []
    
    for symbol in symbols:
        sym_df = df[df['symbol'] == symbol].copy()
        if len(sym_df) < 21:
            continue
            
        last_vol = sym_df['volume'].iloc[-1]
        avg_vol = sym_df['volume'].iloc[-21:-1].mean()
        vol_surge = last_vol / avg_vol if avg_vol > 0 else 0
        
        state = engine.analyze(sym_df)
        
        is_structure_valid = False
        dist_to_ob = None
        
        if state.is_internal_realigned or state.is_pullback:
            if state.strong_protected_level:
                close_last = sym_df['close'].iloc[-1]
                dist_to_ob = abs((close_last - state.strong_protected_level) / close_last) * 100
                if dist_to_ob <= 2.0:
                    is_structure_valid = True
                    passed_structure.append(symbol)
        
        if vol_surge >= 1.5:
            passed_volume.append(symbol)
            if is_structure_valid:
                passed_all.append(symbol)
                
        # Specifically print out TCS and TATAELXSI to explain why they were missed
        if symbol in ["NSE:TCS-EQ", "NSE:TATAELXSI-EQ"]:
            print(f"--- {symbol} ---")
            print(f"Volume Surge: {vol_surge:.2f}x (Threshold >= 1.5x)")
            print(f"Internal Realigned: {state.is_internal_realigned}")
            print(f"Is Pullback: {state.is_pullback}")
            if state.strong_protected_level:
                print(f"Strong Protected Level: {state.strong_protected_level:.2f} (Dist: {dist_to_ob if dist_to_ob else 999:.2f}%) (Threshold <= 2.0%)")
            else:
                print("Strong Protected Level: None")
            print(f"Passed Volume Filter: {vol_surge >= 1.5}")
            print(f"Passed Structure Filter: {is_structure_valid}")
            print(f"Passed ALL (Shown in Sniper): {vol_surge >= 1.5 and is_structure_valid}\n")

    print("--- SUMMARY ---")
    print(f"Stocks passing Volume Surge (>= 1.5x): {len(passed_volume)}")
    print(f"Stocks passing SMC Structure (CHoCH/Pullback & Dist <= 2%): {len(passed_structure)}")
    print(f"Stocks passing BOTH (Actually shown in SMC Sniper): {len(passed_all)}")
    print(f"SMC Sniper Final Candidates: {', '.join([s.replace('NSE:', '').replace('-EQ', '') for s in passed_all])}")

if __name__ == "__main__":
    main()
