import os, sys, sqlite3
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from trade_system.domains.strategy.application.indicators.smc_structure import SMCStructureEngine

def run_filtered_test():
    conn = sqlite3.connect('data/trade_system.db')
    test_symbols = [
        "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:INFY-EQ",
        "NSE:TCS-EQ", "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ",
        "NSE:LT-EQ", "NSE:AXISBANK-EQ"
    ]
    
    trades = []
    engine = SMCStructureEngine(swing_length=5, internal_length=2)
    
    for symbol in test_symbols:
        df = pd.read_sql("SELECT timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE symbol = ? ORDER BY timestamp ASC", conn, params=(symbol,))
        if len(df) < 200:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['date'] = df['timestamp'].dt.date
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Track already mitigated OBs so we only trade FIRST TOUCH
        mitigated_obs = set()
        
        n = len(df)
        for i in range(50, n - 8):
            curr = df.iloc[i]
            c_open = curr['open']
            c_high = curr['high']
            c_low = curr['low']
            c_close = curr['close']
            c_body = abs(c_close - c_open)
            c_upper_wick = c_high - max(c_close, c_open)
            c_lower_wick = min(c_close, c_open) - c_low
            
            bar_hour = curr['timestamp'].hour
            bar_minute = curr['timestamp'].minute
            if bar_hour < 9 or (bar_hour == 9 and bar_minute < 30) or bar_hour >= 15:
                continue
                
            past_window = df.iloc[max(0, i-60) : i+1]
            
            sig = None
            direction = None
            
            # 1. Sweeps with Trend
            prior_low = past_window['low'].iloc[-16:-1].min()
            prior_high = past_window['high'].iloc[-16:-1].max()
            
            # Bullish Sweep + Above EMA50
            if c_low < prior_low and c_close > prior_low and c_close > c_open and c_lower_wick > c_upper_wick and c_close > curr['ema50']:
                sig = "Bullish Sweep (Trend Aligned)"
                direction = 1
            # Bearish Sweep + Below EMA50
            elif c_high > prior_high and c_close < prior_high and c_close < c_open and c_upper_wick > c_lower_wick and c_close < curr['ema50']:
                sig = "Bearish Sweep (Trend Aligned)"
                direction = -1
                
            # 2. First Touch Order Blocks with Trend Filter
            if not sig:
                # Find OBs in window
                body = (past_window['close'] - past_window['open']).abs()
                ab = body.rolling(14).mean()
                for k in range(len(past_window)-15, len(past_window)-2):
                    c0 = past_window.iloc[k]
                    c1 = past_window.iloc[k+1]
                    ob_id = (symbol, str(c0['timestamp']))
                    if ob_id in mitigated_obs:
                        continue
                    
                    # Bullish OB + Price > EMA50
                    if c0['close'] < c0['open'] and c1['close'] > c1['open'] and (c1['close'] - c1['open']) > (1.2 * ab.iloc[k]):
                        if c_close > curr['ema50'] and c_low <= c0['high'] and c_close >= c0['low'] and c_close > c_open:
                            mitigated_obs.add(ob_id)
                            sig = "Bullish OB 1st Tap (Trend Aligned)"
                            direction = 1
                            break
                    # Bearish OB + Price < EMA50
                    elif c0['close'] > c0['open'] and c1['close'] < c1['open'] and (c1['open'] - c1['close']) > (1.2 * ab.iloc[k]):
                        if c_close < curr['ema50'] and c_high >= c0['low'] and c_close <= c0['high'] and c_close < c_open:
                            mitigated_obs.add(ob_id)
                            sig = "Bearish OB 1st Tap (Trend Aligned)"
                            direction = -1
                            break
                            
            if sig and direction:
                entry = c_close
                curr_date = curr['date']
                exit_price = entry
                for fwd in range(1, 9):
                    f_bar = df.iloc[i + fwd]
                    if f_bar['date'] != curr_date:
                        break
                    exit_price = f_bar['close']
                    
                pnl = ((exit_price - entry) / entry * 100) if direction == 1 else ((entry - exit_price) / entry * 100)
                trades.append({'sig': sig, 'pnl': pnl})
                
    tdf = pd.DataFrame(trades)
    print("=== FILTERED INTRADAY RESULTS ===")
    for s in tdf['sig'].unique():
        sub = tdf[tdf['sig'] == s]
        print(f"[{s:35s}] Count: {len(sub):4d} | Win: {(sub['pnl']>0).mean()*100:.1f}% | Avg: {sub['pnl'].mean():+.2f}%")
    print(f"Total: {len(tdf)} | Overall Win: {(tdf['pnl']>0).mean()*100:.1f}% | Overall Avg: {tdf['pnl'].mean():+.2f}%")

if __name__ == "__main__":
    run_filtered_test()
