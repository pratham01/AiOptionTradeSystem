#!/usr/bin/env python
import sqlite3
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "data/trade_system.db"
SYMBOLS = [
    "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:INFY-EQ",
    "NSE:TCS-EQ", "NSE:ITC-EQ", "NSE:LT-EQ", "NSE:SBIN-EQ", "NSE:BAJFINANCE-EQ",
    "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ", "NSE:AXISBANK-EQ", "NSE:ASIANPAINT-EQ",
    "NSE:MARUTI-EQ", "NSE:HCLTECH-EQ", "NSE:TATAMOTORS-EQ", "NSE:SUNPHARMA-EQ",
    "NSE:TATASTEEL-EQ", "NSE:ULTRACEMCO-EQ", "NSE:NTPC-EQ"
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def run_exit_method_backtest():
    print("🚀 Running Isolated Backtest: COMPARING ENTRY/EXIT METHODS")
    print("Strategy: Exhaustion Reversal (Buying Oversold Top Losers)")
    conn = sqlite3.connect(DB_PATH)
    
    # Trackers for different methods
    results = {
        "Fixed 5-Day Hold": [],
        "Standard Pivot Points (SL: S1, TP: R1)": [],
        "Fibonacci (TP: 61.8% Retracement, SL: Recent Low)": [],
        "ATR Trailing Stop (2x ATR)": []
    }
    
    for sym in tqdm(SYMBOLS, desc="Analyzing Stocks"):
        query = "SELECT * FROM ohlcv_daily WHERE symbol = ? ORDER BY timestamp ASC"
        df = pd.read_sql(query, conn, params=(sym,))
        if len(df) < 50:
            continue
            
        df['prev_close'] = df['close'].shift(1)
        df['day_gain_pct'] = ((df['close'] - df['prev_close']) / df['prev_close']) * 100.0
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['atr_14'] = calculate_atr(df, 14)
        
        # Calculate Daily Pivot Points (based on prev day)
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = (2 * df['pivot']) - df['low'].shift(1)
        df['s1'] = (2 * df['pivot']) - df['high'].shift(1)
        
        df = df.dropna(subset=['rsi_14', 'atr_14', 'pivot']).reset_index(drop=True)
        
        for i in range(20, len(df) - 20):
            current_bar = df.iloc[i]
            
            # Setup: Top Loser (>4% drop) AND Oversold (RSI < 30)
            if current_bar['day_gain_pct'] <= -4.0 and current_bar['rsi_14'] < 30:
                entry_price = df.iloc[i+1]['open']
                
                # --- Method 1: Fixed 5-Day Hold ---
                exit_price_fixed = df.iloc[i+5]['close']
                pnl_fixed = ((exit_price_fixed - entry_price) / entry_price) * 100
                results["Fixed 5-Day Hold"].append(pnl_fixed)
                
                # --- Method 2: Pivot Points ---
                tp_pivot = df.iloc[i+1]['r1']
                sl_pivot = df.iloc[i+1]['s1']
                # Forward simulation for pivots
                pnl_pivot = 0.0
                for j in range(i+1, min(i+11, len(df))): # 10 day max hold
                    if df.iloc[j]['high'] >= tp_pivot:
                        pnl_pivot = ((tp_pivot - entry_price) / entry_price) * 100
                        break
                    elif df.iloc[j]['low'] <= sl_pivot:
                        pnl_pivot = ((sl_pivot - entry_price) / entry_price) * 100
                        break
                if pnl_pivot == 0.0: # Time exit if neither hit
                    pnl_pivot = ((df.iloc[min(i+10, len(df)-1)]['close'] - entry_price) / entry_price) * 100
                results["Standard Pivot Points (SL: S1, TP: R1)"].append(pnl_pivot)
                
                # --- Method 3: Fibonacci Retracement ---
                # Calculate recent swing high over last 15 days before drop
                recent_high = df.iloc[i-15:i]['high'].max()
                recent_low = current_bar['low']
                drop_range = recent_high - recent_low
                tp_fibo = recent_low + (drop_range * 0.618) # 61.8% Retracement
                sl_fibo = recent_low * 0.99 # Stop just below the low
                
                pnl_fibo = 0.0
                for j in range(i+1, min(i+16, len(df))): # 15 day max hold
                    if df.iloc[j]['high'] >= tp_fibo:
                        pnl_fibo = ((tp_fibo - entry_price) / entry_price) * 100
                        break
                    elif df.iloc[j]['low'] <= sl_fibo:
                        pnl_fibo = ((sl_fibo - entry_price) / entry_price) * 100
                        break
                if pnl_fibo == 0.0:
                    pnl_fibo = ((df.iloc[min(i+15, len(df)-1)]['close'] - entry_price) / entry_price) * 100
                results["Fibonacci (TP: 61.8% Retracement, SL: Recent Low)"].append(pnl_fibo)
                
                # --- Method 4: ATR Trailing Stop ---
                atr_val = current_bar['atr_14']
                trailing_stop = entry_price - (2 * atr_val)
                highest_seen = entry_price
                
                pnl_atr = 0.0
                for j in range(i+1, min(i+21, len(df))): # 20 day max hold
                    if df.iloc[j]['high'] > highest_seen:
                        highest_seen = df.iloc[j]['high']
                        # Move stop up
                        new_stop = highest_seen - (2 * df.iloc[j]['atr_14'])
                        if new_stop > trailing_stop:
                            trailing_stop = new_stop
                    
                    if df.iloc[j]['low'] <= trailing_stop:
                        pnl_atr = ((trailing_stop - entry_price) / entry_price) * 100
                        break
                if pnl_atr == 0.0:
                    pnl_atr = ((df.iloc[min(i+20, len(df)-1)]['close'] - entry_price) / entry_price) * 100
                results["ATR Trailing Stop (2x ATR)"].append(pnl_atr)

    print("\n============================================================")
    print("📊 EXIT METHOD PERFORMANCE COMPARISON")
    print("============================================================")
    for name, pnls in results.items():
        if not pnls:
            print(f"{name}: No trades.")
            continue
        total = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / total) * 100
        avg_pnl = sum(pnls) / total
        
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        print(f"\nMethod: {name}")
        print(f"Win Rate:      {win_rate:.1f}%")
        print(f"Average PnL:   {avg_pnl:+.2f}%")
        print(f"Profit Factor: {pf:.2f}")
    print("============================================================\n")

if __name__ == "__main__":
    run_exit_method_backtest()
