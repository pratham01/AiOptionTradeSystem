#!/usr/bin/env python
import sqlite3
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "data/trade_system.db"
# Test on a broad set of liquid F&O stocks
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

def backtest_exhaustion_reversals():
    print("🚀 Starting Isolated Backtest: EXHAUSTION REVERSALS (Top Gainers/Losers)")
    conn = sqlite3.connect(DB_PATH)
    
    long_trades = []
    short_trades = []
    
    for sym in tqdm(SYMBOLS, desc="Analyzing Stocks"):
        query = "SELECT * FROM ohlcv_daily WHERE symbol = ? ORDER BY timestamp ASC"
        df = pd.read_sql(query, conn, params=(sym,))
        if len(df) < 50:
            continue
            
        df['prev_close'] = df['close'].shift(1)
        df['day_gain_pct'] = ((df['close'] - df['prev_close']) / df['prev_close']) * 100.0
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        
        # Drop NaNs to avoid indexing issues
        df = df.dropna(subset=['rsi_14']).reset_index(drop=True)
        
        for i in range(1, len(df) - 5):  # Need 5 days forward for exit
            current_bar = df.iloc[i]
            
            # --- LONG SETUP (Top Loser + Oversold) ---
            if current_bar['day_gain_pct'] <= -4.0 and current_bar['rsi_14'] < 30:
                entry_price = df.iloc[i+1]['open']  # Enter next day open
                exit_price = df.iloc[i+5]['close']  # Exit 5 days later
                pnl = ((exit_price - entry_price) / entry_price) * 100
                long_trades.append({'symbol': sym, 'entry': entry_price, 'pnl': pnl})
                
            # --- SHORT SETUP (Top Gainer + Overbought) ---
            if current_bar['day_gain_pct'] >= 4.0 and current_bar['rsi_14'] > 70:
                entry_price = df.iloc[i+1]['open']  # Enter next day open
                exit_price = df.iloc[i+5]['close']  # Exit 5 days later
                pnl = ((entry_price - exit_price) / entry_price) * 100  # Inverse for short
                short_trades.append({'symbol': sym, 'entry': entry_price, 'pnl': pnl})

    print("\n============================================================")
    print("🎯 BACKTEST RESULTS: EXHAUSTION REVERSAL STRATEGY (5-Day Hold)")
    print("============================================================")
    
    def print_stats(trades, name):
        total = len(trades)
        if total == 0:
            print(f"{name}: 0 trades found.")
            return
            
        pnls = [t['pnl'] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = total - wins
        win_rate = (wins / total) * 100
        avg_pnl = sum(pnls) / total
        
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        print(f"\n--- {name} ---")
        print(f"Total Trades: {total}")
        print(f"Win Rate:     {win_rate:.1f}%")
        print(f"Avg PnL:      {avg_pnl:+.2f}%")
        print(f"Profit Factor:{profit_factor:.2f}")
        
    print_stats(long_trades, "LONG TRADES (Buying Oversold Top Losers)")
    print_stats(short_trades, "SHORT TRADES (Shorting Overbought Top Gainers)")
    print("============================================================\n")

if __name__ == "__main__":
    backtest_exhaustion_reversals()
