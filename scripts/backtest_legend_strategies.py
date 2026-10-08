#!/usr/bin/env python
import sqlite3
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)

DB_PATH = "data/trade_system.db"
# Representative Liquid Universe
SYMBOLS = [
    "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:INFY-EQ",
    "NSE:TCS-EQ", "NSE:ITC-EQ", "NSE:LT-EQ", "NSE:SBIN-EQ", "NSE:BAJFINANCE-EQ",
    "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ", "NSE:AXISBANK-EQ", "NSE:ASIANPAINT-EQ",
    "NSE:MARUTI-EQ", "NSE:HCLTECH-EQ", "NSE:TATAMOTORS-EQ", "NSE:SUNPHARMA-EQ"
]

def calc_atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(n).mean()

def calc_adx(df, n=14):
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = calc_atr(df, 1).fillna(0.001)
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).sum() / tr.rolling(n).sum())
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).sum() / tr.rolling(n).sum())
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))
    return dx.rolling(n).mean()

def backtest_legends():
    print("🚀 LEGEND TRADERS BACKTEST (Global & Dubai Indian Traders)")
    conn = sqlite3.connect(DB_PATH)
    
    stats = {
        "1. Turtle Traders (Richard Dennis - 20D Breakout)": {"trades": 0, "wins": 0, "pnl": []},
        "2. VCP (Mark Minervini - Contraction Breakout)": {"trades": 0, "wins": 0, "pnl": []},
        "3. Darvas Box (Nicolas Darvas - 52W High Breakout)": {"trades": 0, "wins": 0, "pnl": []},
        "4. Holy Grail (Linda Raschke - ADX Pullback to 20 EMA)": {"trades": 0, "wins": 0, "pnl": []},
        "5. Power of Stocks (Subasish Pani - 5 EMA Pullback Reversal)": {"trades": 0, "wins": 0, "pnl": []},
        "6. Booming Bulls (Anish Singh - 200/50 EMA + Fib Pullback)": {"trades": 0, "wins": 0, "pnl": []}
    }
    
    for sym in tqdm(SYMBOLS, desc="Backtesting Strategies"):
        df = pd.read_sql("SELECT * FROM ohlcv_daily WHERE symbol = ? ORDER BY timestamp ASC", conn, params=(sym,))
        if len(df) < 250:
            continue
            
        df['atr'] = calc_atr(df)
        df['adx'] = calc_adx(df)
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        df['high_20d'] = df['high'].rolling(20).max().shift(1)
        df['low_10d'] = df['low'].rolling(10).min().shift(1)
        df['high_52w'] = df['high'].rolling(250).max().shift(1)
        
        for i in range(250, len(df) - 10):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            entry_price = df.iloc[i+1]['open']
            future_price = df.iloc[i+10]['close'] # Default 10 day hold for simplicity
            
            # 1. Turtle 20D Breakout
            if row['close'] > row['high_20d'] and prev['close'] <= prev['high_20d']:
                pnl = ((future_price - entry_price) / entry_price) * 100
                stats["1. Turtle Traders (Richard Dennis - 20D Breakout)"]["pnl"].append(pnl)
                
            # 2. Minervini VCP (Uptrend + Low Volume + Tight Range breakout)
            if row['close'] > row['ema_50'] and row['ema_50'] > row['ema_200']:
                if row['high'] - row['low'] < row['atr'] * 0.5: # Contraction
                    if df.iloc[i+1]['volume'] > df['volume'].rolling(20).mean().iloc[i] * 1.5:
                        pnl = ((future_price - entry_price) / entry_price) * 100
                        stats["2. VCP (Mark Minervini - Contraction Breakout)"]["pnl"].append(pnl)
            
            # 3. Darvas Box (Near 52W High Breakout)
            if row['close'] > row['high_52w'] * 0.98 and row['close'] > prev['close'] * 1.02:
                pnl = ((future_price - entry_price) / entry_price) * 100
                stats["3. Darvas Box (Nicolas Darvas - 52W High Breakout)"]["pnl"].append(pnl)
                
            # 4. Holy Grail (ADX > 30, Pullback touching 20 EMA)
            if row['adx'] > 30 and row['close'] > row['ema_20']:
                if row['low'] <= row['ema_20'] and prev['low'] > prev['ema_20']: # First touch
                    pnl = ((future_price - entry_price) / entry_price) * 100
                    stats["4. Holy Grail (Linda Raschke - ADX Pullback to 20 EMA)"]["pnl"].append(pnl)
                    
            # 5. Power of Stocks (5 EMA Detached Reversal - Daily scale)
            # A candle whose Low is completely above the 5 EMA. Next candle breaks its low (Short)
            if prev['low'] > prev['ema_5'] and row['close'] < prev['low']:
                # Short trade
                pnl = ((entry_price - future_price) / entry_price) * 100
                stats["5. Power of Stocks (Subasish Pani - 5 EMA Pullback Reversal)"]["pnl"].append(pnl)

            # 6. Booming Bulls (200/50 EMA Trend + Pullback)
            if row['ema_50'] > row['ema_200']:
                if row['low'] <= row['ema_50'] and row['close'] > row['ema_50']: # Hammer off 50 EMA
                    pnl = ((future_price - entry_price) / entry_price) * 100
                    stats["6. Booming Bulls (Anish Singh - 200/50 EMA + Fib Pullback)"]["pnl"].append(pnl)

    print("\n============================================================")
    print("🏆 GLOBAL & DUBAI YOUTUBE LEGENDS: STRATEGY SHOWDOWN (10-Day Hold)")
    print("============================================================")
    
    for name, data in stats.items():
        pnls = data["pnl"]
        total = len(pnls)
        if total == 0:
            print(f"\n{name}\nNo trades triggered.")
            continue
            
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / total) * 100
        avg_pnl = sum(pnls) / total
        pf = sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if sum(p for p in pnls if p < 0) != 0 else float('inf')
        
        print(f"\n{name}")
        print(f"Total Trades: {total} | Win Rate: {win_rate:.1f}%")
        print(f"Avg PnL: {avg_pnl:+.2f}% | Profit Factor: {pf:.2f}")
    print("============================================================\n")

if __name__ == "__main__":
    backtest_legends()
