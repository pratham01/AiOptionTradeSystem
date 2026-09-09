import sqlite3
import pandas as pd
import numpy as np

def analyze_symbol(con, symbol, min_gain_pct=4.0):
    query = f"""
    SELECT timestamp, open, high, low, close, volume 
    FROM ohlcv_daily 
    WHERE symbol = '{symbol}'
    ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, con)
    if df.empty:
        print(f"No data for {symbol}")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df['prev_close'] = df['close'].shift(1)
    df['day_gain_pct'] = ((df['close'] - df['prev_close']) / df['prev_close']) * 100.0
    df['intraday_gain_pct'] = ((df['high'] - df['open']) / df['open']) * 100.0
    df['vol_20_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_20_sma']
    
    # 14-period ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['range'] = df['high'] - df['low']
    df['range_atr_ratio'] = df['range'] / df['atr_14']
    
    # Moving averages
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    
    # Bollinger Bandwidth
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (4.0 * df['bb_std']) / df['bb_mid'] * 100.0
    
    # NR7 (Narrowest range of last 7 bars)
    df['is_nr7'] = df['range'] == df['range'].rolling(7).min()
    # Inside bar
    df['is_inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    
    # Find big move days
    big_days = df[df['day_gain_pct'] >= min_gain_pct].copy()
    print(f"\n========================================================")
    print(f"ANALYSIS FOR {symbol} (Found {len(big_days)} days with gain >= {min_gain_pct}%)")
    print(f"========================================================")
    
    results = []
    for idx in big_days.index:
        if idx < 20:
            continue
        row_t = df.loc[idx]
        row_t1 = df.loc[idx - 1]
        row_t2 = df.loc[idx - 2]
        
        t1_date = row_t1['timestamp'].strftime('%Y-%m-%d')
        t_date = row_t['timestamp'].strftime('%Y-%m-%d')
        
        # Leading Footprint Metrics on T-1
        t1_vol_ratio = row_t1['vol_ratio']
        t1_range_ratio = row_t1['range_atr_ratio']
        t1_gain = row_t1['day_gain_pct']
        t1_bb_width = row_t1['bb_width']
        t1_nr7 = row_t1['is_nr7']
        t1_inside = row_t1['is_inside_bar']
        above_20ema = row_t1['close'] > row_t1['ema_20']
        above_50ema = row_t1['close'] > row_t1['ema_50']
        above_200sma = row_t1['close'] > row_t1['sma_200'] if pd.notna(row_t1['sma_200']) else True
        
        # Volatility squeeze (BB contraction on T-1)
        bb_contracting = row_t1['bb_width'] < row_t2['bb_width']
        
        print(f"\n--- BIG MOVE DAY (T): {t_date} | Gain: +{row_t['day_gain_pct']:.2f}% | Vol: {row_t['vol_ratio']:.2f}x ---")
        print(f"   [Day T-1 Footprint: {t1_date}]")
        print(f"   • T-1 Price Action  : Gain: {t1_gain:+.2f}% | Close: ₹{row_t1['close']:.2f}")
        print(f"   • T-1 Range Coiling : Range/ATR: {t1_range_ratio:.2f} | NR7: {t1_nr7} | Inside Bar: {t1_inside}")
        print(f"   • T-1 Volume Behavior: {t1_vol_ratio:.2f}x of 20-DMA (Dry-up: {t1_vol_ratio < 0.85}, Surge: {t1_vol_ratio > 1.3})")
        print(f"   • T-1 Squeeze Metric: BB Width: {t1_bb_width:.2f}% (Contracting: {bb_contracting})")
        print(f"   • T-1 Trend Baseline: >20 EMA: {above_20ema} | >50 EMA: {above_50ema} | >200 SMA: {above_200sma}")
        
        results.append({
            "t_date": t_date,
            "t_gain": row_t['day_gain_pct'],
            "t1_vol_ratio": t1_vol_ratio,
            "t1_range_ratio": t1_range_ratio,
            "t1_gain": t1_gain,
            "t1_nr7_or_inside": t1_nr7 or t1_inside,
            "above_20ema": above_20ema,
            "above_50ema": above_50ema,
            "bb_contracting": bb_contracting
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(f"\n>>> STATISTICAL SUMMARY FOR {symbol} ON DAY T-1:")
        print(f"   • % of times price was above 20 EMA on T-1: {res_df['above_20ema'].mean()*100:.1f}%")
        print(f"   • % of times price was above 50 EMA on T-1: {res_df['above_50ema'].mean()*100:.1f}%")
        print(f"   • % of times Range contracted (NR7/Inside Bar) on T-1: {res_df['t1_nr7_or_inside'].mean()*100:.1f}%")
        print(f"   • % of times Bollinger Band was contracting on T-1: {res_df['bb_contracting'].mean()*100:.1f}%")
        print(f"   • Median T-1 Volume Ratio: {res_df['t1_vol_ratio'].median():.2f}x")
        print(f"   • Median T-1 Range/ATR Ratio: {res_df['t1_range_ratio'].median():.2f}")

con = sqlite3.connect('data/trade_system.db')
analyze_symbol(con, 'NSE:ADANIENT-EQ', min_gain_pct=3.5)
analyze_symbol(con, 'NSE:PAYTM-EQ', min_gain_pct=4.0)
con.close()
