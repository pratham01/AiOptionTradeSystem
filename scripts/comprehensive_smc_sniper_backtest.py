#!/usr/bin/env python3
"""
Comprehensive Backtest for Daily SMC Sniper:
Tests Institutional Volume Surge (>= 1.5x) combined with:
1. Internal CHoCH Realignment (Trend Continuation)
2. Validated Order Block Tap & Bounce/Rejection
3. High-Volume Liquidity Sweeps
Compares Old Unaligned Logic vs Rectified Directional Precision.
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from trade_system.domains.strategy.application.indicators.smc_structure import SMCStructureEngine

def identify_daily_order_blocks(df: pd.DataFrame, atr_multiplier: float = 1.2) -> dict:
    if len(df) < 15:
        return {'bullish': [], 'bearish': []}
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(14).mean()
    bullish_obs = []
    bearish_obs = []
    for i in range(14, len(df) - 1):
        c0 = df.iloc[i]
        c1 = df.iloc[i+1]
        ab = avg_body.iloc[i]
        if c0['close'] < c0['open'] and c1['close'] > c1['open'] and (c1['close'] - c1['open']) > (atr_multiplier * ab):
            bullish_obs.append({
                'start_idx': i,
                'impulse_idx': i + 1,
                'timestamp': c0['timestamp'],
                'top': c0['high'],
                'bottom': c0['low']
            })
        elif c0['close'] > c0['open'] and c1['close'] < c1['open'] and (c1['open'] - c1['close']) > (atr_multiplier * ab):
            bearish_obs.append({
                'start_idx': i,
                'impulse_idx': i + 1,
                'timestamp': c0['timestamp'],
                'top': c0['high'],
                'bottom': c0['low']
            })
    return {'bullish': bullish_obs, 'bearish': bearish_obs}

def run_sniper_backtest():
    conn = sqlite3.connect('data/trade_system.db')
    query = "SELECT DISTINCT symbol FROM ohlcv_daily"
    symbols = [r[0] for r in conn.execute(query).fetchall()]
    
    print("================================================================================")
    print("🚀 COMPREHENSIVE DAILY SMC SNIPER BACKTEST (2020 - 2026)")
    print(f"Testing {len(symbols)} Stocks across Daily Timeframe with Volume Surge >= 1.5x")
    print("================================================================================\n")
    
    engine = SMCStructureEngine(swing_length=5, internal_length=2)
    
    trades = []
    
    for symbol in tqdm(symbols, desc="Backtesting SMC Sniper"):
        q = "SELECT timestamp, open, high, low, close, volume FROM ohlcv_daily WHERE symbol = ? ORDER BY timestamp ASC"
        df = pd.read_sql(q, conn, params=(symbol,))
        if len(df) < 60:
            continue
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        vol_sma = df['volume'].rolling(20).mean().replace(0, 1)
        df['vol_surge'] = df['volume'] / vol_sma
        
        n = len(df)
        for i in range(50, n - 10):
            curr = df.iloc[i]
            
            # Filter 1: Institutional Volume Heat
            if curr['vol_surge'] < 1.5:
                continue
                
            past_df = df.iloc[max(0, i-60) : i+1].copy()
            state = engine.analyze(past_df)
            
            c_open = curr['open']
            c_high = curr['high']
            c_low = curr['low']
            c_close = curr['close']
            c_body = abs(c_close - c_open)
            c_upper_wick = c_high - max(c_close, c_open)
            c_lower_wick = min(c_close, c_open) - c_low
            
            sig_name = None
            direction = None # 1 = Long, -1 = Short
            
            # Setup A: Internal CHoCH Realignment
            if state.is_internal_realigned:
                if state.swing_trend.value == "BULLISH":
                    sig_name = "Bullish Realignment (Internal CHoCH)"
                    direction = 1
                elif state.swing_trend.value == "BEARISH":
                    sig_name = "Bearish Realignment (Internal CHoCH)"
                    direction = -1
                    
            # Setup B: Order Block Tap & Defense
            if not sig_name:
                obs = identify_daily_order_blocks(past_df.iloc[:-1], atr_multiplier=1.2)
                for ob in obs['bullish']:
                    if ob['impulse_idx'] < len(past_df) - 1:
                        if c_low <= ob['top'] and c_close >= ob['bottom'] and c_close >= c_open:
                            sig_name = "Bullish OB Tap & Bounce"
                            direction = 1
                            break
                if not sig_name:
                    for ob in obs['bearish']:
                        if ob['impulse_idx'] < len(past_df) - 1:
                            if c_high >= ob['bottom'] and c_close <= ob['top'] and c_close <= c_open:
                                sig_name = "Bearish OB Tap & Reject"
                                direction = -1
                                break
                                
            # Setup C: Liquidity Sweep
            if not sig_name:
                prior_low = past_df['low'].iloc[-21:-1].min()
                prior_high = past_df['high'].iloc[-21:-1].max()
                if c_low < prior_low and c_close > prior_low and c_close > c_open and c_lower_wick > c_upper_wick:
                    sig_name = "Bullish Sweep & Reclaim"
                    direction = 1
                elif c_high > prior_high and c_close < prior_high and c_close < c_open and c_upper_wick > c_lower_wick:
                    sig_name = "Bearish Sweep & Reject"
                    direction = -1
                    
            if sig_name and direction:
                entry_price = c_close
                p3 = df.iloc[i+3]['close']
                p5 = df.iloc[i+5]['close']
                p10 = df.iloc[i+10]['close']
                
                ret_3d = ((p3 - entry_price) / entry_price * 100) if direction == 1 else ((entry_price - p3) / entry_price * 100)
                ret_5d = ((p5 - entry_price) / entry_price * 100) if direction == 1 else ((entry_price - p5) / entry_price * 100)
                ret_10d = ((p10 - entry_price) / entry_price * 100) if direction == 1 else ((entry_price - p10) / entry_price * 100)
                
                trades.append({
                    'symbol': symbol,
                    'timestamp': curr['timestamp'],
                    'signal': sig_name,
                    'direction': "LONG" if direction == 1 else "SHORT",
                    'entry_price': entry_price,
                    'vol_surge': curr['vol_surge'],
                    'ret_3d': ret_3d,
                    'ret_5d': ret_5d,
                    'ret_10d': ret_10d
                })
                
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print("No sniper setups triggered.")
        return
        
    print(f"\nTotal SMC Sniper Trades Triggered: {len(tdf)}")
    print("--------------------------------------------------------------------------------")
    for sig in tdf['signal'].unique():
        sub = tdf[tdf['signal'] == sig]
        win_3d = (sub['ret_3d'] > 0).mean() * 100
        win_5d = (sub['ret_5d'] > 0).mean() * 100
        win_10d = (sub['ret_10d'] > 0).mean() * 100
        avg_5d = sub['ret_5d'].mean()
        print(f"[{sig:38s}] Trades: {len(sub):4d} | Win (3d): {win_3d:.1f}% | Win (5d): {win_5d:.1f}% | Win (10d): {win_10d:.1f}% | Avg (5d): {avg_5d:+.2f}%")
        
    print("--------------------------------------------------------------------------------")
    win_5d_tot = (tdf['ret_5d'] > 0).mean() * 100
    avg_3d_tot = tdf['ret_3d'].mean()
    avg_5d_tot = tdf['ret_5d'].mean()
    avg_10d_tot = tdf['ret_10d'].mean()
    
    pos = tdf[tdf['ret_5d'] > 0]['ret_5d'].sum()
    neg = abs(tdf[tdf['ret_5d'] <= 0]['ret_5d'].sum())
    pf = (pos / neg) if neg > 0 else 999.0
    
    print(f"OVERALL WIN RATE (5-Day Hold): {win_5d_tot:.2f}%")
    print(f"AVERAGE RETURN (3-Day Hold):   {avg_3d_tot:+.2f}%")
    print(f"AVERAGE RETURN (5-Day Hold):   {avg_5d_tot:+.2f}%")
    print(f"AVERAGE RETURN (10-Day Hold):  {avg_10d_tot:+.2f}%")
    print(f"PROFIT FACTOR:                 {pf:.2f}")
    print("================================================================================\n")
    
    tdf.to_csv("scratch/smc_sniper_backtest_results.csv", index=False)
    print("Saved detailed trade log to scratch/smc_sniper_backtest_results.csv")

if __name__ == "__main__":
    run_sniper_backtest()
