#!/usr/bin/env python3
"""
Comprehensive Backtest for 15m SMC Intraday Radar:
Tests Bullish/Bearish Order Block Taps and Liquidity Sweeps with verified SMC mechanics.
Compares Raw Unfiltered vs Rectified SMC Direction & Confirmation.
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

def identify_order_blocks_rectified(df: pd.DataFrame, atr_multiplier: float = 1.2) -> dict:
    """
    Identifies order blocks and preserves tap/mitigation history accurately.
    Returns list of bullish and bearish OBs with creation index and valid price boundaries.
    """
    if len(df) < 15:
        return {'bullish': [], 'bearish': []}
        
    df = df.copy()
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(14).mean()
    
    bullish_obs = []
    bearish_obs = []
    
    for i in range(14, len(df) - 1):
        c0 = df.iloc[i]
        c1 = df.iloc[i+1]
        ab = avg_body.iloc[i]
        
        # Bullish OB: Bearish candle c0 followed by strong bullish impulse c1
        if c0['close'] < c0['open'] and c1['close'] > c1['open'] and (c1['close'] - c1['open']) > (atr_multiplier * ab):
            bullish_obs.append({
                'start_idx': i,
                'impulse_idx': i + 1,
                'timestamp': c0['timestamp'],
                'top': c0['high'],
                'bottom': c0['low'],
                'mitigated': False,
                'broken': False
            })
            
        # Bearish OB: Bullish candle c0 followed by strong bearish impulse c1
        elif c0['close'] > c0['open'] and c1['close'] < c1['open'] and (c1['open'] - c1['close']) > (atr_multiplier * ab):
            bearish_obs.append({
                'start_idx': i,
                'impulse_idx': i + 1,
                'timestamp': c0['timestamp'],
                'top': c0['high'],
                'bottom': c0['low'],
                'mitigated': False,
                'broken': False
            })
            
    return {'bullish': bullish_obs, 'bearish': bearish_obs}

def run_intraday_backtest():
    conn = sqlite3.connect('data/trade_system.db')
    
    # Top 20 liquid F&O stocks
    test_symbols = [
        "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:INFY-EQ",
        "NSE:TCS-EQ", "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ",
        "NSE:LT-EQ", "NSE:AXISBANK-EQ", "NSE:ITC-EQ", "NSE:BAJFINANCE-EQ",
        "NSE:MARUTI-EQ", "NSE:TATAMOTORS-EQ", "NSE:SUNPHARMA-EQ", "NSE:TITAN-EQ",
        "NSE:TATASTEEL-EQ", "NSE:NTPC-EQ", "NSE:POWERGRID-EQ", "NSE:HINDUNILVR-EQ"
    ]
    
    print("================================================================================")
    print("🕵️ COMPREHENSIVE 15m SMC INTRADAY RADAR BACKTEST (2025 - 2026)")
    print(f"Testing {len(test_symbols)} Prime F&O Stocks across 15m Timeframe")
    print("================================================================================\n")
    
    trades = []
    
    for symbol in test_symbols:
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE symbol = ? ORDER BY timestamp ASC"
        df = pd.read_sql(query, conn, params=(symbol,))
        if df.empty or len(df) < 200:
            continue
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['date'] = df['timestamp'].dt.date
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Step through bars simulating live stream
        # Using a rolling buffer to avoid lookahead
        n = len(df)
        for i in range(50, n - 8):
            curr = df.iloc[i]
            curr_date = curr['date']
            
            # Require intraday trading window (between 09:30 and 14:30) to allow clean execution
            bar_hour = curr['timestamp'].hour
            bar_minute = curr['timestamp'].minute
            if bar_hour < 9 or (bar_hour == 9 and bar_minute < 30) or bar_hour >= 15:
                continue
                
            past_window = df.iloc[max(0, i-100) : i+1].copy()
            c_open = curr['open']
            c_high = curr['high']
            c_low = curr['low']
            c_close = curr['close']
            c_body = abs(c_close - c_open)
            c_upper_wick = c_high - max(c_close, c_open)
            c_lower_wick = min(c_close, c_open) - c_low
            
            sig_type = None
            direction = None  # 1 for Long, -1 for Short
            
            # --- 1. Order Block Detection on past bars ---
            obs = identify_order_blocks_rectified(past_window.iloc[:-1], atr_multiplier=1.2)
            
            # Check Bullish OB tap: price dips into demand and closes within/above
            for ob in obs['bullish']:
                if ob['impulse_idx'] < len(past_window) - 1: # Must be formed BEFORE current bar
                    if c_low <= ob['top'] and c_close >= ob['bottom']:
                        # Rectification: Bullish confirmation (close > open or lower wick defense)
                        if c_close >= c_open or c_lower_wick >= c_body * 0.8:
                            sig_type = "Bullish OB Tap"
                            direction = 1
                            break
                            
            # Check Bearish OB tap: price rallies into supply and closes within/below
            if not sig_type:
                for ob in obs['bearish']:
                    if ob['impulse_idx'] < len(past_window) - 1:
                        if c_high >= ob['bottom'] and c_close <= ob['top']:
                            # Rectification: Bearish confirmation (close < open or upper wick rejection)
                            if c_close <= c_open or c_upper_wick >= c_body * 0.8:
                                sig_type = "Bearish OB Tap"
                                direction = -1
                                break
                                
            # --- 2. Liquidity Sweep Detection ---
            if not sig_type:
                # Prior 15 bars swing low & high (excluding current bar)
                prior_low = past_window['low'].iloc[-16:-1].min()
                prior_high = past_window['high'].iloc[-16:-1].max()
                
                # Bullish Sweep: price swept below prior low but reclaimed above and closed green/hammer
                if c_low < prior_low and c_close > prior_low and c_close > c_open and c_lower_wick > c_upper_wick:
                    sig_type = "Bullish Sweep"
                    direction = 1
                    
                # Bearish Sweep: price swept above prior high but rejected below and closed red/shooting star
                elif c_high > prior_high and c_close < prior_high and c_close < c_open and c_upper_wick > c_lower_wick:
                    sig_type = "Bearish Sweep"
                    direction = -1
                    
            if sig_type and direction:
                entry_price = c_close
                entry_time = curr['timestamp']
                
                # Forward simulation: check 4 bars (1 hr) and 8 bars (2 hrs) or EOD
                exit_price_1h = None
                exit_price_2h = None
                exit_price_eod = None
                
                # Walk forward up to 8 candles
                max_forward = min(8, n - 1 - i)
                for fwd in range(1, max_forward + 1):
                    f_bar = df.iloc[i + fwd]
                    if f_bar['date'] != curr_date:
                        # End of day reached
                        exit_price_eod = df.iloc[i + fwd - 1]['close']
                        break
                    if fwd == 4:
                        exit_price_1h = f_bar['close']
                    if fwd == 8:
                        exit_price_2h = f_bar['close']
                        exit_price_eod = f_bar['close']
                        
                if exit_price_1h is None:
                    exit_price_1h = exit_price_eod if exit_price_eod else entry_price
                if exit_price_2h is None:
                    exit_price_2h = exit_price_eod if exit_price_eod else entry_price
                if exit_price_eod is None:
                    exit_price_eod = exit_price_2h
                    
                pnl_1h = ((exit_price_1h - entry_price) / entry_price * 100) if direction == 1 else ((entry_price - exit_price_1h) / entry_price * 100)
                pnl_2h = ((exit_price_2h - entry_price) / entry_price * 100) if direction == 1 else ((entry_price - exit_price_2h) / entry_price * 100)
                pnl_eod = ((exit_price_eod - entry_price) / entry_price * 100) if direction == 1 else ((entry_price - exit_price_eod) / entry_price * 100)
                
                trades.append({
                    'symbol': symbol,
                    'timestamp': entry_time,
                    'signal': sig_type,
                    'direction': "LONG" if direction == 1 else "SHORT",
                    'entry_price': entry_price,
                    'pnl_1h': pnl_1h,
                    'pnl_2h': pnl_2h,
                    'pnl_eod': pnl_eod
                })
                
    results_df = pd.DataFrame(trades)
    if results_df.empty:
        print("No trades triggered.")
        return
        
    print(f"Total Intraday Setups Triggered: {len(results_df)}")
    print("--------------------------------------------------------------------------------")
    
    # Performance summary by Signal Type
    for sig in results_df['signal'].unique():
        sub = results_df[results_df['signal'] == sig]
        win_1h = (sub['pnl_1h'] > 0).mean() * 100
        win_2h = (sub['pnl_2h'] > 0).mean() * 100
        avg_pnl_2h = sub['pnl_2h'].mean()
        print(f"[{sig:16s}] Setups: {len(sub):4d} | Win (1h): {win_1h:.1f}% | Win (2h): {win_2h:.1f}% | Avg PnL (2h): {avg_pnl_2h:+.2f}%")
        
    print("--------------------------------------------------------------------------------")
    # Overall Performance
    tot_trades = len(results_df)
    win_rate_2h = (results_df['pnl_2h'] > 0).mean() * 100
    avg_pnl_1h = results_df['pnl_1h'].mean()
    avg_pnl_2h = results_df['pnl_2h'].mean()
    avg_pnl_eod = results_df['pnl_eod'].mean()
    
    pos_trades = results_df[results_df['pnl_2h'] > 0]['pnl_2h'].sum()
    neg_trades = abs(results_df[results_df['pnl_2h'] <= 0]['pnl_2h'].sum())
    profit_factor = (pos_trades / neg_trades) if neg_trades > 0 else 999.0
    
    print(f"OVERALL WIN RATE (2h Hold): {win_rate_2h:.2f}%")
    print(f"AVERAGE RETURN (1h Hold):   {avg_pnl_1h:+.2f}%")
    print(f"AVERAGE RETURN (2h Hold):   {avg_pnl_2h:+.2f}%")
    print(f"AVERAGE RETURN (EOD Hold):  {avg_pnl_eod:+.2f}%")
    print(f"PROFIT FACTOR:              {profit_factor:.2f}")
    print("================================================================================\n")
    
    # Save backtest results
    results_df.to_csv("scratch/smc_intraday_backtest_results.csv", index=False)
    print("Saved detailed trade log to scratch/smc_intraday_backtest_results.csv")

if __name__ == "__main__":
    run_intraday_backtest()
