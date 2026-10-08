#!/usr/bin/env python
import sqlite3
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from trade_system.domains.strategy.application.indicators.smc_structure import SMCStructureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "data/trade_system.db"
SYMBOLS_TO_TEST = ["NSE:HDFCBANK-EQ", "NSE:RELIANCE-EQ", "NSE:COLPAL-EQ", "NSE:INDHOTEL-EQ", "NSE:SOLARINDS-EQ", "NSE:VBL-EQ", "NSE:ADANIGREEN-EQ"]

def calculate_vol_surge(df, window=20):
    """Calculate volume surge compared to 20-period SMA."""
    if len(df) < window:
        return np.ones(len(df))
    vol_sma = df['volume'].rolling(window=window).mean()
    vol_sma = vol_sma.replace(0, 1)
    return (df['volume'] / vol_sma).fillna(1.0)

def backtest_smc_sniper():
    print("🚀 Starting SMC Sniper Deep Backtest on DAILY Timeframe...")
    conn = sqlite3.connect(DB_PATH)
    
    engine = SMCStructureEngine(swing_length=5, internal_length=2)
    
    total_trades = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    for sym in SYMBOLS_TO_TEST:
        query = "SELECT * FROM ohlcv_daily WHERE symbol = ? ORDER BY timestamp ASC"
        df = pd.read_sql(query, conn, params=(sym,))
        if len(df) < 100:
            continue
            
        df['vol_surge'] = calculate_vol_surge(df)
        
        print(f"Analyzing {sym} ({len(df)} daily candles)...")
        
        # Step 1 day at a time, check forward 5 days
        for i in tqdm(range(60, len(df) - 5, 1)):
            window_df = df.iloc[i-60:i].copy()
            current_bar = window_df.iloc[-1]
            
            # Condition 1: Volume Surge
            if current_bar['vol_surge'] < 1.5:
                continue
                
            state = engine.analyze(window_df)
            
            # Condition 2: Realignment or Pullback
            if state.is_internal_realigned or state.is_pullback:
                dist_to_ob = None
                if state.strong_protected_level:
                    dist_to_ob = abs((current_bar['close'] - state.strong_protected_level) / current_bar['close']) * 100
                
                # Condition 3: Proximity to Order Block
                if dist_to_ob is not None and dist_to_ob <= 2.0:
                    # Trade Triggered! Check forward return (5 days later)
                    entry_price = current_bar['close']
                    future_price = df.iloc[i+5]['close']
                    
                    if state.swing_trend.value == "BULLISH":
                        pnl_pct = ((future_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - future_price) / entry_price) * 100
                        
                    total_trades += 1
                    total_pnl += pnl_pct
                    
                    # Win condition: > 2.0% move in 5 days
                    if pnl_pct > 2.0:
                        wins += 1
                    else:
                        losses += 1

    print("\n============================================================")
    print("🎯 SMC SNIPER LOGIC - SIMULATION RESULTS (DAILY Timeframe)")
    print("============================================================")
    print(f"Total Snipe Setups Found: {total_trades}")
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        avg_pnl = total_pnl / total_trades
        print(f"Win Rate (>2.0% move in 5 days): {win_rate:.1f}%")
        print(f"Average PnL per trade: {avg_pnl:+.2f}%")
        print(f"Wins: {wins} | Losses: {losses}")
    print("============================================================\n")

if __name__ == "__main__":
    backtest_smc_sniper()
