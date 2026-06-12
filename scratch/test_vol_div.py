import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import text
from trade_system.infrastructure.database.connection import get_engine
from trade_system.application.strategies.volume_divergence import VolumeDivergenceStrategy

def get_daily_candles_from_15m(df_15m):
    df = df_15m.copy()
    df["date"] = pd.to_datetime(df["timestamp"], format="mixed").dt.date
    daily = df.groupby(["symbol", "date"]).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    daily["timestamp"] = pd.to_datetime(daily["date"])
    return daily.drop(columns=["date"])

def main():
    engine = get_engine()
    
    # 1. Load data
    print("Loading daily data from DB...")
    with engine.connect() as conn:
        df_daily_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_daily"), conn)
        df_15m_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE timestamp >= '2026-05-05 00:00:00'"), conn)
        
    df_15m_raw["timestamp"] = pd.to_datetime(df_15m_raw["timestamp"], format="mixed")
    df_daily_raw["timestamp"] = pd.to_datetime(df_daily_raw["timestamp"], format="mixed")
    
    df_15m_daily = get_daily_candles_from_15m(df_15m_raw)
    df_daily_raw["date"] = df_daily_raw["timestamp"].dt.date
    df_15m_daily["date"] = df_15m_daily["timestamp"].dt.date
    
    dates_15m = set(df_15m_daily["date"])
    df_daily_filtered = df_daily_raw[~df_daily_raw["date"].isin(dates_15m)].copy()
    
    df_combined_daily = pd.concat([df_daily_filtered, df_15m_daily], ignore_index=True)
    df_combined_daily = df_combined_daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    print(f"Total Combined Daily Candles: {len(df_combined_daily)}")
    
    # Instantiate the strategy
    # Let's try different parameters (pivot_window=3, lookback=30)
    strategy = VolumeDivergenceStrategy(pivot_window=3, lookback=30)
    
    all_signals = []
    
    # Run strategy per symbol
    for symbol, grp in df_combined_daily.groupby("symbol"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < 40:
            continue
            
        res = strategy.generate_signals(grp)
        # Find rows where final_signal != 0
        signals = res[res["final_signal"] != 0].copy()
        if not signals.empty:
            for _, r in signals.iterrows():
                all_signals.append({
                    "symbol": symbol,
                    "timestamp": r["timestamp"],
                    "close": r["close"],
                    "obv": r["obv"],
                    "signal": r["final_signal"]
                })
                
    if not all_signals:
        print("No volume divergence signals found.")
        return
        
    df_sig = pd.DataFrame(all_signals)
    df_sig["date"] = df_sig["timestamp"].dt.date
    print(f"\nFound {len(df_sig)} Volume Divergence signals in history!")
    print(df_sig.sort_values("timestamp", ascending=False).head(30))

if __name__ == "__main__":
    main()
