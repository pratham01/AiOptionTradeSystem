import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import text
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.strategy.application.strategies.volume_divergence import VolumeDivergenceStrategy

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

def evaluate_reversal_performance(df_daily, signals_list):
    """
    Simulates trades for the generated signals and calculates returns for:
    - 3-day hold
    - 5-day hold
    - Target (+5%) / Stop Loss (-3%) exit within 10 days
    """
    trades_results = []
    
    # Sort daily data
    df = df_daily.sort_values(["symbol", "timestamp"]).copy()
    
    # Create dict of series per symbol for fast lookup
    symbol_groups = {symbol: grp.reset_index(drop=True) for symbol, grp in df.groupby("symbol")}
    
    for sig in signals_list:
        sym = sig["symbol"]
        sig_date = sig["timestamp"].date()
        direction = sig["signal"] # 1 for LONG, -1 for SHORT
        
        if sym not in symbol_groups:
            continue
            
        grp = symbol_groups[sym]
        # Find index of the signal date
        matches = grp[grp["timestamp"].dt.date == sig_date]
        if matches.empty:
            continue
            
        idx = matches.index[0]
        
        # We need future bars to evaluate
        if idx + 1 >= len(grp):
            continue # No future data
            
        entry_price = grp.iloc[idx]["close"]
        
        # 1. 3-day hold
        idx_3d = min(idx + 3, len(grp) - 1)
        exit_price_3d = grp.iloc[idx_3d]["close"]
        pnl_3d = (exit_price_3d - entry_price) / entry_price * 100 * direction
        
        # 2. 5-day hold
        idx_5d = min(idx + 5, len(grp) - 1)
        exit_price_5d = grp.iloc[idx_5d]["close"]
        pnl_5d = (exit_price_5d - entry_price) / entry_price * 100 * direction
        
        # 3. Target (+5%) and Stop Loss (-3%) within 10 days
        max_idx = min(idx + 10, len(grp) - 1)
        future_bars = grp.iloc[idx+1:max_idx+1]
        
        pnl_bracket = None
        exit_reason = "EXPIRED"
        
        target_pct = 5.0
        sl_pct = 3.0
        
        for _, bar in future_bars.iterrows():
            high_move = (bar["high"] - entry_price) / entry_price * 100 * direction
            low_move = (bar["low"] - entry_price) / entry_price * 100 * direction
            
            # Since high_move and low_move signs depend on direction, let's handle LONG and SHORT separately
            if direction == 1: # LONG
                # Check stop loss first (conservative)
                if (bar["low"] - entry_price) / entry_price * 100 <= -sl_pct:
                    pnl_bracket = -sl_pct
                    exit_reason = "STOP_LOSS"
                    break
                elif (bar["high"] - entry_price) / entry_price * 100 >= target_pct:
                    pnl_bracket = target_pct
                    exit_reason = "TARGET"
                    break
            else: # SHORT
                if (entry_price - bar["high"]) / entry_price * 100 <= -sl_pct:
                    pnl_bracket = -sl_pct
                    exit_reason = "STOP_LOSS"
                    break
                elif (entry_price - bar["low"]) / entry_price * 100 >= target_pct:
                    pnl_bracket = target_pct
                    exit_reason = "TARGET"
                    break
                    
        if pnl_bracket is None:
            # Expired, exit at the close of last bar
            last_bar = grp.iloc[max_idx]
            pnl_bracket = (last_bar["close"] - entry_price) / entry_price * 100 * direction
            exit_reason = "EXPIRED"
            
        trades_results.append({
            "symbol": sym,
            "date": sig_date,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": entry_price,
            "pnl_3d": pnl_3d,
            "pnl_5d": pnl_5d,
            "pnl_bracket": pnl_bracket,
            "exit_reason": exit_reason
        })
        
    return trades_results

def print_performance_table(name, df_trades, col_name):
    if df_trades.empty:
        print(f"{name}: No trades to evaluate.")
        return
        
    total = len(df_trades)
    win_rate = (df_trades[col_name] > 0).sum() / total * 100
    net_pnl = df_trades[col_name].sum()
    avg_pnl = df_trades[col_name].mean()
    
    gp = df_trades[df_trades[col_name] > 0][col_name].sum()
    gl = abs(df_trades[df_trades[col_name] <= 0][col_name].sum())
    pf = gp / gl if gl > 0 else float("inf")
    
    pf_str = f"{pf:.2f}" if pf != float("inf") else "Infinite"
    print(f"• {name:<12} | Trades: {total:4d} | WinRate: {win_rate:5.1f}% | NetPnL: {net_pnl:+7.2f}% | AvgPnL: {avg_pnl:+5.2f}% | PF: {pf_str}")

def main():
    engine = get_engine()
    
    # 1. Load data
    print("📥 Loading daily data from DB...")
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
    
    print(f"📊 Combined Daily candle dataset: {len(df_combined_daily)} rows across {df_combined_daily['symbol'].nunique()} symbols.")
    
    # We will test multiple configurations
    configs = [
        {"pivot": 2, "lookback": 20},
        {"pivot": 3, "lookback": 30},
        {"pivot": 5, "lookback": 50}
    ]
    
    for conf in configs:
        pivot = conf["pivot"]
        lookback = conf["lookback"]
        print(f"\n================================================================================")
        print(f"🔬 RUNNING BACKTEST: VolumeDivergenceStrategy (pivot_window={pivot}, lookback={lookback})")
        print(f"================================================================================")
        
        strategy = VolumeDivergenceStrategy(pivot_window=pivot, lookback=lookback)
        all_signals = []
        
        for symbol, grp in df_combined_daily.groupby("symbol"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            if len(grp) < lookback + pivot * 2:
                continue
            res = strategy.generate_signals(grp)
            signals = res[res["final_signal"] != 0]
            for _, r in signals.iterrows():
                all_signals.append({
                    "symbol": symbol,
                    "timestamp": r["timestamp"],
                    "signal": r["final_signal"]
                })
                
        if not all_signals:
            print("No signals found.")
            continue
            
        print(f"Generated {len(all_signals)} historical trend reversal signals.")
        trades = evaluate_reversal_performance(df_combined_daily, all_signals)
        df_trades = pd.DataFrame(trades)
        
        # 1. Evaluate 3-day hold
        print("\n📈 3-Session Hold Performance:")
        print("-" * 70)
        print_performance_table("Overall", df_trades, "pnl_3d")
        print_performance_table("Long Only", df_trades[df_trades["direction"] == "LONG"], "pnl_3d")
        print_performance_table("Short Only", df_trades[df_trades["direction"] == "SHORT"], "pnl_3d")
        
        # 2. Evaluate 5-day hold
        print("\n📈 5-Session Hold Performance:")
        print("-" * 70)
        print_performance_table("Overall", df_trades, "pnl_5d")
        print_performance_table("Long Only", df_trades[df_trades["direction"] == "LONG"], "pnl_5d")
        print_performance_table("Short Only", df_trades[df_trades["direction"] == "SHORT"], "pnl_5d")
        
        # 3. Evaluate Bracket hold (Target 5%, SL 3% within 10 days)
        print("\n📈 Bracket Hold Performance (Target +5% / Stop Loss -3% / 10-day Exp):")
        print("-" * 70)
        print_performance_table("Overall", df_trades, "pnl_bracket")
        print_performance_table("Long Only", df_trades[df_trades["direction"] == "LONG"], "pnl_bracket")
        print_performance_table("Short Only", df_trades[df_trades["direction"] == "SHORT"], "pnl_bracket")
        
        # Print exit reasons breakdown for bracket hold
        print("\n🚪 Bracket Exit Reason Distribution:")
        print(df_trades["exit_reason"].value_counts().to_string())

if __name__ == "__main__":
    main()
