import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.indicators.supertrend import SupertrendIndicator

def load_data():
    """Load daily candle data for all symbols from the SQLite database."""
    db_path = root_path / "data" / "trade_system.db"
    if not db_path.exists():
        print(f"Database not found at {db_path}!")
        sys.exit(1)
        
    engine = create_engine(f"sqlite:///{db_path}")
    print(f"Reading from database {db_path}...")
    
    # Load daily candles
    query = """
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_daily 
        ORDER BY symbol, timestamp ASC
    """
    df = pd.read_sql(query, engine)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.replace(r"\.\d+", "", regex=True), errors="coerce")
    print(f"Loaded {len(df)} daily records across {df['symbol'].nunique()} symbols.")
    return df

def calculate_patterns(df):
    """Calculate NR7, Inside Bar, EMA 20, and Supertrend direction for each symbol's data."""
    # We will process group by group to avoid look-ahead bias
    processed_dfs = []
    
    # Initialize daily supertrend indicator
    st_indicator = SupertrendIndicator(period=10, multiplier=2)
    
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 21:
            continue
            
        # 1. Range calculations
        group["range"] = group["high"] - group["low"]
        
        # 2. NR7 detection: Range is the narrowest of the last 7 sessions (rolling min of last 7)
        # Using closed='right' to include current bar
        group["nr7"] = group["range"] == group["range"].rolling(window=7).min()
        
        # 3. Inside Bar detection: completely engulfed by the previous session
        group["prev_high"] = group["high"].shift(1)
        group["prev_low"] = group["low"].shift(1)
        group["inside_bar"] = (group["high"] < group["prev_high"]) & (group["low"] > group["prev_low"])
        
        # 4. EMA 20 for directional bias
        group["ema_20"] = group["close"].ewm(span=20, adjust=False).mean()
        
        # 5. Supertrend (daily regime)
        try:
            group = st_indicator.calculate(group)
        except Exception as e:
            # Fallback if supertrend calculation fails
            group["supertrend_direction"] = 0
            
        processed_dfs.append(group)
        
    return pd.concat(processed_dfs, ignore_index=True)

def run_backtest(df, use_nifty_regime=False, use_directional_bias=True, holding_period=5):
    """
    Simulates bracket order entry on the next trading day after an NR7 or Inside Bar is detected.
    """
    # Create Nifty Regime mapping by date
    nifty_df = df[df["symbol"] == "NSE:NIFTY50-INDEX"].sort_values("timestamp").reset_index(drop=True)
    if not nifty_df.empty:
        # Calculate Nifty Supertrend direction
        st_indicator = SupertrendIndicator(period=10, multiplier=2)
        try:
            nifty_df = st_indicator.calculate(nifty_df)
            nifty_regime = dict(zip(nifty_df["timestamp"].dt.date, nifty_df["supertrend_direction"]))
        except:
            nifty_regime = {}
    else:
        nifty_regime = {}
        
    trades = []
    
    # Process symbol by symbol
    for symbol, group in df.groupby("symbol"):
        if symbol in ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX", "NSE:NIFTYBANK-INDEX"]:
            continue # Skip index symbols for trading metrics
            
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_rows = len(group)
        
        for i in range(7, n_rows - 1):
            row = group.iloc[i]
            
            # Setup identified at EOD of day T (index i)
            is_nr7 = bool(row["nr7"])
            is_ib = bool(row["inside_bar"])
            
            if not (is_nr7 or is_ib):
                continue
                
            setup_date = row["timestamp"].date()
            setup_close = row["close"]
            setup_high = row["high"]
            setup_low = row["low"]
            setup_ema = row["ema_20"]
            
            # Directional bias based on EMA 20
            bias = "CALL" if setup_close > setup_ema else "PUT"
            
            # Determine pattern type
            pattern_type = "BOTH" if (is_nr7 and is_ib) else "NR7" if is_nr7 else "Inside Bar"
            
            # Filter check: Nifty Daily Regime Alignment on setup day
            if use_nifty_regime:
                nifty_dir = nifty_regime.get(setup_date, 0)
                if bias == "CALL" and nifty_dir != 1:
                    continue
                if bias == "PUT" and nifty_dir != -1:
                    continue
            
            # Initial Risk
            risk = setup_high - setup_low
            if risk <= 0:
                continue
                
            # Range size filter (limit coiling range to 2.5% of close price)
            range_pct = (risk / setup_close) * 100
            if range_pct > 2.5:
                continue
                
            # Simulate Day T+1 to Day T+Holding
            triggered = False
            position = None
            entry_price = 0.0
            entry_idx = i + 1
            
            # Day T+1 is the entry day.
            entry_day_row = group.iloc[entry_idx]
            entry_open = entry_day_row["open"]
            entry_high = entry_day_row["high"]
            entry_low = entry_day_row["low"]
            
            if use_directional_bias:
                if bias == "CALL":
                    if entry_high > setup_high:
                        triggered = True
                        position = "LONG"
                        entry_price = max(setup_high, entry_open)
                else: # bias == "PUT"
                    if entry_low < setup_low:
                        triggered = True
                        position = "SHORT"
                        entry_price = min(setup_low, entry_open)
            else:
                both_breached = (entry_high > setup_high) and (entry_low < setup_low)
                if both_breached:
                    triggered = True
                    position = "LONG" if entry_day_row["close"] < entry_open else "SHORT"
                    entry_price = max(setup_high, entry_open) if position == "LONG" else min(setup_low, entry_open)
                elif entry_high > setup_high:
                    triggered = True
                    position = "LONG"
                    entry_price = max(setup_high, entry_open)
                elif entry_low < setup_low:
                    triggered = True
                    position = "SHORT"
                    entry_price = min(setup_low, entry_open)
                    
            if not triggered:
                continue
                
            # Simulate exit holding period with 50% split booking & breakeven stops
            exit_reason = None
            exit_pnl_1 = 0.0
            exit_pnl_2 = 0.0
            t1_closed = False
            t2_closed = False
            exit_date = None
            
            target_1 = entry_price + risk if position == "LONG" else entry_price - risk
            target_2 = entry_price + 2.0 * risk if position == "LONG" else entry_price - 2.0 * risk
            sl_price = setup_low if position == "LONG" else setup_high
            
            end_idx = min(entry_idx + holding_period, n_rows)
            for j in range(entry_idx, end_idx):
                sim_row = group.iloc[j]
                sim_high = sim_row["high"]
                sim_low = sim_row["low"]
                sim_close = sim_row["close"]
                sim_open = sim_row["open"]
                sim_date = sim_row["timestamp"].date()
                
                if position == "LONG":
                    if not t1_closed:
                        if sim_low <= sl_price:
                            # Both stopped out at SL
                            t1_closed = True
                            t2_closed = True
                            exit_pnl_1 = (min(sl_price, sim_open) - entry_price) / entry_price * 100
                            exit_pnl_2 = exit_pnl_1
                            exit_reason = "STOP_LOSS"
                            exit_date = sim_date
                            break
                        elif sim_high >= target_1:
                            t1_closed = True
                            exit_pnl_1 = (max(target_1, sim_open) - entry_price) / entry_price * 100
                            sl_price = entry_price # Move stop to breakeven
                            
                            # Check if Tranche 2 is hit on the same day
                            if sim_low <= sl_price:
                                t2_closed = True
                                exit_pnl_2 = 0.0
                                exit_reason = "T1_HIT_BE_HIT"
                                exit_date = sim_date
                                break
                            elif sim_high >= target_2:
                                t2_closed = True
                                exit_pnl_2 = (max(target_2, sim_open) - entry_price) / entry_price * 100
                                exit_reason = "BOTH_TARGETS"
                                exit_date = sim_date
                                break
                    else: # Tranche 1 is closed, Tranche 2 is open (breakeven active)
                        if sim_low <= sl_price: # entry price
                            t2_closed = True
                            exit_pnl_2 = (min(sl_price, sim_open) - entry_price) / entry_price * 100
                            exit_reason = "T1_HIT_BE_HIT"
                            exit_date = sim_date
                            break
                        elif sim_high >= target_2:
                            t2_closed = True
                            exit_pnl_2 = (max(target_2, sim_open) - entry_price) / entry_price * 100
                            exit_reason = "BOTH_TARGETS"
                            exit_date = sim_date
                            break
                else: # SHORT position
                    if not t1_closed:
                        if sim_high >= sl_price:
                            t1_closed = True
                            t2_closed = True
                            exit_pnl_1 = (entry_price - max(sl_price, sim_open)) / entry_price * 100
                            exit_pnl_2 = exit_pnl_1
                            exit_reason = "STOP_LOSS"
                            exit_date = sim_date
                            break
                        elif sim_low <= target_1:
                            t1_closed = True
                            exit_pnl_1 = (entry_price - min(target_1, sim_open)) / entry_price * 100
                            sl_price = entry_price # Move stop to breakeven
                            
                            if sim_high >= sl_price:
                                t2_closed = True
                                exit_pnl_2 = 0.0
                                exit_reason = "T1_HIT_BE_HIT"
                                exit_date = sim_date
                                break
                            elif sim_low <= target_2:
                                t2_closed = True
                                exit_pnl_2 = (entry_price - min(target_2, sim_open)) / entry_price * 100
                                exit_reason = "BOTH_TARGETS"
                                exit_date = sim_date
                                break
                    else: # Tranche 1 is closed, Tranche 2 is open
                        if sim_high >= sl_price: # entry price
                            t2_closed = True
                            exit_pnl_2 = (entry_price - max(sl_price, sim_open)) / entry_price * 100
                            exit_reason = "T1_HIT_BE_HIT"
                            exit_date = sim_date
                            break
                        elif sim_low <= target_2:
                            t2_closed = True
                            exit_pnl_2 = (entry_price - min(target_2, sim_open)) / entry_price * 100
                            exit_reason = "BOTH_TARGETS"
                            exit_date = sim_date
                            break
                            
                # Time exit on last day of holding period
                if j == end_idx - 1:
                    exit_date = sim_date
                    if not t1_closed:
                        exit_pnl_1 = (sim_close - entry_price) / entry_price * 100 if position == "LONG" else (entry_price - sim_close) / entry_price * 100
                        exit_pnl_2 = exit_pnl_1
                        exit_reason = "TIME_EXIT"
                    else:
                        exit_pnl_2 = (sim_close - entry_price) / entry_price * 100 if position == "LONG" else (entry_price - sim_close) / entry_price * 100
                        exit_reason = "T1_HIT_TIME_EXIT"
                        
            pnl = 0.5 * exit_pnl_1 + 0.5 * exit_pnl_2
            
            trades.append({
                "symbol": symbol,
                "setup_date": setup_date,
                "pattern": pattern_type,
                "direction": "LONG" if position == "LONG" else "SHORT",
                "entry_price": entry_price,
                "exit_price": sim_close if not t2_closed else (target_2 if exit_reason == "BOTH_TARGETS" else sl_price),
                "exit_date": exit_date,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "risk_pct": (risk / entry_price) * 100
            })
            
    return pd.DataFrame(trades)

def print_stats(trades_df, title="Backtest Results"):
    """Calculate and print statistics from the backtest trades DataFrame."""
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)
    
    if trades_df.empty:
        print("No trades generated.")
        return
        
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df["pnl"] > 0]
    win_count = len(winning_trades)
    win_rate = (win_count / total_trades) * 100
    
    avg_pnl = trades_df["pnl"].mean()
    net_pnl = trades_df["pnl"].sum()
    
    gross_profits = winning_trades["pnl"].sum()
    gross_losses = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")
    
    avg_win = winning_trades["pnl"].mean() if win_count > 0 else 0.0
    avg_loss = trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if (total_trades - win_count) > 0 else 0.0
    
    print(f"Total Trades         : {total_trades}")
    print(f"Win Rate             : {win_rate:.2f}% ({win_count}/{total_trades})")
    print(f"Net Cumulative Return: {net_pnl:+.2f}%")
    print(f"Average Return/Trade : {avg_pnl:+.2f}%")
    print(f"Profit Factor        : {profit_factor:.2f}")
    print(f"Average Win          : {avg_win:+.2f}%")
    print(f"Average Loss         : {avg_loss:+.2f}%")
    
    print("\n--- Exit Reason Breakdown ---")
    reasons = trades_df["exit_reason"].value_counts()
    for reason, count in reasons.items():
        pct = (count / total_trades) * 100
        print(f"  {reason:<12}: {count:<4} ({pct:.1f}%)")
        
    print("\n--- Pattern Type Breakdown ---")
    for pattern, group in trades_df.groupby("pattern"):
        grp_total = len(group)
        grp_wr = (len(group[group["pnl"] > 0]) / grp_total) * 100
        grp_avg = group["pnl"].mean()
        grp_net = group["pnl"].sum()
        print(f"  {pattern:<12} -> Trades: {grp_total:<4} | Win Rate: {grp_wr:.1f}% | Avg PnL: {grp_avg:+.2f}% | Net PnL: {grp_net:+.2f}%")

def main():
    df = load_data()
    df_with_patterns = calculate_patterns(df)
    
    # Config 1: Directional Bias (EMA 20), No Nifty regime filter
    trades_dir_no_regime = run_backtest(df_with_patterns, use_nifty_regime=False, use_directional_bias=True)
    print_stats(trades_dir_no_regime, "CONFIG 1: Directional Bias (EMA 20) | No Nifty Regime Gate")
    
    # Config 2: Directional Bias (EMA 20), With Nifty regime filter
    trades_dir_with_regime = run_backtest(df_with_patterns, use_nifty_regime=True, use_directional_bias=True)
    print_stats(trades_dir_with_regime, "CONFIG 2: Directional Bias (EMA 20) | With Nifty Regime Gate")
    
    # Config 3: Bi-directional (no bias), No Nifty regime filter
    trades_bidir_no_regime = run_backtest(df_with_patterns, use_nifty_regime=False, use_directional_bias=False)
    print_stats(trades_bidir_no_regime, "CONFIG 3: Bi-directional (No Bias) | No Nifty Regime Gate")
    
    # Config 4: Bi-directional (no bias), With Nifty regime filter
    trades_bidir_with_regime = run_backtest(df_with_patterns, use_nifty_regime=True, use_directional_bias=False)
    print_stats(trades_bidir_with_regime, "CONFIG 4: Bi-directional (No Bias) | With Nifty Regime Gate")
    
    # Config 5: BOTH patterns only (Inside NR7), Bi-directional, With Nifty regime filter
    both_patterns_df = df_with_patterns.copy()
    # Mask to keep only BOTH rows (where both nr7 and inside_bar are true)
    # We need to preserve the historical sequences, but run_backtest expects the raw df.
    # So we'll run the backtest on the full df and then filter the trades by pattern == "BOTH".
    trades_both_only = trades_bidir_with_regime[trades_bidir_with_regime["pattern"] == "BOTH"]
    print_stats(trades_both_only, "CONFIG 5: Inside NR7 (BOTH) Only | Bi-directional | With Nifty Regime Gate")
    
    # Save the best configuration trades to CSV (Config 3 bi-directional is best overall, but Config 5 is the most precise/profitable setup)
    output_path = root_path / "scratch" / "backtest_nr7_results.csv"
    trades_bidir_no_regime.to_csv(output_path, index=False)
    print(f"\nSaved best config (Config 3) trades to {output_path}")

if __name__ == "__main__":
    main()

