import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from sqlalchemy import create_engine, text

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.strategy.application.indicators.supertrend import SupertrendIndicator

def load_data():
    db_path = root_path / "data" / "trade_system.db"
    if not db_path.exists():
        print(f"Database not found at {db_path}!")
        sys.exit(1)
        
    engine = create_engine(f"sqlite:///{db_path}")
    print(f"Reading daily candles...")
    
    # 1. Load daily candles from database
    query_daily = """
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_daily 
        ORDER BY symbol, timestamp ASC
    """
    df_daily = pd.read_sql(query_daily, engine)
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"].astype(str).str.replace(r"\.\d+", "", regex=True), errors="coerce")
    
    # Get max daily date
    max_daily_date = df_daily["timestamp"].max()
    print(f"Max date in ohlcv_daily: {max_daily_date.strftime('%Y-%m-%d')}")
    
    # 2. Load recent 15m candles to fill in missing days (from max_daily_date onwards)
    print(f"Reading 15-minute candles from {max_daily_date.strftime('%Y-%m-%d')}...")
    query_15m = f"""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= '{max_daily_date.strftime('%Y-%m-%d')}'
        ORDER BY symbol, timestamp ASC
    """
    df_15m = pd.read_sql(query_15m, engine)
    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"].astype(str).str.replace(r"\.\d+", "", regex=True), errors="coerce")
    
    if not df_15m.empty:
        # Resample 15m to daily
        df_15m["date"] = df_15m["timestamp"].dt.date
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        df_resampled = df_15m.groupby(["symbol", "date"]).agg(agg_rules).reset_index()
        df_resampled["timestamp"] = pd.to_datetime(df_resampled["date"])
        df_resampled = df_resampled.drop(columns=["date"])
        
        # Merge resampled with daily and drop duplicates
        df_combined = pd.concat([df_daily, df_resampled]).drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        df_combined = df_combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    else:
        df_combined = df_daily
        
    print(f"Combined daily records: {len(df_combined)} rows, max date is {df_combined['timestamp'].max().strftime('%Y-%m-%d')}")
    return df_combined

def calculate_patterns(df):
    processed_dfs = []
    st_indicator = SupertrendIndicator(period=10, multiplier=2)
    
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 21:
            continue
            
        group["range"] = group["high"] - group["low"]
        group["nr7"] = group["range"] == group["range"].rolling(window=7).min()
        
        group["prev_high"] = group["high"].shift(1)
        group["prev_low"] = group["low"].shift(1)
        group["inside_bar"] = (group["high"] < group["prev_high"]) & (group["low"] > group["prev_low"])
        
        group["ema_20"] = group["close"].ewm(span=20, adjust=False).mean()
        
        try:
            group = st_indicator.calculate(group)
        except Exception as e:
            group["supertrend_direction"] = 0
            
        processed_dfs.append(group)
        
    return pd.concat(processed_dfs, ignore_index=True)

def run_backtest_last_week(df, start_date_str="2026-05-29", end_date_str="2026-06-05"):
    """
    Runs the backtest specifically for setups detected from start_date_str to end_date_str.
    We track trades step-by-step to print a detailed log.
    """
    start_date = pd.to_datetime(start_date_str).date()
    end_date = pd.to_datetime(end_date_str).date()
    
    # Nifty Regime
    nifty_df = df[df["symbol"] == "NSE:NIFTY50-INDEX"].sort_values("timestamp").reset_index(drop=True)
    nifty_regime = {}
    if not nifty_df.empty:
        st_indicator = SupertrendIndicator(period=10, multiplier=2)
        try:
            nifty_df = st_indicator.calculate(nifty_df)
            nifty_regime = dict(zip(nifty_df["timestamp"].dt.date, nifty_df["supertrend_direction"]))
        except:
            pass
            
    trades = []
    
    for symbol, group in df.groupby("symbol"):
        if symbol in ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX", "NSE:NIFTYBANK-INDEX"]:
            continue
            
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_rows = len(group)
        
        for i in range(7, n_rows - 1):
            row = group.iloc[i]
            setup_date = row["timestamp"].date()
            
            # Only consider setups within the requested range
            if not (start_date <= setup_date <= end_date):
                continue
                
            is_nr7 = bool(row["nr7"])
            is_ib = bool(row["inside_bar"])
            
            if not (is_nr7 or is_ib):
                continue
                
            setup_close = row["close"]
            setup_high = row["high"]
            setup_low = row["low"]
            
            risk = setup_high - setup_low
            if risk <= 0:
                continue
                
            # Range size filter (limit coiling range to 2.5% of close price)
            range_pct = (risk / setup_close) * 100
            if range_pct > 2.5:
                continue
                
            # Simulate Day T+1 to Day T+5
            triggered = False
            position = None
            entry_price = 0.0
            entry_idx = i + 1
            
            entry_day_row = group.iloc[entry_idx]
            entry_open = entry_day_row["open"]
            entry_high = entry_day_row["high"]
            entry_low = entry_day_row["low"]
            entry_date = entry_day_row["timestamp"].date()
            
            # Bi-directional trigger check
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
            
            holding_period = 5
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
            
            pattern_type = "BOTH" if (is_nr7 and is_ib) else "NR7" if is_nr7 else "Inside Bar"
            
            trades.append({
                "symbol": symbol.split(":")[-1].replace("-EQ", ""),
                "setup_date": setup_date.strftime("%Y-%m-%d"),
                "pattern": pattern_type,
                "direction": "LONG" if position == "LONG" else "SHORT",
                "entry_price": round(entry_price, 2),
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_price": round(sim_close if not t2_closed else (target_2 if exit_reason == "BOTH_TARGETS" else sl_price), 2),
                "exit_date": exit_date.strftime("%Y-%m-%d") if exit_date else "",
                "exit_reason": exit_reason,
                "pnl": round(pnl, 2),
                "risk_pct": round((risk / entry_price) * 100, 2)
            })
            
    return pd.DataFrame(trades)

def generate_report(trades_df):
    report_path = root_path / "scratch" / "last_week_report.md"
    
    # Calculate statistics
    total = len(trades_df)
    if total == 0:
        content = "# Last Week Backtest Report\n\nNo trades triggered during last week (2026-06-01 to 2026-06-05)."
        with open(report_path, "w") as f:
            f.write(content)
        print("No trades found.")
        return
        
    winners = trades_df[trades_df["pnl"] > 0]
    win_rate = (len(winners) / total) * 100
    net_pnl = trades_df["pnl"].sum()
    avg_pnl = trades_df["pnl"].mean()
    
    gross_profits = winners["pnl"].sum()
    gross_losses = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")
    
    # Group by pattern
    pattern_breakdown = ""
    for pattern, grp in trades_df.groupby("pattern"):
        grp_tot = len(grp)
        grp_wr = (len(grp[grp["pnl"] > 0]) / grp_tot) * 100
        grp_net = grp["pnl"].sum()
        pattern_breakdown += f"| {pattern} | {grp_tot} | {grp_wr:.2f}% | {grp_net:+.2f}% |\n"
        
    # Trade table
    trades_table = "| Symbol | Setup Date | Pattern | Direction | Entry Date | Entry Px | Exit Date | Exit Px | Exit Reason | PnL (%) |\n"
    trades_table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for _, t in trades_df.sort_values("setup_date").iterrows():
        pnl_str = f"**{t['pnl']:+.2f}%**" if t['pnl'] > 0 else f"{t['pnl']:+.2f}%"
        trades_table += f"| {t['symbol']} | {t['setup_date']} | {t['pattern']} | {t['direction']} | {t['entry_date']} | {t['entry_price']} | {t['exit_date']} | {t['exit_price']} | {t['exit_reason']} | {pnl_str} |\n"
        
    report_content = f"""# Detailed Backtest Report: Last Week (June 1 - June 5, 2026)

This report presents a detailed trade-by-trade breakdown and performance analysis for the Narrow Range 7 (NR7) and Inside Bar compression breakout strategies. The simulation represents a **Bi-directional (No Bias) OCO Bracket Order** entry strategy using yesterday's high/low.

## 1. Performance Summary

| Metric | Value |
| :--- | :--- |
| **Total Trades** | {total} |
| **Win Rate** | {win_rate:.2f}% ({len(winners)}/{total}) |
| **Net Cumulative Return** | {net_pnl:+.2f}% |
| **Average Return/Trade** | {avg_pnl:+.2f}% |
| **Profit Factor** | {profit_factor:.2f} |

---

## 2. Pattern Breakdown

| Pattern | Trades | Win Rate | Net PnL |
| :--- | :--- | :--- | :--- |
{pattern_breakdown}

---

## 3. Detailed Trade Logs

{trades_table}
"""
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"Report saved to {report_path}")
    print(report_content)

def main():
    df = load_data()
    df_with_patterns = calculate_patterns(df)
    
    # Run backtest for last week (setup detected on May 29 to June 4)
    # The trades will trigger on June 1 through June 5.
    trades_df = run_backtest_last_week(df_with_patterns, start_date_str="2026-05-29", end_date_str="2026-06-04")
    generate_report(trades_df)

if __name__ == "__main__":
    main()
