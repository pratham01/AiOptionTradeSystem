import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine

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
    query = """
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_daily 
        ORDER BY symbol, timestamp ASC
    """
    df = pd.read_sql(query, engine)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.replace(r"\.\d+", "", regex=True), errors="coerce")
    return df

def calculate_indicators(df):
    processed_dfs = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 21:
            continue
            
        group["range"] = group["high"] - group["low"]
        group["range_pct"] = (group["range"] / group["close"]) * 100
        
        # NR7
        group["nr7"] = group["range"] == group["range"].rolling(window=7).min()
        
        # Inside Bar
        group["prev_high"] = group["high"].shift(1)
        group["prev_low"] = group["low"].shift(1)
        group["inside_bar"] = (group["high"] < group["prev_high"]) & (group["low"] > group["prev_low"])
        
        # EMA 20
        group["ema_20"] = group["close"].ewm(span=20, adjust=False).mean()
        
        # Volume MA
        group["vol_ma_20"] = group["volume"].rolling(20).mean()
        group["vol_ratio"] = group["volume"] / group["vol_ma_20"].replace(0, 1)
        
        # ATR (14-period)
        tr0 = abs(group["high"] - group["low"])
        tr1 = abs(group["high"] - group["close"].shift(1))
        tr2 = abs(group["low"] - group["close"].shift(1))
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        group["atr"] = tr.rolling(14).mean()
        
        processed_dfs.append(group)
        
    return pd.concat(processed_dfs, ignore_index=True)

def run_simulation(df, min_range_pct=0.5, max_range_pct=3.0, vol_contraction_filter=False, use_atr_sl=False, target_mult=2.0, sl_mult=1.0):
    trades = []
    
    # Process symbol by symbol
    for symbol, group in df.groupby("symbol"):
        if symbol in ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX", "NSE:NIFTYBANK-INDEX"]:
            continue
            
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_rows = len(group)
        
        for i in range(20, n_rows - 1):
            row = group.iloc[i]
            
            is_nr7 = bool(row["nr7"])
            is_ib = bool(row["inside_bar"])
            
            if not (is_nr7 or is_ib):
                continue
                
            # Range size filter (yesterday's range must be in the sweet spot)
            range_pct = row["range_pct"]
            if range_pct < min_range_pct or range_pct > max_range_pct:
                continue
                
            # Volume contraction filter (setup day volume should be less than average to confirm quiet coiling)
            if vol_contraction_filter and row["vol_ratio"] >= 1.0:
                continue
                
            setup_high = row["high"]
            setup_low = row["low"]
            setup_close = row["close"]
            atr = row["atr"]
            
            risk = setup_high - setup_low
            if risk <= 0:
                continue
                
            # Entry Day (i + 1)
            entry_day_row = group.iloc[i + 1]
            entry_open = entry_day_row["open"]
            entry_high = entry_day_row["high"]
            entry_low = entry_day_row["low"]
            
            triggered = False
            position = None
            entry_price = 0.0
            
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
                
            # Stop Loss and Target placement
            if use_atr_sl:
                # Dynamic ATR-based SL instead of range boundaries
                trade_risk = atr * sl_mult
                sl_price = entry_price - trade_risk if position == "LONG" else entry_price + trade_risk
            else:
                trade_risk = risk * sl_mult
                sl_price = setup_low if position == "LONG" else setup_high
                
            tp_price = entry_price + target_mult * trade_risk if position == "LONG" else entry_price - target_mult * trade_risk
            
            # Exit holding period check (up to 5 days)
            exit_reason = None
            exit_price = 0.0
            end_idx = min(i + 1 + 5, n_rows)
            
            for j in range(i + 1, end_idx):
                sim_row = group.iloc[j]
                sim_high = sim_row["high"]
                sim_low = sim_row["low"]
                sim_close = sim_row["close"]
                
                if position == "LONG":
                    sl_hit = sim_low <= sl_price
                    tp_hit = sim_high >= tp_price
                    
                    if sl_hit and tp_hit:
                        exit_reason = "STOP_LOSS"
                        exit_price = sl_price
                        break
                    elif sl_hit:
                        exit_reason = "STOP_LOSS"
                        exit_price = min(sl_price, sim_row["open"])
                        break
                    elif tp_hit:
                        exit_reason = "TARGET"
                        exit_price = max(tp_price, sim_row["open"])
                        break
                else: # SHORT
                    sl_hit = sim_high >= sl_price
                    tp_hit = sim_low <= tp_price
                    
                    if sl_hit and tp_hit:
                        exit_reason = "STOP_LOSS"
                        exit_price = sl_price
                        break
                    elif sl_hit:
                        exit_reason = "STOP_LOSS"
                        exit_price = max(sl_price, sim_row["open"])
                        break
                    elif tp_hit:
                        exit_reason = "TARGET"
                        exit_price = min(tp_price, sim_row["open"])
                        break
                        
                if j == end_idx - 1:
                    exit_reason = "TIME_EXIT"
                    exit_price = sim_close
                    
            if position == "LONG":
                pnl = (exit_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - exit_price) / entry_price * 100
                
            trades.append(pnl)
            
    if not trades:
        return 0, 0.0, 0.0, 0.0
        
    trades_df = pd.Series(trades)
    total_trades = len(trades_df)
    win_rate = (trades_df > 0).sum() / total_trades * 100
    net_pnl = trades_df.sum()
    
    gross_profits = trades_df[trades_df > 0].sum()
    gross_losses = abs(trades_df[trades_df <= 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")
    
    return total_trades, win_rate, net_pnl, profit_factor

def main():
    df = load_data()
    df_indicators = calculate_indicators(df)
    
    print("Running parameter sweeps to optimize NR7 / Inside Bar breakout strategy...")
    print("="*100)
    print(f"{'Min Range %':<12}{'Max Range %':<12}{'Vol Contract':<13}{'Use ATR SL':<12}{'Target Mult':<12}{'Trades':<8}{'Win Rate':<12}{'Net PnL':<12}{'Profit Factor':<12}")
    print("="*100)
    
    # Run sweeps
    min_ranges = [0.5, 1.0]
    max_ranges = [2.5, 3.5]
    vol_contractions = [False, True]
    use_atr_sls = [False, True]
    target_mults = [1.5, 2.0]
    
    results = []
    
    for mr in min_ranges:
        for mxr in max_ranges:
            for vc in vol_contractions:
                for atr_sl in use_atr_sls:
                    for tm in target_mults:
                        t, wr, net_pnl, pf = run_simulation(
                            df_indicators, 
                            min_range_pct=mr, 
                            max_range_pct=mxr, 
                            vol_contraction_filter=vc, 
                            use_atr_sl=atr_sl, 
                            target_mult=tm
                        )
                        if t >= 10:
                            print(f"{mr:<12}{mxr:<12}{str(vc):<13}{str(atr_sl):<12}{tm:<12.1f}{t:<8}{wr:.2f}%     {net_pnl:+.2f}%   {pf:.2f}")
                            results.append({
                                "mr": mr, "mxr": mxr, "vc": vc, "atr_sl": atr_sl, "tm": tm,
                                "trades": t, "win_rate": wr, "net_pnl": net_pnl, "pf": pf
                            })
                            
    print("="*100)
    
    # Find best parameter set
    if results:
        best_pnl = max(results, key=lambda x: x["net_pnl"])
        best_pf = max(results, key=lambda x: x["pf"])
        
        print("\n🏆 OPTIMIZATION SUMMARY:")
        print(f"• Best Net PnL Configuration: Min Range {best_pnl['mr']}%, Max Range {best_pnl['mxr']}%, Vol Contraction={best_pnl['vc']}, Use ATR SL={best_pnl['atr_sl']}, Target Mult={best_pnl['tm']}")
        print(f"  Trades: {best_pnl['trades']} | Win Rate: {best_pnl['win_rate']:.2f}% | Net PnL: {best_pnl['net_pnl']:+.2f}% | Profit Factor: {best_pnl['pf']:.2f}")
        
        print(f"\n• Best Profit Factor Configuration: Min Range {best_pf['mr']}%, Max Range {best_pf['mxr']}%, Vol Contraction={best_pf['vc']}, Use ATR SL={best_pf['atr_sl']}, Target Mult={best_pf['tm']}")
        print(f"  Trades: {best_pf['trades']} | Win Rate: {best_pf['win_rate']:.2f}% | Net PnL: {best_pf['net_pnl']:+.2f}% | Profit Factor: {best_pf['pf']:.2f}")

if __name__ == "__main__":
    main()
