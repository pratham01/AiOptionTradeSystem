import os
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime

def run_backtest_on_file(file_path, vol_ratio_threshold=1.5, compression_quantile=0.20, target_multiplier=2.0, sl_multiplier=1.2):
    df = pd.read_csv(file_path)
    if df.empty or len(df) < 50:
        return []

    # Sort and parse times
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Extract date and time components
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    # Calculate indicators
    df["close_ma"] = df["close"].rolling(20).mean()
    df["close_std"] = df["close"].rolling(20).std()
    df["upper_band"] = df["close_ma"] + 2 * df["close_std"]
    df["lower_band"] = df["close_ma"] - 2 * df["close_std"]
    df["bb_width"] = (df["upper_band"] - df["lower_band"]) / df["close_ma"] * 100
    
    # Volume indicator
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, 1)
    
    # ATR (simple range representation)
    df["tr"] = np.maximum(df["high"] - df["low"], 
                          np.maximum(abs(df["high"] - df["close"].shift(1)), 
                                     abs(df["low"] - df["close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()

    # Cumulative Daily VWAP
    df["pv"] = df["close"] * df["volume"]
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["cum_pv"] = df.groupby("date")["pv"].cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"].replace(0, 1)

    # Compression indicator
    df["bbw_limit"] = df["bb_width"].rolling(100).quantile(compression_quantile)
    df["compression"] = df["bb_width"] <= df["bbw_limit"]

    trades = []
    position = None
    entry_price = 0.0
    entry_time = None
    target_price = 0.0
    sl_price = 0.0
    
    symbol = os.path.basename(file_path).replace("_5min_historical.csv", "")

    for i in range(25, len(df)):
        row = df.iloc[i]
        curr_time = row["time"]
        curr_price = row["close"]
        curr_atr = row["atr"] if not pd.isna(row["atr"]) else (row["high"] - row["low"])
        
        # EOD Square-off check
        is_eod = curr_time >= datetime.strptime("15:15:00", "%H:%M:%S").time()

        if position is not None:
            # Check Exit Conditions
            if position == "LONG":
                if curr_price >= target_price:
                    pnl = (target_price - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "LONG", "pnl": pnl, "exit_reason": "TARGET"})
                    position = None
                elif curr_price <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "LONG", "pnl": pnl, "exit_reason": "STOP_LOSS"})
                    position = None
                elif is_eod:
                    pnl = (curr_price - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "LONG", "pnl": pnl, "exit_reason": "EOD"})
                    position = None
            elif position == "SHORT":
                if curr_price <= target_price:
                    pnl = (entry_price - target_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "SHORT", "pnl": pnl, "exit_reason": "TARGET"})
                    position = None
                elif curr_price >= sl_price:
                    pnl = (entry_price - sl_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "SHORT", "pnl": pnl, "exit_reason": "STOP_LOSS"})
                    position = None
                elif is_eod:
                    pnl = (entry_price - curr_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "SHORT", "pnl": pnl, "exit_reason": "EOD"})
                    position = None
        else:
            if curr_time >= datetime.strptime("15:00:00", "%H:%M:%S").time():
                continue
                
            # Compression check
            has_compression = df.iloc[i-5:i]["compression"].any()
            if not has_compression:
                continue

            # Long breakout trigger
            is_long_breakout = (curr_price > row["upper_band"]) and (row["vol_ratio"] >= vol_ratio_threshold) and (curr_price > row["vwap"])
            # Short breakout trigger
            is_short_breakout = (curr_price < row["lower_band"]) and (row["vol_ratio"] >= vol_ratio_threshold) and (curr_price < row["vwap"])

            if is_long_breakout:
                position = "LONG"
                entry_price = curr_price
                entry_time = row["timestamp"]
                target_price = entry_price + target_multiplier * curr_atr
                sl_price = entry_price - sl_multiplier * curr_atr
            elif is_short_breakout:
                position = "SHORT"
                entry_price = curr_price
                entry_time = row["timestamp"]
                target_price = entry_price - target_multiplier * curr_atr
                sl_price = entry_price + sl_multiplier * curr_atr

    return trades

def evaluate_params(files, vol_ratio, comp_q, target_mult, sl_mult):
    all_trades = []
    for f in files:
        all_trades.extend(run_backtest_on_file(f, vol_ratio_threshold=vol_ratio, compression_quantile=comp_q, target_multiplier=target_mult, sl_multiplier=sl_mult))
    
    if not all_trades:
        return 0, 0.0, 0.0, 0.0
        
    df = pd.DataFrame(all_trades)
    total = len(df)
    win_rate = (df["pnl"] > 0).sum() / total * 100
    net_pnl = df["pnl"].sum()
    
    gross_profits = df[df["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(df[df["pnl"] <= 0]["pnl"].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")
    
    return total, win_rate, net_pnl, profit_factor

def main():
    historical_dir = "data/fo_historical"
    file_pattern = os.path.join(historical_dir, "*_5min_historical.csv")
    files = glob(file_pattern)
    
    print(f"Loaded {len(files)} files. Running optimization sweep...")
    print("="*80)
    print(f"{'Vol Ratio':<12}{'BBW Quantile':<15}{'Target Mult':<12}{'SL Mult':<10}{'Trades':<8}{'Win Rate':<12}{'Net PnL':<12}{'Profit Factor':<12}")
    print("="*80)
    
    # Grid search space
    vol_ratios = [1.5, 2.0, 2.5]
    comp_quantiles = [0.15, 0.20, 0.25]
    target_mults = [1.5, 2.0]
    sl_mults = [1.0, 1.2]
    
    best_pf = 0.0
    best_params = {}
    
    for vr in vol_ratios:
        for cq in comp_quantiles:
            for tm in target_mults:
                for sm in sl_mults:
                    t, wr, net_pnl, pf = evaluate_params(files, vr, cq, tm, sm)
                    if t >= 10: # Minimum sample size
                        wr_str = f"{wr:.2f}%"
                        pnl_str = f"{net_pnl:+.2f}%"
                        print(f"{vr:<12}{cq:<15.2f}{tm:<12.1f}{sm:<10.1f}{t:<8}{wr_str:<12}{pnl_str:<12}{pf:<12.2f}")
                        if pf > best_pf:
                            best_pf = pf
                            best_params = {"vol_ratio": vr, "comp_q": cq, "target_mult": tm, "sl_mult": sm, "trades": t, "win_rate": wr, "net_pnl": net_pnl}
                            
    print("="*80)
    if best_params:
        print("\n🏆 BEST PARAMETER SET FOUND:")
        print(f"• Volume Ratio Threshold : {best_params['vol_ratio']}")
        print(f"• BBW Compression Quantile: {best_params['comp_q']}")
        print(f"• Target Multiplier      : {best_params['target_mult']}")
        print(f"• Stop Loss Multiplier   : {best_params['sl_mult']}")
        print(f"• Total Trades           : {best_params['trades']}")
        print(f"• Win Rate               : {best_params['win_rate']:.2f}%")
        print(f"• Net Cumulative PnL     : {best_params['net_pnl']:+.2f}%")
        print(f"• Best Profit Factor     : {best_pf:.2f}")
    else:
        print("No configurations generated enough trades.")

if __name__ == "__main__":
    main()
