import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import text
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

def get_daily_candles_from_15m(df_15m):
    """Aggregates 15-minute candles into daily candles."""
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

def add_supertrend(df, period=10, multiplier=3.0):
    """Computes daily Supertrend direction (+1 for bullish, -1 for bearish)."""
    def calc_st(grp):
        high, low, close = grp['high'], grp['low'], grp['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        direction = pd.Series(1, index=grp.index)
        for i in range(1, len(grp)):
            if close.iloc[i] > upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
        grp["st_dir"] = direction
        return grp
    return df.groupby("symbol", group_keys=False).apply(calc_st)

def compute_all_daily_indicators(df):
    """Calculates ATR%, BB Width, RSI, Volume MA, 52-Week High, and Supertrend on daily data."""
    df = df.sort_values(["symbol", "timestamp"]).copy()
    
    # ATR-14
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["tr"] = np.maximum(df["high"] - df["low"], 
                          np.maximum((df["high"] - df["prev_close"]).abs(), 
                                     (df["low"] - df["prev_close"]).abs()))
    df["atr14"] = df.groupby("symbol")["tr"].transform(lambda x: x.rolling(14, min_periods=5).mean())
    df["atr_pct"] = (df["atr14"] / df["close"]) * 100
    
    # BB Width
    df["close_ma"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["close_std"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20, min_periods=5).std())
    df["bb_width"] = (4 * df["close_std"]) / df["close_ma"].replace(0, 1) * 100
    
    # RSI-14
    df["delta"] = df.groupby("symbol")["close"].diff()
    df["gain"] = df["delta"].clip(lower=0)
    df["loss"] = -df["delta"].clip(upper=0)
    df["avg_gain"] = df.groupby("symbol")["gain"].transform(lambda x: x.rolling(14, min_periods=5).mean())
    df["avg_loss"] = df.groupby("symbol")["loss"].transform(lambda x: x.rolling(14, min_periods=5).mean())
    df["rs"] = df["avg_gain"] / df["avg_loss"].replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + df["rs"])
    
    # Volume MA
    df["vol_ma"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, 1)
    
    # 52w High
    df["w52_high"] = df.groupby("symbol")["high"].transform(lambda x: x.rolling(252, min_periods=100).max())
    
    # Add Supertrend
    df = add_supertrend(df)
    
    return df

def run_scan(df_daily, scan_date):
    """Scans the daily data as of scan_date to find BTST and Intraday stocks."""
    print(f"\n🔍 Scanning F&O stocks on Date: {scan_date.strftime('%Y-%m-%d')}...")
    
    # Get all unique symbols
    df = df_daily.sort_values(["symbol", "timestamp"]).copy()
    all_dates = sorted(df["timestamp"].dt.date.unique())
    
    if scan_date not in all_dates:
        print(f"❌ Scan date {scan_date} is not a valid trading day in the database.")
        # Find closest date before scan_date
        past_dates = [d for d in all_dates if d <= scan_date]
        if not past_dates:
            print("No valid historical data before the specified scan date.")
            return
        scan_date = past_dates[-1]
        print(f"⚠️ Adjusted scan date to latest available: {scan_date.strftime('%Y-%m-%d')}")
        
    df_curr = df[df["timestamp"].dt.date == scan_date].copy()
    
    btst_picks = []
    intraday_watchlist = []
    
    for _, row in df_curr.iterrows():
        sym = row["symbol"]
        clean_sym = sym.replace('NSE:', '').replace('-EQ', '')
        
        # Get historical slice for compression quantiles
        sym_hist = df[(df["symbol"] == sym) & (df["timestamp"].dt.date <= scan_date)]
        if len(sym_hist) < 20:
            continue
            
        bbw_limit = sym_hist["bb_width"].quantile(0.25)
        is_compressed = row["bb_width"] <= bbw_limit
        
        change_pct = ((row["close"] - row["prev_close"]) / row["prev_close"] * 100) if not pd.isna(row["prev_close"]) else 0.0
        high_range = row["high"] - row["low"]
        close_high_ratio = (row["close"] - row["low"]) / high_range if high_range > 0 else 0
        
        # 1. BTST Conditions
        is_squeeze_btst = is_compressed and row["atr_pct"] >= 1.4 and change_pct >= 1.5 and close_high_ratio >= 0.92 and row["volume"] >= 1.5 * row["vol_ma"]
        is_52w_btst = row["close"] >= row["w52_high"] * 0.985 and row["volume"] >= 1.5 * row["vol_ma"] and change_pct > 0
        
        # Supertrend flip
        st_flipped = False
        if len(sym_hist) >= 2:
            st_prev = sym_hist.iloc[-2]["st_dir"]
            if st_prev == -1 and row["st_dir"] == 1:
                st_flipped = True
        
        is_st_btst = st_flipped and change_pct >= 1.0 and row["volume"] >= 1.2 * row["vol_ma"]
        
        if is_squeeze_btst or is_52w_btst or is_st_btst:
            reasons = []
            if is_squeeze_btst: reasons.append("Squeeze Breakout")
            if is_52w_btst: reasons.append("52W High Close")
            if is_st_btst: reasons.append("Supertrend Flip")
            
            btst_picks.append({
                "Symbol": clean_sym,
                "Close": f"₹{row['close']:.2f}",
                "Change%": f"{change_pct:+.1f}%",
                "Vol Surge": f"{row['volume']/row['vol_ma']:.1f}x",
                "Close-High%": f"{close_high_ratio*100:.0f}%",
                "Reason": " + ".join(reasons)
            })
            
        # 2. Intraday Watchlist Conditions
        is_squeeze_watch = is_compressed and row["atr_pct"] >= 1.4
        is_52w_watch = row["close"] >= row["w52_high"] * 0.985
        
        if is_squeeze_watch or is_52w_watch:
            reasons_w = []
            if is_squeeze_watch: reasons_w.append("Daily Compression")
            if is_52w_watch: reasons_w.append("Near 52W High")
            
            intraday_watchlist.append({
                "Symbol": clean_sym,
                "Close": f"₹{row['close']:.2f}",
                "ATR%": f"{row['atr_pct']:.1f}%",
                "BBW": f"{row['bb_width']:.1f}%",
                "RSI": f"{row['rsi']:.0f}" if not pd.isna(row['rsi']) else "N/A",
                "Watch Reason": " + ".join(reasons_w)
            })
            
    # Print results
    print("\n" + "="*80)
    print(f"🚀 BTST OVERNIGHT BUYING CANDIDATES (Entry on {scan_date.strftime('%Y-%m-%d')} EOD)")
    print("="*80)
    if not btst_picks:
        print("No candidates matched BTST criteria today.")
    else:
        df_btst = pd.DataFrame(btst_picks)
        print(df_btst.to_string(index=False))
        
    print("\n" + "="*80)
    print(f"📅 INTRADAY BREAKOUT WATCHLIST FOR NEXT SESSION")
    print("="*80)
    if not intraday_watchlist:
        print("No candidates matched Intraday Watchlist criteria today.")
    else:
        df_intra = pd.DataFrame(intraday_watchlist)
        print(df_intra.to_string(index=False))
    print("\n" + "="*80 + "\n")

def backtest_intraday(df_15m, watchlists, vol_ratio_threshold=2.0, target_mult=2.5, sl_mult=1.0):
    """Simulates 15m breakout trades for watchlisted symbols."""
    test_start = date(2026, 5, 25)
    test_end = date(2026, 5, 29)
    df_test = df_15m[(df_15m["date"] >= test_start) & (df_15m["date"] <= test_end)].copy()
    
    if df_test.empty:
        return []
        
    wl_map = {(d, s): True for d, symbols in watchlists.items() for s in symbols}
    df_test["on_watchlist"] = df_test.apply(lambda r: wl_map.get((r["date"], r["symbol"]), False), axis=1)
    df_wl = df_test[df_test["on_watchlist"]].copy()
    
    if df_wl.empty:
        return []
        
    trades = []
    for (symbol, dt), grp in df_wl.groupby(["symbol", "date"]):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        position = None
        entry_price = 0.0
        target_price = 0.0
        sl_price = 0.0
        
        grp["has_compression"] = grp["compression"].rolling(5).max().fillna(0).astype(bool)
        
        for i in range(len(grp)):
            row = grp.iloc[i]
            curr_time = row["timestamp"].time()
            curr_price = row["close"]
            curr_atr = row["atr"] if not pd.isna(row["atr"]) else (row["high"] - row["low"])
            is_eod = curr_time >= datetime.strptime("15:15:00", "%H:%M:%S").time()
            
            if position is not None:
                if position == "LONG":
                    if curr_price >= target_price:
                        trades.append({"symbol": symbol, "date": dt, "pnl": (target_price - entry_price)/entry_price*100, "exit": "TARGET"})
                        position = None
                    elif curr_price <= sl_price:
                        trades.append({"symbol": symbol, "date": dt, "pnl": (sl_price - entry_price)/entry_price*100, "exit": "SL"})
                        position = None
                    elif is_eod:
                        trades.append({"symbol": symbol, "date": dt, "pnl": (curr_price - entry_price)/entry_price*100, "exit": "EOD"})
                        position = None
                elif position == "SHORT":
                    if curr_price <= target_price:
                        trades.append({"symbol": symbol, "date": dt, "pnl": (entry_price - target_price)/entry_price*100, "exit": "TARGET"})
                        position = None
                    elif curr_price >= sl_price:
                        trades.append({"symbol": symbol, "date": dt, "pnl": (entry_price - sl_price)/entry_price*100, "exit": "SL"})
                        position = None
                    elif is_eod:
                        trades.append({"symbol": symbol, "date": dt, "pnl": (entry_price - curr_price)/entry_price*100, "exit": "EOD"})
                        position = None
            else:
                if curr_time >= datetime.strptime("15:00:00", "%H:%M:%S").time():
                    continue
                if not row["has_compression"]:
                    continue
                    
                is_long_breakout = (curr_price > row["upper_band"]) and (row["vol_ratio"] >= vol_ratio_threshold) and (curr_price > row["vwap"])
                is_short_breakout = (curr_price < row["lower_band"]) and (row["vol_ratio"] >= vol_ratio_threshold) and (curr_price < row["vwap"])
                
                if is_long_breakout:
                    position = "LONG"
                    entry_price = curr_price
                    target_price = entry_price + target_mult * curr_atr
                    sl_price = entry_price - sl_mult * curr_atr
                elif is_short_breakout:
                    position = "SHORT"
                    entry_price = curr_price
                    target_price = entry_price - target_mult * curr_atr
                    sl_price = entry_price + sl_mult * curr_atr
                    
    return trades

def backtest_btst(df_daily, watchlists, exit_rule='NEXT_OPEN'):
    """Simulates BTST trades for watchlisted symbols."""
    trades = []
    df = df_daily.sort_values(["symbol", "timestamp"]).copy()
    
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(grp) - 1):
            row_curr = grp.iloc[i]
            curr_date = row_curr["timestamp"].date()
            
            if curr_date in watchlists and symbol in watchlists[curr_date]:
                entry_price = row_curr["close"]
                row_next = grp.iloc[i+1]
                next_open = row_next["open"]
                next_high = row_next["high"]
                next_low = row_next["low"]
                next_close = row_next["close"]
                
                if exit_rule == 'NEXT_OPEN':
                    pnl = (next_open - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "date": curr_date, "pnl": pnl, "exit": "OPEN"})
                elif exit_rule == 'NEXT_CLOSE':
                    pnl = (next_close - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "date": curr_date, "pnl": pnl, "exit": "CLOSE"})
                elif exit_rule.startswith('NEXT_HIGH_LOW'):
                    parts = exit_rule.split('_')
                    t_val = float(parts[3][1:])
                    s_val = float(parts[4][1:])
                    target = entry_price * (1 + t_val / 100)
                    sl = entry_price * (1 - s_val / 100)
                    
                    if next_open >= target or next_open <= sl:
                        pnl = (next_open - entry_price) / entry_price * 100
                        trades.append({"symbol": symbol, "date": curr_date, "pnl": pnl, "exit": "GAP"})
                    elif next_high >= target:
                        trades.append({"symbol": symbol, "date": curr_date, "pnl": t_val, "exit": "TARGET"})
                    elif next_low <= sl:
                        trades.append({"symbol": symbol, "date": curr_date, "pnl": -s_val, "exit": "SL"})
                    else:
                        pnl = (next_close - entry_price) / entry_price * 100
                        trades.append({"symbol": symbol, "date": curr_date, "pnl": pnl, "exit": "CLOSE"})
    return trades

def print_backtest_stats(name, trades):
    if not trades:
        print(f"• {name}: No trades triggered.")
        return
    df = pd.DataFrame(trades)
    total = len(df)
    win_rate = (df["pnl"] > 0).sum() / total * 100
    net_pnl = df["pnl"].sum()
    avg_pnl = df["pnl"].mean()
    
    gp = df[df["pnl"] > 0]["pnl"].sum()
    gl = abs(df[df["pnl"] <= 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    
    pf_str = f"{pf:.2f}" if pf != float("inf") else "Infinite (No Losses)"
    print(f"• {name:<30} | Trades: {total:2d} | WinRate: {win_rate:5.1f}% | NetPnL: {net_pnl:+6.2f}% | Avg: {avg_pnl:+5.2f}% | PF: {pf_str}")

def run_historical_backtest(df_combined_daily, df_15m):
    """Executes a full backtest simulation over the test week."""
    print("\n" + "="*80)
    print("🔬 RUNNING HISTORICAL BACKTEST SIMULATION (MAY 25 – MAY 29, 2026)")
    print("="*80)
    
    test_dates = [date(2026, 5, 25), date(2026, 5, 26), date(2026, 5, 27), date(2026, 5, 29)]
    
    # Generate watchlists for each day
    squeeze_wl_btst = {}
    squeeze_wl_intra = {}
    high52w_wl = {}
    st_wl = {}
    
    df = df_combined_daily.sort_values(["symbol", "timestamp"]).copy()
    all_dates = sorted(df["timestamp"].dt.date.unique())
    
    for t_date in test_dates:
        idx = all_dates.index(t_date)
        prev_date = all_dates[idx-1]
        df_prev = df[df["timestamp"].dt.date == prev_date].copy()
        
        sq_btst = set()
        sq_intra = set()
        h52 = set()
        st_flip = set()
        
        for _, row in df_prev.iterrows():
            sym = row["symbol"]
            
            # Squeeze calculations
            sym_hist = df[(df["symbol"] == sym) & (df["timestamp"].dt.date <= prev_date)]
            if len(sym_hist) < 20: continue
            
            bbw_limit = sym_hist["bb_width"].quantile(0.25)
            is_compressed = row["bb_width"] <= bbw_limit
            change_pct = ((row["close"] - row["prev_close"])/row["prev_close"]*100) if not pd.isna(row["prev_close"]) else 0.0
            high_range = row["high"] - row["low"]
            close_high_ratio = (row["close"] - row["low"])/high_range if high_range > 0 else 0
            
            # Watchlist 1: Daily Squeeze Watchlist
            if is_compressed and row["atr_pct"] >= 1.4:
                sq_intra.add(sym)
                if change_pct >= 1.5 and close_high_ratio >= 0.92 and row["volume"] >= 1.5 * row["vol_ma"]:
                    sq_btst.add(sym)
                    
            # Watchlist 2: 52w High Breakout Watchlist
            if row["close"] >= row["w52_high"] * 0.985:
                h52.add(sym)
                
            # Watchlist 3: Supertrend Flip
            if len(sym_hist) >= 2:
                if sym_hist.iloc[-2]["st_dir"] == -1 and row["st_dir"] == 1:
                    st_flip.add(sym)
                    
        squeeze_wl_btst[t_date] = sq_btst
        squeeze_wl_intra[t_date] = sq_intra
        high52w_wl[t_date] = h52
        st_wl[t_date] = st_flip
        
    print("Executing BTST backtests...")
    trades_sq_btst = backtest_btst(df_combined_daily, squeeze_wl_btst, 'NEXT_OPEN')
    trades_52w_btst = backtest_btst(df_combined_daily, high52w_wl, 'NEXT_HIGH_LOW_T1_S1')
    trades_st_btst = backtest_btst(df_combined_daily, st_wl, 'NEXT_CLOSE')
    
    print("Executing Intraday 15m breakout backtests...")
    trades_sq_intra = backtest_intraday(df_15m, squeeze_wl_intra, vol_ratio_threshold=2.0, target_mult=2.5, sl_mult=1.0)
    trades_52w_intra = backtest_intraday(df_15m, high52w_wl, vol_ratio_threshold=2.0, target_mult=2.5, sl_mult=1.0)
    
    print("\n" + "-"*50)
    print("📊 BACKTEST PERFORMANCE SUMMARY")
    print("-"*50)
    print_backtest_stats("BTST Vol Squeeze (Exit Open)", trades_sq_btst)
    print_backtest_stats("BTST 52W High Close (T1/SL1)", trades_52w_btst)
    print_backtest_stats("BTST Supertrend Flip (Exit Close)", trades_st_btst)
    print_backtest_stats("Intraday BB Squeeze (15m BO)", trades_sq_intra)
    print_backtest_stats("Intraday 52W High (15m BO)", trades_52w_intra)
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="F&O Stock Selector and Backtester CLI")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtests for the test week (May 25-29, 2026)")
    parser.add_argument("--date", type=str, help="Specific scan date (YYYY-MM-DD), default is the latest available in DB")
    args = parser.parse_args()
    
    engine = get_engine()
    
    # Load raw data
    with engine.connect() as conn:
        df_daily_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_daily"), conn)
        df_15m_raw = pd.read_sql(text("SELECT symbol, timestamp, open, high, low, close, volume FROM ohlcv_15m WHERE timestamp >= '2026-05-05 00:00:00'"), conn)
        
    df_15m_raw["timestamp"] = pd.to_datetime(df_15m_raw["timestamp"], format="mixed")
    df_daily_raw["timestamp"] = pd.to_datetime(df_daily_raw["timestamp"], format="mixed")
    
    # Process and combine candles
    df_15m_daily = get_daily_candles_from_15m(df_15m_raw)
    df_daily_raw["date"] = df_daily_raw["timestamp"].dt.date
    df_15m_daily["date"] = df_15m_daily["timestamp"].dt.date
    
    dates_15m = set(df_15m_daily["date"])
    df_daily_filtered = df_daily_raw[~df_daily_raw["date"].isin(dates_15m)].copy()
    
    df_combined_daily = pd.concat([df_daily_filtered, df_15m_daily], ignore_index=True)
    df_combined_daily = df_combined_daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df_combined_daily = compute_all_daily_indicators(df_combined_daily)
    
    if args.backtest:
        # Precompute 15m indicators for backtesting
        df_15m = df_15m_raw.sort_values(["symbol", "timestamp"]).copy()
        df_15m["close_ma"] = df_15m.groupby("symbol")["close"].transform(lambda x: x.rolling(20).mean())
        df_15m["close_std"] = df_15m.groupby("symbol")["close"].transform(lambda x: x.rolling(20).std())
        df_15m["upper_band"] = df_15m["close_ma"] + 2 * df_15m["close_std"]
        df_15m["lower_band"] = df_15m["close_ma"] - 2 * df_15m["close_std"]
        df_15m["bb_width"] = (df_15m["upper_band"] - df_15m["lower_band"]) / df_15m["close_ma"] * 100
        df_15m["vol_ma"] = df_15m.groupby("symbol")["volume"].transform(lambda x: x.rolling(20).mean())
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_ma"].replace(0, 1)

        df_15m["prev_close"] = df_15m.groupby("symbol")["close"].shift(1)
        df_15m["tr"] = np.maximum(df_15m["high"] - df_15m["low"], 
                                  np.maximum((df_15m["high"] - df_15m["prev_close"]).abs(), 
                                             (df_15m["low"] - df_15m["prev_close"]).abs()))
        df_15m["atr"] = df_15m.groupby("symbol")["tr"].transform(lambda x: x.rolling(14).mean())

        df_15m["date"] = df_15m["timestamp"].dt.date
        df_15m["pv"] = df_15m["close"] * df_15m["volume"]
        df_15m["cum_vol"] = df_15m.groupby(["symbol", "date"])["volume"].cumsum()
        df_15m["cum_pv"] = df_15m.groupby(["symbol", "date"])["pv"].cumsum()
        df_15m["vwap"] = df_15m["cum_pv"] / df_15m["cum_vol"].replace(0, 1)

        df_15m["bbw_limit"] = df_15m.groupby("symbol")["bb_width"].transform(lambda x: x.rolling(100).quantile(0.25))
        df_15m["compression"] = df_15m["bb_width"] <= df_15m["bbw_limit"]
        
        run_historical_backtest(df_combined_daily, df_15m)
    else:
        # Determine scan date
        if args.date:
            try:
                scan_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            except ValueError:
                print("❌ Invalid date format. Use YYYY-MM-DD.")
                sys.exit(1)
        else:
            scan_date = df_combined_daily["timestamp"].max().date()
            
        run_scan(df_combined_daily, scan_date)

if __name__ == "__main__":
    main()
