import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sqlalchemy import text
from trade_system.infrastructure.database.connection import get_engine

def get_daily_candles_from_15m(df_15m):
    """Dynamically aggregates 15-minute candles into daily candles."""
    df = df_15m.copy()
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date").agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily

def compute_daily_indicators(daily_df):
    """Computes daily ATR %, Bollinger Band Width, and Volume MA."""
    if daily_df.empty:
        return daily_df

    df = daily_df.copy()
    # Daily ATR (10-day lookback, min 5 periods)
    df["tr"] = np.maximum(df["high"] - df["low"], 
                          np.maximum(abs(df["high"] - df["close"].shift(1)), 
                                     abs(df["low"] - df["close"].shift(1))))
    df["atr"] = df["tr"].rolling(10, min_periods=5).mean()
    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    # Daily BB Width (10-day lookback, min 5 periods)
    df["close_ma"] = df["close"].rolling(10, min_periods=5).mean()
    df["close_std"] = df["close"].rolling(10, min_periods=5).std()
    df["bb_width"] = (4 * df["close_std"]) / df["close_ma"].replace(0, 1) * 100
    df["bbw_limit"] = df["bb_width"].rolling(10, min_periods=5).quantile(0.25)
    df["compression"] = df["bb_width"] <= df["bbw_limit"]

    # Daily Volume MA (10-day lookback, min 5 periods)
    df["vol_ma"] = df["volume"].rolling(10, min_periods=5).mean()
    
    return df

def run_intraday_on_watchlist(conn, symbol, watchlist_dates, start_test_date="2026-05-25", end_test_date="2026-05-29"):
    # Load 15m data with warm-up lookback
    query = text("""
        SELECT timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE symbol = :symbol 
          AND timestamp >= '2026-05-08 00:00:00'
        ORDER BY timestamp ASC
    """)
    df = pd.read_sql(query, conn, params={"symbol": symbol})
    
    if df.empty or len(df) < 120:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    # Calculate indicators
    df["close_ma"] = df["close"].rolling(20).mean()
    df["close_std"] = df["close"].rolling(20).std()
    df["upper_band"] = df["close_ma"] + 2 * df["close_std"]
    df["lower_band"] = df["close_ma"] - 2 * df["close_std"]
    df["bb_width"] = (df["upper_band"] - df["lower_band"]) / df["close_ma"] * 100
    
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, 1)
    
    df["tr"] = np.maximum(df["high"] - df["low"], 
                          np.maximum(abs(df["high"] - df["close"].shift(1)), 
                                     abs(df["low"] - df["close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()

    df["pv"] = df["close"] * df["volume"]
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["cum_pv"] = df.groupby("date")["pv"].cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"].replace(0, 1)

    df["bbw_limit"] = df["bb_width"].rolling(100).quantile(0.25)
    df["compression"] = df["bb_width"] <= df["bbw_limit"]

    trades = []
    position = None
    entry_price = 0.0
    entry_time = None
    target_price = 0.0
    sl_price = 0.0
    
    start_dt = datetime.strptime(start_test_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_test_date, "%Y-%m-%d").date()

    for i in range(100, len(df)):
        row = df.iloc[i]
        curr_date = row["date"]
        curr_time = row["time"]
        curr_price = row["close"]
        curr_atr = row["atr"] if not pd.isna(row["atr"]) else (row["high"] - row["low"])
        
        # EOD Square-off check
        is_eod = curr_time >= datetime.strptime("15:15:00", "%H:%M:%S").time()

        if position is not None:
            if position == "LONG":
                if curr_price >= target_price:
                    pnl = (target_price - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "LONG", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": target_price, "pnl": pnl, "exit_reason": "TARGET"})
                    position = None
                elif curr_price <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "LONG", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": sl_price, "pnl": pnl, "exit_reason": "STOP_LOSS"})
                    position = None
                elif is_eod:
                    pnl = (curr_price - entry_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "LONG", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": curr_price, "pnl": pnl, "exit_reason": "EOD"})
                    position = None
            elif position == "SHORT":
                if curr_price <= target_price:
                    pnl = (entry_price - target_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "SHORT", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": target_price, "pnl": pnl, "exit_reason": "TARGET"})
                    position = None
                elif curr_price >= sl_price:
                    pnl = (entry_price - sl_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "SHORT", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": sl_price, "pnl": pnl, "exit_reason": "STOP_LOSS"})
                    position = None
                elif is_eod:
                    pnl = (entry_price - curr_price) / entry_price * 100
                    trades.append({"symbol": symbol, "type": "SHORT", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": curr_price, "pnl": pnl, "exit_reason": "EOD"})
                    position = None
        else:
            # Check if this symbol is on the watchlist for THIS SPECIFIC date
            if curr_date not in watchlist_dates:
                continue
            if curr_date < start_dt or curr_date > end_dt:
                continue
            if curr_time >= datetime.strptime("15:00:00", "%H:%M:%S").time():
                continue
                
            # Compression check
            has_compression = df.iloc[i-5:i]["compression"].any()
            if not has_compression:
                continue

            # Breakout rules (Optimized: Vol Ratio >= 2.5)
            is_long_breakout = (curr_price > row["upper_band"]) and (row["vol_ratio"] >= 2.5) and (curr_price > row["vwap"])
            is_short_breakout = (curr_price < row["lower_band"]) and (row["vol_ratio"] >= 2.5) and (curr_price < row["vwap"])

            if is_long_breakout:
                position = "LONG"
                entry_price = curr_price
                entry_time = row["timestamp"]
                target_price = entry_price + 2.0 * curr_atr
                sl_price = entry_price - 1.2 * curr_atr
            elif is_short_breakout:
                position = "SHORT"
                entry_price = curr_price
                entry_time = row["timestamp"]
                target_price = entry_price - 2.0 * curr_atr
                sl_price = entry_price + 1.2 * curr_atr

    return trades

def main():
    engine = get_engine()
    
    # 1. Fetch unique symbols
    with engine.connect() as conn:
        symbols_query = text("""
            SELECT DISTINCT symbol 
            FROM ohlcv_15m 
            WHERE timestamp >= '2026-05-08 00:00:00'
        """)
        symbols_df = pd.read_sql(symbols_query, conn)
        symbols = symbols_df["symbol"].tolist()
        
    print(f"Aggregating 15m candles to build Daily watchlists for {len(symbols)} symbols...")
    
    intraday_watchlists = {} # date -> set of symbols
    btst_candidates = [] # list of dicts representing overnight trades
    
    # Process daily candles and build filters
    with engine.connect() as conn:
        for sym in symbols:
            # Load 15m
            query = text("""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_15m 
                WHERE symbol = :symbol 
                  AND timestamp >= '2026-05-05 00:00:00'
                ORDER BY timestamp ASC
            """)
            df_15m = pd.read_sql(query, conn, params={"symbol": sym})
            if df_15m.empty or len(df_15m) < 80:
                continue
                
            df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
            daily_df = get_daily_candles_from_15m(df_15m)
            daily_df = compute_daily_indicators(daily_df)
            
            # Run watchlist filters
            for idx in range(1, len(daily_df)):
                row_prev = daily_df.iloc[idx-1]
                row_curr = daily_df.iloc[idx]
                curr_date = row_curr["date"]
                
                # A. INTRADAY WATCHLIST FILTER (Applied at start of day)
                # Volatility: ATR% >= 1.4%
                # Daily BB Squeeze: compression == True on previous day
                if row_prev["atr_pct"] >= 1.4 and row_prev["compression"] and row_prev["vol_ma"] > 10000:
                    if curr_date not in intraday_watchlists:
                        intraday_watchlists[curr_date] = set()
                    intraday_watchlists[curr_date].add(sym)
                    
                # B. BTST OVERNIGHT FILTER (Applied at 3:15 PM on current day)
                # Daily change >= 1.5%
                # Close-to-High ratio >= 92% (buyers dominate EOD)
                # Volume surge >= 1.8x the 20-day Average Daily Volume
                change_pct = ((row_curr["close"] - row_prev["close"]) / row_prev["close"]) * 100
                high_range = row_curr["high"] - row_curr["low"]
                close_high_ratio = (row_curr["close"] - row_curr["low"]) / high_range if high_range > 0 else 0
                
                is_btst_breakout = (change_pct >= 1.5) and (close_high_ratio >= 0.92) and (row_curr["volume"] >= 1.8 * row_curr["vol_ma"])
                
                # We limit entries to the test week: May 25 to May 29, 2026
                if is_btst_breakout and (date(2026, 5, 25) <= curr_date <= date(2026, 5, 29)):
                    # Get next day open price
                    if idx + 1 < len(daily_df):
                        next_day = daily_df.iloc[idx+1]
                        next_open = next_day["open"]
                        pnl = (next_open - row_curr["close"]) / row_curr["close"] * 100
                        btst_candidates.append({
                            "symbol": sym,
                            "date": curr_date,
                            "entry_price": row_curr["close"],
                            "exit_price": next_open,
                            "pnl": pnl,
                            "change_pct": change_pct,
                            "vol_surge": row_curr["volume"] / row_curr["vol_ma"]
                        })

    print(f"Dynamic watchlists constructed. Backtesting Intraday & BTST trades...")
    
    # 2. Execute Intraday Backtest ONLY on watchlisted symbols
    intraday_trades = []
    with engine.connect() as conn:
        for sym in symbols:
            # Get dates this symbol is on the watchlist
            wl_dates = {d for d, syms in intraday_watchlists.items() if sym in syms}
            if not wl_dates:
                continue
            trades = run_intraday_on_watchlist(conn, sym, wl_dates, "2026-05-25", "2026-05-29")
            intraday_trades.extend(trades)
            
    # Print Intraday watchlist sizes
    print("\n📅 DAILY INTRADAY WATCHLIST SIZE:")
    for d in sorted(intraday_watchlists.keys()):
        if date(2026, 5, 25) <= d <= date(2026, 5, 29):
            print(f"• {d.strftime('%Y-%m-%d')}: {len(intraday_watchlists[d])} selected stocks")

    # 3. Print Intraday Watchlist Backtest Report
    print("\n" + "="*60)
    print("🔬 WATCHLIST-FILTERED INTRADAY BREAKOUT PERFORMANCE")
    print("="*60)
    
    if not intraday_trades:
        print("No intraday trades triggered on watchlisted stocks.")
    else:
        df_intra = pd.DataFrame(intraday_trades)
        total_i = len(df_intra)
        win_i = df_intra[df_intra["pnl"] > 0]
        win_rate_i = len(win_i) / total_i * 100
        avg_pnl_i = df_intra["pnl"].mean()
        net_pnl_i = df_intra["pnl"].sum()
        
        gross_p = win_i["pnl"].sum()
        gross_l = abs(df_intra[df_intra["pnl"] <= 0]["pnl"].sum())
        pf_i = gross_p / gross_l if gross_l > 0 else float("inf")
        
        print(f"Total Trades Triggered: {total_i} (Raw strategy had 40)")
        print(f"Win Rate              : {win_rate_i:.2f}% (Raw strategy had 50.00%)")
        print(f"Average Return / Trade: {avg_pnl_i:+.3f}% (Raw strategy had +0.149%)")
        print(f"Net Cumulative PnL    : {net_pnl_i:+.2f}% (Raw strategy had +5.96%)")
        print(f"Profit Factor         : {pf_i:.2f} (Raw strategy had 1.72)")
        
        print("\nTop 3 Intraday Watchlist Trades:")
        for _, r in df_intra.sort_values("pnl", ascending=False).head(3).iterrows():
            print(f"• {r['symbol']} ({r['type']}) | Date: {r['entry_time'].strftime('%Y-%m-%d')} | PnL: {r['pnl']:+.2f}%")

    # 4. Print BTST Overnight Backtest Report
    print("\n" + "="*60)
    print("🚀 OPTIMIZED BTST OVERNIGHT PERFORMANCE")
    print("="*60)
    
    if not btst_candidates:
        print("No BTST trades triggered last week.")
    else:
        df_btst = pd.DataFrame(btst_candidates)
        total_b = len(df_btst)
        win_b = df_btst[df_btst["pnl"] > 0]
        win_rate_b = len(win_b) / total_b * 100
        avg_pnl_b = df_btst["pnl"].mean()
        net_pnl_b = df_btst["pnl"].sum()
        
        gross_pb = win_b["pnl"].sum()
        gross_lb = abs(df_btst[df_btst["pnl"] <= 0]["pnl"].sum())
        pf_b = gross_pb / gross_lb if gross_lb > 0 else float("inf")
        
        print(f"Total BTST Trades     : {total_b}")
        print(f"Win Rate              : {win_rate_b:.2f}%")
        print(f"Average Return / Trade: {avg_pnl_b:+.3f}%")
        print(f"Net Cumulative PnL    : {net_pnl_b:+.2f}%")
        print(f"Profit Factor         : {pf_b:.2f}")
        
        print("\nAll BTST Trades Detailed:")
        for _, row in df_btst.sort_values("date").iterrows():
            print(f"• {row['symbol']} | Entry Date: {row['date'].strftime('%Y-%m-%d')} | Vol Surge: {row['vol_surge']:.1f}x | PnL: {row['pnl']:+.2f}%")

if __name__ == "__main__":
    main()
