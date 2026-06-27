import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import text
from trade_system.infrastructure.database.connection import get_engine

def run_backtest_for_symbol(conn, symbol, start_test_date="2026-05-25", end_test_date="2026-05-29"):
    # Load data with a 2-week warm-up lookback (from May 10, 2026) to avoid indicator cold-starts
    query = text("""
        SELECT timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE symbol = :symbol 
          AND timestamp >= '2026-05-08 00:00:00'
        ORDER BY timestamp ASC
    """)
    df = pd.read_sql(query, conn, params={"symbol": symbol})
    
    if df.empty or len(df) < 120: # Needs sufficient lookback
        return []

    # Sort and parse times
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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
            # Check Exit Conditions
            if position == "LONG":
                if curr_price >= target_price:
                    pnl = (target_price - entry_price) / entry_price * 100
                    # Only record if the trade was initiated in our test week
                    if start_dt <= entry_time.date() <= end_dt:
                        trades.append({"symbol": symbol, "type": "LONG", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": target_price, "pnl": pnl, "exit_reason": "TARGET"})
                    position = None
                elif curr_price <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price * 100
                    if start_dt <= entry_time.date() <= end_dt:
                        trades.append({"symbol": symbol, "type": "LONG", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": sl_price, "pnl": pnl, "exit_reason": "STOP_LOSS"})
                    position = None
                elif is_eod:
                    pnl = (curr_price - entry_price) / entry_price * 100
                    if start_dt <= entry_time.date() <= end_dt:
                        trades.append({"symbol": symbol, "type": "LONG", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": curr_price, "pnl": pnl, "exit_reason": "EOD"})
                    position = None
            elif position == "SHORT":
                if curr_price <= target_price:
                    pnl = (entry_price - target_price) / entry_price * 100
                    if start_dt <= entry_time.date() <= end_dt:
                        trades.append({"symbol": symbol, "type": "SHORT", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": target_price, "pnl": pnl, "exit_reason": "TARGET"})
                    position = None
                elif curr_price >= sl_price:
                    pnl = (entry_price - sl_price) / entry_price * 100
                    if start_dt <= entry_time.date() <= end_dt:
                        trades.append({"symbol": symbol, "type": "SHORT", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": sl_price, "pnl": pnl, "exit_reason": "STOP_LOSS"})
                    position = None
                elif is_eod:
                    pnl = (entry_price - curr_price) / entry_price * 100
                    if start_dt <= entry_time.date() <= end_dt:
                        trades.append({"symbol": symbol, "type": "SHORT", "entry_time": entry_time, "exit_time": row["timestamp"], "entry": entry_price, "exit": curr_price, "pnl": pnl, "exit_reason": "EOD"})
                    position = None
        else:
            # Entry rules
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
                # 2.0x ATR Target, 1.2x ATR SL
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
    
    # 1. Fetch unique symbols having data for last week
    with engine.connect() as conn:
        symbols_query = text("""
            SELECT DISTINCT symbol 
            FROM ohlcv_15m 
            WHERE timestamp >= '2026-05-25 00:00:00' 
              AND timestamp <= '2026-05-29 23:59:59'
        """)
        symbols_df = pd.read_sql(symbols_query, conn)
        symbols = symbols_df["symbol"].tolist()
        
    print(f"Starting Multi-Asset 15m Squeeze Breakout Backtest for Last Week...")
    print(f"Total Symbols to Process: {len(symbols)}")
    
    all_trades = []
    processed_count = 0
    
    with engine.connect() as conn:
        for sym in symbols:
            trades = run_backtest_for_symbol(conn, sym, "2026-05-25", "2026-05-29")
            all_trades.extend(trades)
            processed_count += 1
            if processed_count % 30 == 0:
                print(f"Processed {processed_count}/{len(symbols)} symbols...")
                
    print(f"Backtest complete. Total trades recorded: {len(all_trades)}")
    
    if not all_trades:
        print("No trades triggered last week. This indicates extremely tight compressions or lack of volume expansion.")
        return
        
    df = pd.DataFrame(all_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["trade_date"] = df["entry_time"].dt.date
    
    # Calculate global stats
    total_trades = len(df)
    win_trades = df[df["pnl"] > 0]
    loss_trades = df[df["pnl"] <= 0]
    
    win_rate = len(win_trades) / total_trades * 100
    avg_pnl = df["pnl"].mean()
    total_pnl = df["pnl"].sum()
    
    gross_profits = win_trades["pnl"].sum()
    gross_losses = abs(loss_trades["pnl"].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")
    
    print("\n" + "="*60)
    print("📋 LAST WEEK PERFORMANCE REPORT (MAY 25 - MAY 29, 2026)")
    print("="*60)
    print(f"Total Trades Triggered: {total_trades}")
    print(f"Win Rate              : {win_rate:.2f}%")
    print(f"Average Return / Trade: {avg_pnl:+.3f}%")
    print(f"Net Cumulative PnL    : {total_pnl:+.2f}%")
    print(f"Profit Factor         : {profit_factor:.2f}")
    
    # 2. Daily breakdown
    print("\n📅 DAILY PERFORMANCE BREAKDOWN")
    print("-" * 50)
    daily_stats = df.groupby("trade_date").agg(
        trades_count=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        avg_pnl=('pnl', 'mean'),
        net_pnl=('pnl', 'sum')
    ).reset_index()
    
    print(f"{'Date':<15}{'Trades':<10}{'Win Rate %':<12}{'Avg PnL':<12}{'Net PnL':<10}")
    for _, row in daily_stats.iterrows():
        dt_str = row["trade_date"].strftime("%Y-%m-%d")
        wr_str = f"{row['win_rate']:.2f}%"
        avg_str = f"{row['avg_pnl']:+.3f}%"
        net_str = f"{row['net_pnl']:+.2f}%"
        print(f"{dt_str:<15}{int(row['trades_count']):<10}{wr_str:<12}{avg_str:<12}{net_str:<10}")
        
    # 3. Asset type breakdown (Nifty Index vs Stocks)
    print("\n🧬 ASSET SEGMENT BREAKDOWN")
    print("-" * 50)
    df["segment"] = df["symbol"].apply(lambda s: "INDEX" if "INDEX" in s else "STOCK")
    seg_stats = df.groupby("segment").agg(
        trades_count=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        avg_pnl=('pnl', 'mean'),
        net_pnl=('pnl', 'sum')
    ).reset_index()
    
    print(f"{'Segment':<15}{'Trades':<10}{'Win Rate %':<12}{'Avg PnL':<12}{'Net PnL':<10}")
    for _, row in seg_stats.iterrows():
        wr_str = f"{row['win_rate']:.2f}%"
        avg_str = f"{row['avg_pnl']:+.3f}%"
        net_str = f"{row['net_pnl']:+.2f}%"
        print(f"{row['segment']:<15}{int(row['trades_count']):<10}{wr_str:<12}{avg_str:<12}{net_str:<10}")

    # 4. Exit reason breakdown
    print("\n🚪 EXIT REASONS")
    print("-" * 50)
    print(df["exit_reason"].value_counts().to_string())

    # 5. Top 3 Winners & Top 3 Losers
    print("\n🏆 TOP 3 WINNING TRADES")
    print("-" * 50)
    top_winners = df.sort_values("pnl", ascending=False).head(3)
    for _, r in top_winners.iterrows():
        print(f"• {r['symbol']} ({r['type']}) | Date: {r['entry_time'].strftime('%Y-%m-%d %H:%M')} | PnL: {r['pnl']:+.2f}% | Exit: {r['exit_reason']}")

    print("\n⚠️ TOP 3 LOSING TRADES")
    print("-" * 50)
    top_losers = df.sort_values("pnl", ascending=True).head(3)
    for _, r in top_losers.iterrows():
        print(f"• {r['symbol']} ({r['type']}) | Date: {r['entry_time'].strftime('%Y-%m-%d %H:%M')} | PnL: {r['pnl']:+.2f}% | Exit: {r['exit_reason']}")

if __name__ == "__main__":
    main()
