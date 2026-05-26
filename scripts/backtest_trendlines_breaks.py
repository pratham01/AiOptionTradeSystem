import pandas as pd
import numpy as np
import logging
from pathlib import Path
from trade_system.application.indicators.trendlines_breaks import TrendlinesWithBreaks

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_trendlines")

import argparse

def run_backtest(csv_path: str, length: int = 14, mult: float = 1.0):
    logger.info(f"Loading historical data from {csv_path}...")
    path = Path(csv_path)
    if not path.exists():
        logger.error(f"Data file not found at {csv_path}.")
        return None

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if len(df) < length * 3:
        logger.warning(f"Insufficient data for {path.name}")
        return None

    # 1. Apply Indicator
    strategy = TrendlinesWithBreaks(length=length, mult=mult)
    df = strategy.calculate(df)
    
    # 2. Simulate Trades
    trades = []
    current_position = 0 
    entry_price = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row['signal']
        price = row['close']
        
        if current_position == 1 and sig == -1:
            pnl_pct = (price - entry_price) / entry_price
            trades.append({'symbol': path.name, 'type': 'LONG', 'entry': entry_price, 'exit': price, 'pnl': pnl_pct, 'date': row['timestamp']})
            current_position = 0
            
        elif current_position == -1 and sig == 1:
            pnl_pct = (entry_price - price) / entry_price
            trades.append({'symbol': path.name, 'type': 'SHORT', 'entry': entry_price, 'exit': price, 'pnl': pnl_pct, 'date': row['timestamp']})
            current_position = 0
            
        if current_position == 0:
            if sig == 1:
                current_position = 1
                entry_price = price
            elif sig == -1:
                current_position = -1
                entry_price = price
                
    return trades

def main():
    parser = argparse.ArgumentParser(description="Backtest Trendlines with Breaks")
    parser.add_argument("--asset", type=str, help="Path to single CSV file")
    parser.add_argument("--dir", type=str, help="Directory containing CSV files")
    parser.add_argument("--length", type=int, default=14)
    parser.add_argument("--mult", type=float, default=1.0)
    args = parser.parse_args()

    all_trades = []
    
    if args.asset:
        trades = run_backtest(args.asset, args.length, args.mult)
        if trades: all_trades.extend(trades)
    elif args.dir:
        files = list(Path(args.dir).glob("*.csv"))
        logger.info(f"Scanning {len(files)} files in {args.dir}...")
        for f in files:
            trades = run_backtest(str(f), args.length, args.mult)
            if trades: all_trades.extend(trades)
    else:
        # Default to Nifty
        trades = run_backtest("data/fo_historical/NSE_NIFTY50-INDEX_d_historical.csv", args.length, args.mult)
        if trades: all_trades.extend(trades)

    if not all_trades:
        print("No trades generated.")
        return

    df = pd.DataFrame(all_trades)
    win_rate = (df['pnl'] > 0).mean()
    
    print("\n" + "="*60)
    print(f"📊 AGGREGATED BACKTEST RESULTS (Len={args.length}, Mult={args.mult})")
    print("="*60)
    print(f"Total Trades:   {len(df)}")
    print(f"Win Rate:       {win_rate:.1%}")
    print(f"Total PnL %:    {df['pnl'].sum():.1%}")
    print(f"Avg PnL %:      {df['pnl'].mean():.2%}")
    print(f"Max Drawdown %: {df['pnl'].min():.1%}")
    print(f"Profit Factor:  {abs(df[df['pnl']>0]['pnl'].sum() / df[df['pnl']<0]['pnl'].sum()):.2f}")
    print("="*60)
    
    print("\nTop 5 Performing Assets in this strategy:")
    asset_perf = df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
    print(asset_perf.head(5))

if __name__ == "__main__":
    main()
