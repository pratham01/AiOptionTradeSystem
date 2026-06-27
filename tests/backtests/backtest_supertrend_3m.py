import pandas as pd
import numpy as np
import logging
from pathlib import Path
from trade_system.application.indicators.supertrend import SupertrendIndicator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_supertrend_3m")

def run_3m_backtest(csv_path: str, period: int = 7, multiplier: int = 3):
    logger.info(f"Loading 3m historical data from {csv_path}...")
    if not Path(csv_path).exists():
        logger.error(f"Data file not found at {csv_path}.")
        return

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} rows. Calculating Supertrend ({period}, {multiplier})...")
    
    # 1. Apply Indicator
    indicator = SupertrendIndicator(period=period, multiplier=multiplier)
    df = indicator.calculate(df)
    
    # 2. Simulate Trades (Intraday Option Buyer logic)
    # Rules: 
    # - Buy on supertrend_signal == 1 (Bullish flip)
    # - Sell on supertrend_signal == -1 (Bearish flip)
    # - Close all positions at EOD (3:15 PM)
    
    trades = []
    current_position = 0 
    entry_price = 0
    entry_time = None
    peak_price = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row['supertrend_signal']
        price = row['close']
        high = row['high']
        low = row['low']
        timestamp = row['timestamp']
        
        is_eod = (timestamp.hour == 15 and timestamp.minute >= 15)
        
        # Track peak profit reached during trade
        if current_position == 1:
            peak_price = max(peak_price, high)
        elif current_position == -1:
            peak_price = min(peak_price, low)

        # Exit logic
        if current_position != 0:
            exit_reason = None
            if current_position == 1 and (sig == -1 or is_eod):
                exit_reason = "Signal Flip" if sig == -1 else "EOD Squareoff"
                pnl_pct = (price - entry_price) / entry_price
                points_traveled = peak_price - entry_price
            elif current_position == -1 and (sig == 1 or is_eod):
                exit_reason = "Signal Flip" if sig == 1 else "EOD Squareoff"
                pnl_pct = (entry_price - price) / entry_price
                points_traveled = entry_price - peak_price
            
            if exit_reason:
                trades.append({
                    'type': 'LONG' if current_position == 1 else 'SHORT',
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'peak_profit_pts': points_traveled,
                    'pnl': pnl_pct,
                    'reason': exit_reason
                })
                current_position = 0
                
        # Entry logic (Avoid entries after 3:00 PM)
        if current_position == 0 and not is_eod and timestamp.hour < 15:
            if sig == 1:
                current_position = 1
                entry_price = price
                entry_time = timestamp
                peak_price = high
            elif sig == -1:
                current_position = -1
                entry_price = price
                entry_time = timestamp
                peak_price = low
                
    # 3. Report Results
    if not trades:
        logger.warning("No trades generated.")
        return

    results_df = pd.DataFrame(trades)
    
    # Save Report
    report_dir = Path("reports/supertrend_3m")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "nifty50_3m_last_month.csv"
    results_df.to_csv(report_path, index=False)
    
    win_rate = (results_df['pnl'] > 0).mean()
    total_return = results_df['pnl'].sum()
    
    print("\n" + "="*60)
    print(f"🚀 SUPERTREND 3M BACKTEST RESULTS (NIFTY 50)")
    print("="*60)
    print(f"Period:         Last 30 Days")
    print(f"Config:         Period={period}, Multiplier={multiplier}")
    print(f"Total Trades:   {len(results_df)}")
    print(f"Win Rate:       {win_rate:.1%}")
    print(f"Total PnL %:    {total_return:.2%}")
    print(f"Avg per Trade:  {results_df['pnl'].mean():.2%}")
    print(f"Profit Factor:  {abs(results_df[results_df['pnl']>0]['pnl'].sum() / results_df[results_df['pnl']<0]['pnl'].sum()):.2f}")
    print("="*60)
    
    print("\nLast 5 Trades:")
    print(results_df.tail(5)[['type', 'entry_time', 'exit_time', 'peak_profit_pts', 'pnl', 'reason']].to_string(index=False))

if __name__ == "__main__":
    run_3m_backtest("data/fo_historical/NSE_NIFTY50-INDEX_3min_historical.csv")
