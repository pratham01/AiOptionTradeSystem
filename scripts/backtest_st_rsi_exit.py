import pandas as pd
import numpy as np
import logging
from pathlib import Path
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.application.indicators.rsi_divergence import RsiDivergence

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_st_rsi_exit")

def run_st_rsi_exit_backtest(csv_path: str, period: int = 7, multiplier: int = 3, rsi_fast: int = 5, rsi_slow: int = 14):
    logger.info(f"Loading 3m historical data from {csv_path}...")
    if not Path(csv_path).exists():
        logger.error(f"Data file not found at {csv_path}.")
        return

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Calculate Indicators
    logger.info(f"Calculating Supertrend ({period}, {multiplier}) and RSI Divergence ({rsi_fast}, {rsi_slow})...")
    st_indicator = SupertrendIndicator(period=period, multiplier=multiplier)
    df = st_indicator.calculate(df)
    
    rsi_indicator = RsiDivergence(len_fast=rsi_fast, len_slow=rsi_slow)
    df = rsi_indicator.calculate(df)
    
    # 2. Simulate Trades
    # Entry: Supertrend Flip
    # Exit: Supertrend Reverse Flip OR RSI Divergence cross 0 OR EOD
    
    trades = []
    current_position = 0 
    entry_price = 0
    entry_time = None
    peak_price = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        st_sig = row['supertrend_signal']      # Flip trigger: 1, -1, 0
        st_dir = row['supertrend_direction']   # Current state: 1, -1
        rsi_div = row['divergence']            # Fast RSI - Slow RSI
        price = row['close']
        timestamp = row['timestamp']
        
        is_eod = (timestamp.hour == 15 and timestamp.minute >= 15)
        
        # Peak profit tracking
        if current_position == 1:
            peak_price = max(peak_price, row['high'])
        elif current_position == -1:
            peak_price = min(peak_price, row['low'])

        # Exit logic
        if current_position != 0:
            exit_reason = None
            
            # Condition 1: Supertrend Flip (Stop Loss / Trend Change)
            if current_position == 1 and st_dir == -1:
                exit_reason = "ST Flip"
            elif current_position == -1 and st_dir == 1:
                exit_reason = "ST Flip"
            
            # Condition 2: RSI Divergence Early Exit (Take Profit)
            # If LONG, exit if fast RSI drops below slow RSI (lost momentum)
            elif current_position == 1 and rsi_div < 0:
                exit_reason = "RSI Early Exit"
            # If SHORT, exit if fast RSI rises above slow RSI (lost momentum)
            elif current_position == -1 and rsi_div > 0:
                exit_reason = "RSI Early Exit"
            
            # Condition 3: EOD Squareoff
            elif is_eod:
                exit_reason = "EOD"
            
            if exit_reason:
                if current_position == 1:
                    pnl = (price - entry_price) / entry_price
                    pts = peak_price - entry_price
                else:
                    pnl = (entry_price - price) / entry_price
                    pts = entry_price - peak_price
                
                trades.append({
                    'type': 'LONG' if current_position == 1 else 'SHORT',
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'peak_pts': pts,
                    'pnl': pnl,
                    'reason': exit_reason
                })
                current_position = 0
                
        # Entry logic (Avoid entries after 3:00 PM)
        if current_position == 0 and not is_eod and timestamp.hour < 15:
            if st_sig == 1:
                current_position = 1
                entry_price = price
                entry_time = timestamp
                peak_price = row['high']
            elif st_sig == -1:
                current_position = -1
                entry_price = price
                entry_time = timestamp
                peak_price = row['low']
                
    # 3. Report Results
    if not trades:
        logger.warning("No trades generated.")
        return

    results_df = pd.DataFrame(trades)
    win_rate = (results_df['pnl'] > 0).mean()
    total_pnl = results_df['pnl'].sum()
    
    # Save Report
    report_dir = Path("reports/supertrend_3m")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "nifty50_st_rsi_exit.csv"
    results_df.to_csv(report_path, index=False)
    
    print("\n" + "="*60)
    print(f"🧬 SUPERTREND + RSI DIVERGENCE EXIT (NIFTY 50)")
    print("="*60)
    print(f"Strategy:       ST(7,3) Entry | RSI({rsi_fast},{rsi_slow}) Early Exit")
    print(f"Total Trades:   {len(results_df)}")
    print(f"Win Rate:       {win_rate:.1%}")
    print(f"Total PnL %:    {total_pnl:.2%}")
    print(f"Profit Factor:  {abs(results_df[results_df['pnl']>0]['pnl'].sum() / results_df[results_df['pnl']<0]['pnl'].sum()):.2f}")
    print(f"Avg Peak Pts:   {results_df['peak_pts'].mean():.1f}")
    print("-" * 60)
    print("Exit Reason Breakdown:")
    print(results_df['reason'].value_counts(normalize=True).map(lambda x: f"{x:.1%}") )
    print("="*60)
    
    print("\nLast 5 Trades:")
    print(results_df.tail(5)[['type', 'entry_time', 'exit_time', 'pnl', 'reason', 'peak_pts']].to_string(index=False))
    print(f"\nDetailed log: {report_path}")

if __name__ == "__main__":
    run_st_rsi_exit_backtest("data/fo_historical/NSE_NIFTY50-INDEX_3min_historical.csv")
