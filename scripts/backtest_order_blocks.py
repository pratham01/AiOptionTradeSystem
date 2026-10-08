import os
import sys
import pandas as pd
from datetime import timedelta
import logging

# Setup Python Path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.strategy.application.indicators.order_blocks import identify_order_blocks
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_backtest(forward_days=10):
    engine = get_engine()
    
    # Get top liquid stocks
    symbols = [
        "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", 
        "NSE:INFY-EQ", "NSE:TCS-EQ", "NSE:ITC-EQ", "NSE:SBIN-EQ",
        "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ", "NSE:LT-EQ"
    ]
    
    total_trades = 0
    winning_trades = 0
    total_return = 0.0
    
    logging.info(f"Starting Order Block Backtest across {len(symbols)} liquid stocks...")
    logging.info(f"Rules: Enter when price taps a Bullish OB. Hold for {forward_days} days.\n")
    
    for sym in symbols:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT timestamp, open, high, low, close FROM ohlcv_daily WHERE symbol = :sym ORDER BY timestamp ASC"),
                conn, params={"sym": sym}
            )
            
        if df.empty or len(df) < 50:
            continue
            
        # We need to simulate the market step by step to prevent lookahead bias
        # Start from day 50, run the OB detector on past data
        for i in range(50, len(df) - forward_days, 5): # Step by 5 days for speed
            past_df = df.iloc[:i].copy()
            curr_row = df.iloc[i]
            
            # Find OBs using past data only
            obs = identify_order_blocks(past_df, atr_multiplier=1.5)
            bullish_obs = obs.get('bullish', [])
            
            # Check if current day mitigates any active OB
            for ob in bullish_obs:
                if not ob['broken'] and not ob['mitigated']:
                    # Does the current day dip into the OB?
                    if curr_row['low'] <= ob['top'] and curr_row['close'] >= ob['bottom']:
                        # Trade triggered! Price tapped the OB and didn't close below it.
                        entry_price = curr_row['close']
                        
                        # Look forward N days
                        exit_idx = i + forward_days
                        if exit_idx >= len(df):
                            continue
                            
                        exit_price = df.iloc[exit_idx]['close']
                        trade_return = ((exit_price - entry_price) / entry_price) * 100
                        
                        total_trades += 1
                        total_return += trade_return
                        if trade_return > 0:
                            winning_trades += 1
                            
                        logging.info(f"[{sym}] Tapped OB created on {str(ob['start_date']).split()[0]} -> Entry: {entry_price:.2f}, Exit ({forward_days}d): {exit_price:.2f} | Return: {trade_return:+.2f}%")
                        break # Only take one trade per day
                        
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        avg_return = total_return / total_trades
        logging.info(f"\n================ BACKTEST RESULTS ================")
        logging.info(f"Total OB Bounces Traded: {total_trades}")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Average Return per Trade: {avg_return:.2f}%")
        logging.info(f"==================================================")
    else:
        logging.info("No valid OB touches found in dataset.")

if __name__ == "__main__":
    run_backtest(forward_days=5)
