import os
import sys
import pandas as pd
import logging

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.strategy.application.indicators.order_blocks import identify_order_blocks
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_intraday_backtest():
    engine = get_engine()
    symbols = [
        "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", 
        "NSE:INFY-EQ", "NSE:TCS-EQ", "NSE:ITC-EQ", "NSE:SBIN-EQ"
    ]
    
    total_trades = 0
    winning_trades = 0
    total_return = 0.0
    
    logging.info("==================================================")
    logging.info("🕵️ BACKTESTING 15m SMC INTRADAY RADAR")
    logging.info("Rules: Buy on 15m Bullish Sweep or 15m OB Tap.")
    logging.info("Exit: Hold for 8 candles (2 hours) or End of Day.")
    logging.info("==================================================\n")
    
    for sym in symbols:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT timestamp, open, high, low, close FROM ohlcv_15m WHERE symbol = :sym ORDER BY timestamp ASC"),
                conn, params={"sym": sym}
            )
            
        if df.empty or len(df) < 100:
            continue
            
        # Group by day to prevent overnight gaps ruining intraday logic
        df['date'] = pd.to_datetime(df['timestamp'], format='mixed').dt.date
        
        # Step through the dataset (starting at candle 50 to allow OB formation)
        for i in range(50, len(df) - 8, 1):
            past_df = df.iloc[:i].copy()
            curr_row = df.iloc[i]
            
            c_open = curr_row['open']
            c_high = curr_row['high']
            c_low = curr_row['low']
            c_close = curr_row['close']
            c_body = abs(c_close - c_open)
            c_upper_wick = c_high - max(c_close, c_open)
            c_lower_wick = min(c_close, c_open) - c_low
            
            signal = None
            
            # Check OB Tap
            obs = identify_order_blocks(past_df, atr_multiplier=1.2)
            for ob in obs.get('bullish', []):
                if c_low <= ob['top'] and c_close >= ob['bottom']:
                    signal = "Bullish OB Tap"
                    break
                    
            # Check Liq Sweep
            if not signal:
                recent_low = float(past_df['low'].tail(20).min())
                if c_low <= recent_low and c_lower_wick > (c_body * 2) and c_lower_wick > c_upper_wick:
                    signal = "Bullish Sweep"
                    
            if signal:
                entry_price = c_close
                exit_idx = i + 8 # 2 hours forward
                
                # Force EOD exit if it hits the end of the day
                current_date = curr_row['date']
                exit_price = entry_price
                for j in range(i+1, min(i+9, len(df))):
                    if df.iloc[j]['date'] != current_date:
                        exit_price = df.iloc[j-1]['close'] # EOD exit
                        break
                    exit_price = df.iloc[j]['close']
                
                trade_return = ((exit_price - entry_price) / entry_price) * 100
                total_trades += 1
                total_return += trade_return
                if trade_return > 0:
                    winning_trades += 1
                    
                logging.info(f"[{sym}] {signal} at {str(curr_row['timestamp'])[11:16]} -> Ret: {trade_return:+.2f}%")
                
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        avg_ret = total_return / total_trades
        logging.info(f"\n================ OVERALL SCORE ================")
        logging.info(f"Total Intraday Setups: {total_trades}")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Avg Return (2hr/EOD Hold): {avg_ret:.2f}%")
        logging.info(f"Profit Factor estimate: {win_rate / (100 - win_rate) if win_rate < 100 else 'INF'}")
        logging.info(f"==================================================")
    else:
        logging.info("No valid setups found.")

if __name__ == "__main__":
    run_intraday_backtest()
