import os
import sys
import sqlite3
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from trade_system.domains.strategy.application.indicators.order_blocks import identify_order_blocks

def test_ob_mitigation():
    conn = sqlite3.connect('data/trade_system.db')
    df = pd.read_sql("SELECT timestamp, open, high, low, close, volume FROM ohlcv_daily WHERE symbol = 'NSE:RELIANCE-EQ' ORDER BY timestamp ASC", conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    print(f"Reliance daily candles: {len(df)}")
    obs = identify_order_blocks(df)
    print(f"Bullish OBs: {len(obs['bullish'])}, Bearish OBs: {len(obs['bearish'])}")
    for b in obs['bullish']:
        print("Bullish OB:", b)
    for b in obs['bearish']:
        print("Bearish OB:", b)

if __name__ == "__main__":
    test_ob_mitigation()
