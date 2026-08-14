import os
import sys
import logging
import pandas as pd
from sqlalchemy import text
from datetime import date, timedelta
from typing import List, Dict, Any

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

def fetch_daily_data_from_5m(days_lookback: int = 40) -> pd.DataFrame:
    engine = get_engine()
    start_date = (date.today() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
    
    query = text(f"""
        SELECT 
            symbol,
            date(timestamp) as trade_date,
            MIN(open) as min_open, -- Not strictly correct for open, but we'll fix in pandas
            MAX(high) as high,
            MIN(low) as low,
            SUM(volume) as volume
        FROM ohlcv_5m
        WHERE date(timestamp) >= :start_date
        GROUP BY symbol, trade_date
        ORDER BY symbol, trade_date ASC
    """)
    
    # We need the exact Open and Close. SQLite GROUP BY doesn't guarantee first/last easily
    # So we'll fetch all 5m data and resample in pandas which is robust.
    full_query = text(f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv_5m
        WHERE date(timestamp) >= :start_date
        ORDER BY symbol, timestamp ASC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(full_query, conn, params={"start_date": start_date})
        
    if df.empty:
        return df
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['trade_date'] = df['timestamp'].dt.date
    
    # Aggregate to daily
    daily_df = df.groupby(['symbol', 'trade_date']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    return daily_df

def run_screener():
    LOGGER.info("Starting EOD FO Screener Phase 1...")
    
    df = fetch_daily_data_from_5m(days_lookback=45) # Need ~20 trading days for BB
    if df.empty:
        LOGGER.error("No 5m data found. Please run backfill first.")
        return
        
    symbols = df['symbol'].unique()
    results = []
    
    target_date = df['trade_date'].max()
    LOGGER.info(f"Running analysis for latest date: {target_date}")
    
    for symbol in symbols:
        sym_df = df[df['symbol'] == symbol].copy()
        if len(sym_df) < 20:
            continue
            
        sym_df = sym_df.sort_values('trade_date').reset_index(drop=True)
        latest = sym_df.iloc[-1]
        
        # 1. Volatility Contraction (VCP) - Bollinger Band Width
        # 20-day SMA & StdDev
        sym_df['sma_20'] = sym_df['close'].rolling(window=20).mean()
        sym_df['std_20'] = sym_df['close'].rolling(window=20).std()
        sym_df['upper_bb'] = sym_df['sma_20'] + (2 * sym_df['std_20'])
        sym_df['lower_bb'] = sym_df['sma_20'] - (2 * sym_df['std_20'])
        sym_df['bb_width'] = (sym_df['upper_bb'] - sym_df['lower_bb']) / sym_df['sma_20']
        
        latest_bb_width = sym_df['bb_width'].iloc[-1]
        is_vcp = latest_bb_width < 0.08  # Less than 8% width implies tight consolidation
        
        # 2. Volume Surge
        # 10-day average volume (excluding latest day)
        if len(sym_df) >= 11:
            avg_vol_10 = sym_df['volume'].iloc[-11:-1].mean()
            vol_surge = latest['volume'] / avg_vol_10 if avg_vol_10 > 0 else 0
        else:
            vol_surge = 0
            
        is_surge = vol_surge > 1.5
        
        # 3. Closing Strength
        day_range = latest['high'] - latest['low']
        if day_range > 0:
            close_strength = (latest['close'] - latest['low']) / day_range
        else:
            close_strength = 0
            
        is_strong_close = close_strength >= 0.85
        
        # Scoring
        score = sum([is_vcp, is_surge, is_strong_close])
        
        if score >= 1: # Print anything that has at least 1 compelling metric for visibility
            results.append({
                "Symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                "Score": score,
                "BB Width": f"{latest_bb_width:.1%}",
                "VCP": "✅" if is_vcp else "❌",
                "Vol Surge": f"{vol_surge:.2f}x",
                "Surge": "✅" if is_surge else "❌",
                "Close Str": f"{close_strength:.1%}",
                "Strong Close": "✅" if is_strong_close else "❌",
            })
            
    if not results:
        LOGGER.info("No stocks met the criteria today.")
        return
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=["Score", "Vol Surge"], ascending=[False, False])
    
    # Print markdown table
    print("\n# EOD Watchlist Screener")
    print(f"**Date:** {target_date}\n")
    print(res_df.to_markdown(index=False))

if __name__ == "__main__":
    run_screener()
