import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

def run_backtest():
    engine = get_engine()
    print("Loading 15m OHLCV data for backtesting (from 2024 onwards)...")
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= '2024-01-01' AND symbol NOT LIKE '%INDEX%'
        ORDER BY symbol, timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"Loaded {len(df):,} candles across {df['symbol'].nunique()} symbols.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    moc_bars = [dt_time(15, 0), dt_time(15, 15)]
    open_bar = dt_time(9, 15)

    records = []
    symbols = df['symbol'].unique()
    print(f"Analyzing {len(symbols)} symbols...")

    for sym in symbols:
        sdf = df[df['symbol'] == sym].sort_values('timestamp')
        date_groups = list(sdf.groupby('date'))
        
        # We need daily MOC volume history to compute 20d rolling SMA
        daily_stats = []
        for i, (d, d_df) in enumerate(date_groups):
            moc_df = d_df[d_df['time'].isin(moc_bars)]
            if len(moc_df) == 0:
                continue
            moc_vol = moc_df['volume'].sum()
            moc_open = float(moc_df.iloc[0]['open'])
            moc_close = float(moc_df.iloc[-1]['close'])
            moc_ret = ((moc_close - moc_open) / moc_open) * 100 if moc_open > 0 else 0.0
            
            day_high = float(d_df['high'].max())
            day_low = float(d_df['low'].min())
            day_close = moc_close
            day_range = day_high - day_low
            close_pos = ((day_close - day_low) / day_range * 100) if day_range > 0 else 50.0
            
            daily_stats.append({
                'date_idx': i,
                'date': d,
                'moc_vol': moc_vol,
                'moc_ret': moc_ret,
                'close': day_close,
                'close_pos': close_pos,
                'd_df': d_df
            })
            
        if len(daily_stats) < 25:
            continue
            
        # Compute 20-day rolling average of MOC volume
        vols = pd.Series([x['moc_vol'] for x in daily_stats])
        rolling_moc_vol = vols.rolling(20, min_periods=5).mean()

        for j in range(20, len(daily_stats) - 1):
            curr = daily_stats[j]
            next_day = daily_stats[j+1]
            
            avg_v = rolling_moc_vol.iloc[j]
            if avg_v <= 0 or curr['moc_vol'] <= 0:
                continue
                
            rvol = curr['moc_vol'] / avg_v
            
            # Find next day open
            next_open_df = next_day['d_df'][next_day['d_df']['time'] == open_bar]
            if next_open_df.empty:
                continue
            next_open = float(next_open_df.iloc[0]['open'])
            curr_close = curr['close']
            
            gap_pct = ((next_open - curr_close) / curr_close) * 100
            
            records.append({
                'symbol': sym,
                'date': curr['date'],
                'next_date': next_day['date'],
                'rvol_moc': rvol,
                'moc_ret': curr['moc_ret'],
                'close_pos': curr['close_pos'],
                'gap_pct': gap_pct
            })

    res_df = pd.DataFrame(records)
    print(f"\nExtracted {len(res_df):,} total overnight instances across 2024-2026.")
    
    # Baseline
    print("\n" + "="*70)
    print(" 📊 BASELINE (UNCONDITIONAL OVERNIGHT GAP DISTRIBUTION)")
    print("="*70)
    baseline_n = len(res_df)
    baseline_gap_up = (res_df['gap_pct'] > 0).mean() * 100
    baseline_gap_down = (res_df['gap_pct'] < 0).mean() * 100
    baseline_avg_gap = res_df['gap_pct'].mean()
    print(f"Total Observations: {baseline_n:,}")
    print(f"Unconditional Gap Up Probability:   {baseline_gap_up:.2f}%")
    print(f"Unconditional Gap Down Probability: {baseline_gap_down:.2f}%")
    print(f"Unconditional Average Overnight Gap: {baseline_avg_gap:+.3f}%\n")

    # Threshold sweeps: RVOL 1.5x, 2.0x, 2.5x, 3.0x
    for rvol_thresh in [1.2, 1.5, 2.0, 2.5, 3.0]:
        print("="*70)
        print(f" 🚀 VOLUME SURGE THRESHOLD: RVOL >= {rvol_thresh}x")
        print("="*70)
        
        # Bullish MOC: RVOL >= thresh AND MOC Return > +0.2%
        bull_df = res_df[(res_df['rvol_moc'] >= rvol_thresh) & (res_df['moc_ret'] >= 0.2)]
        if not bull_df.empty:
            bull_win = (bull_df['gap_pct'] > 0).mean() * 100
            bull_avg_gap = bull_df['gap_pct'].mean()
            bull_median_gap = bull_df['gap_pct'].median()
            bull_big_win = (bull_df['gap_pct'] >= 0.5).mean() * 100
            bull_loss = (bull_df['gap_pct'] <= -0.5).mean() * 100
            print(f"[🟢 BULLISH MOC SURGE] (N={len(bull_df):,} trades)")
            print(f"  • Gap Up Win Rate:        {bull_win:.2f}% (Edge vs Baseline: {bull_win - baseline_gap_up:+.2f}%)")
            print(f"  • Average Gap:            {bull_avg_gap:+.3f}%")
            print(f"  • Median Gap:             {bull_median_gap:+.3f}%")
            print(f"  • Significant Gap Up (≥+0.5%): {bull_big_win:.2f}%")
            print(f"  • Adverse Gap Down (≤-0.5%):   {bull_loss:.2f}%")
        
        # Bearish MOC: RVOL >= thresh AND MOC Return < -0.2%
        bear_df = res_df[(res_df['rvol_moc'] >= rvol_thresh) & (res_df['moc_ret'] <= -0.2)]
        if not bear_df.empty:
            bear_win = (bear_df['gap_pct'] < 0).mean() * 100
            bear_avg_gap = bear_df['gap_pct'].mean()
            bear_median_gap = bear_df['gap_pct'].median()
            bear_big_win = (bear_df['gap_pct'] <= -0.5).mean() * 100
            bear_loss = (bear_df['gap_pct'] >= 0.5).mean() * 100
            print(f"[🔴 BEARISH MOC DUMP] (N={len(bear_df):,} trades)")
            print(f"  • Gap Down Win Rate:      {bear_win:.2f}% (Edge vs Baseline: {bear_win - baseline_gap_down:+.2f}%)")
            print(f"  • Average Gap:            {bear_avg_gap:+.3f}%")
            print(f"  • Median Gap:             {bear_median_gap:+.3f}%")
            print(f"  • Significant Gap Down (≤-0.5%): {bear_big_win:.2f}%")
            print(f"  • Adverse Gap Up (≥+0.5%):     {bear_loss:.2f}%")

    # High Conviction: RVOL >= 2.0x + Day Extremes (Closing in top 15% / bottom 15% of day range)
    print("\n" + "="*70)
    print(" 🎯 HIGH-CONVICTION COMBO: RVOL >= 2.0x + DAY EXTREME CLOSE")
    print("="*70)
    hc_bull = res_df[(res_df['rvol_moc'] >= 2.0) & (res_df['moc_ret'] >= 0.3) & (res_df['close_pos'] >= 85)]
    if not hc_bull.empty:
        hc_b_win = (hc_bull['gap_pct'] > 0).mean() * 100
        print(f"[🟢 HIGH-CONVICTION BULLISH] N={len(hc_bull):,}")
        print(f"  • Gap Up Win Rate: {hc_b_win:.2f}%")
        print(f"  • Avg Overnight Gap: {hc_bull['gap_pct'].mean():+.3f}%")
        print(f"  • Median Overnight Gap: {hc_bull['gap_pct'].median():+.3f}%")

    hc_bear = res_df[(res_df['rvol_moc'] >= 2.0) & (res_df['moc_ret'] <= -0.3) & (res_df['close_pos'] <= 15)]
    if not hc_bear.empty:
        hc_s_win = (hc_bear['gap_pct'] < 0).mean() * 100
        print(f"[🔴 HIGH-CONVICTION BEARISH] N={len(hc_bear):,}")
        print(f"  • Gap Down Win Rate: {hc_s_win:.2f}%")
        print(f"  • Avg Overnight Gap: {hc_bear['gap_pct'].mean():+.3f}%")
        print(f"  • Median Overnight Gap: {hc_bear['gap_pct'].median():+.3f}%")

    # Save summary stats to CSV for report generation
    res_df.to_csv("scratch/moc_gap_backtest_results.csv", index=False)
    print("\nBacktest complete. Results saved to scratch/moc_gap_backtest_results.csv")

if __name__ == "__main__":
    run_backtest()
