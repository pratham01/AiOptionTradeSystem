import pandas as pd
import numpy as np
import logging
from pathlib import Path
from trade_system.application.indicators.supertrend import SupertrendIndicator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_enhanced_st")

def run_enhanced_backtest(
    csv_3m: str, 
    csv_d: str,
    st_period: int = 7, 
    st_mult: int = 3,
    anchor_period: int = 10,
    anchor_mult: int = 3,
    volume_multiplier: float = 1.2
):
    logger.info(f"Loading data: 3m={csv_3m}, Daily={csv_d}")
    
    df_3m = pd.read_csv(csv_3m, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df_d = pd.read_csv(csv_d, parse_dates=['timestamp']).sort_values('timestamp')
    
    # 1. Prepare Daily Volume Filter
    # Calculate 20-day average volume
    df_d['avg_vol_20'] = df_d['volume'].rolling(window=20).mean().shift(1)
    # Map daily avg volume to 3m bars (simple join on date)
    df_3m['date'] = df_3m['timestamp'].dt.date
    df_d['date'] = df_d['timestamp'].dt.date
    df_3m = df_3m.merge(df_d[['date', 'avg_vol_20']], on='date', how='left')
    
    # Approx 3m benchmark volume = (Daily Avg / 75 bars per day)
    df_3m['vol_benchmark'] = (df_3m['avg_vol_20'] / 75) * volume_multiplier

    # 2. Calculate Indicators
    # 3m Supertrend
    st_3m = SupertrendIndicator(period=st_period, multiplier=st_mult).calculate(df_3m)
    
    # 15m Anchor Trend (Resampled from 3m)
    df_15m = df_3m.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    st_15m = SupertrendIndicator(period=anchor_period, multiplier=anchor_mult).calculate(df_15m)
    st_15m = st_15m[['supertrend_direction']].rename(columns={'supertrend_direction': 'anchor_dir'})
    
    # Join Anchor back to 3m
    df = st_3m.merge(st_15m, left_on='timestamp', right_index=True, how='left')
    df['anchor_dir'] = df['anchor_dir'].ffill()

    # 3. Simulate Trades with CONFLUENCE
    # Rules:
    # - Entry: 3m Flip AND (3m Dir == Anchor Dir) AND (Volume > benchmark)
    # - Exit: 3m Flip OR EOD
    
    trades = []
    current_position = 0 
    entry_price = 0
    entry_time = None
    peak_price = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = row['timestamp']
        sig_3m = row['supertrend_signal'] # 1, -1, or 0
        dir_3m = row['supertrend_direction']
        anchor = row['anchor_dir']
        vol = row['volume']
        bench = row['vol_benchmark']
        price = row['close']
        
        is_eod = (timestamp.hour == 15 and timestamp.minute >= 15)
        
        # Track peak
        if current_position == 1: peak_price = max(peak_price, row['high'])
        elif current_position == -1: peak_price = min(peak_price, row['low'])

        # Exit Logic
        if current_position != 0:
            exit_reason = None
            # Standard flip exit
            if current_position == 1 and (dir_3m == -1 or is_eod):
                exit_reason = "3m Flip" if dir_3m == -1 else "EOD"
                pnl = (price - entry_price) / entry_price
                pts = peak_price - entry_price
            elif current_position == -1 and (dir_3m == 1 or is_eod):
                exit_reason = "3m Flip" if dir_3m == 1 else "EOD"
                pnl = (entry_price - price) / entry_price
                pts = entry_price - peak_price
                
            if exit_reason:
                trades.append({
                    'type': 'LONG' if current_position == 1 else 'SHORT',
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'pnl': pnl,
                    'peak_pts': pts,
                    'reason': exit_reason
                })
                current_position = 0

        # Entry Logic (Confluence)
        if current_position == 0 and not is_eod and timestamp.hour < 15:
            # 1. Trend Flip exists?
            if sig_3m != 0:
                # 2. Does it align with 15m Anchor?
                if sig_3m == anchor:
                    # 3. Is volume significant?
                    if vol > bench:
                        current_position = int(sig_3m)
                        entry_price = price
                        entry_time = timestamp
                        peak_price = row['high'] if current_position == 1 else row['low']
                    else:
                        pass # Filtered by Volume
                else:
                    pass # Filtered by Anchor Trend

    # 4. Report Results
    if not trades:
        print("No trades passed the confluence filters.")
        return

    res = pd.DataFrame(trades)
    win_rate = (res['pnl'] > 0).mean()
    
    print("\n" + "="*60)
    print(f"🛡️ ENHANCED CONFLUENCE BACKTEST (NIFTY 50)")
    print("="*60)
    print(f"Filters:        15m Anchor + Volume > {volume_multiplier}x")
    print(f"Total Trades:   {len(res)} (vs 61 in original)")
    print(f"Win Rate:       {win_rate:.1%} (vs 50.8% original)")
    print(f"Total PnL %:    {res['pnl'].sum():.2%}")
    print(f"Profit Factor:  {abs(res[res['pnl']>0]['pnl'].sum() / res[res['pnl']<0]['pnl'].sum()):.2f}")
    print(f"Avg Peak Pts:   {res['peak_pts'].mean():.1f}")
    print("="*60)
    
    report_path = Path("reports/supertrend_3m/nifty50_enhanced_confluence.csv")
    res.to_csv(report_path, index=False)
    print(f"Detailed log saved to: {report_path}")

if __name__ == "__main__":
    run_enhanced_backtest(
        "data/fo_historical/NSE_NIFTY50-INDEX_3min_historical.csv",
        "data/fo_historical/NSE_NIFTY50-INDEX_d_historical.csv",
        volume_multiplier=0.0
    )
