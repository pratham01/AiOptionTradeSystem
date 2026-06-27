import pandas as pd
import numpy as np
import logging
from pathlib import Path
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.application.indicators.volume_profile import VolumeProfileIndicator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_zone_st")

def run_zone_filtered_backtest(csv_path: str, buffer_pct: float = 0.002):
    logger.info(f"Loading 3m historical data from {csv_path}...")
    if not Path(csv_path).exists():
        logger.error(f"Data file not found at {csv_path}.")
        return

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['date'] = df['timestamp'].dt.date
    
    # 1. Calculate Daily Zones (Supply/Demand)
    logger.info("Calculating Daily Support/Resistance Zones from Volume Profile...")
    vp_indicator = VolumeProfileIndicator(price_step=5.0)
    
    daily_groups = df.groupby('date')
    daily_levels = {}
    
    dates = sorted(daily_groups.groups.keys())
    for i in range(len(dates)):
        day_date = dates[i]
        day_data = daily_groups.get_group(day_date)
        
        # Standard levels
        high = day_data['high'].max()
        low = day_data['low'].min()
        close = day_data['close'].iloc[-1]
        
        # Volume profile levels
        profile = vp_indicator.calculate(day_data)
        
        daily_levels[day_date] = {
            'pdh': high,
            'pdl': low,
            'pdc': close,
            'poc': profile.point_of_control if profile else high,
            'val': profile.value_area_low if profile else low,
            'vah': profile.value_area_high if profile else high
        }

    # 2. Map Previous Day Levels to Current Day
    # We want the 'current row' to know about 'previous day' levels
    zone_data = []
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        levels = daily_levels[prev_date]
        
        curr_day_data = daily_groups.get_group(curr_date).copy()
        for key, val in levels.items():
            curr_day_data[key] = val
        zone_data.append(curr_day_data)
    
    df_with_zones = pd.concat(zone_data)

    # 3. Calculate Supertrend
    logger.info("Calculating 3m Supertrend...")
    indicator = SupertrendIndicator(period=7, multiplier=3)
    df_final = indicator.calculate(df_with_zones)
    
    # 4. Simulate Trades with Zone Confluence
    trades = []
    current_position = 0 
    entry_price = 0
    entry_time = None
    peak_price = 0
    
    for i in range(len(df_final)):
        row = df_final.iloc[i]
        st_sig = row['supertrend_signal']
        st_dir = row['supertrend_direction']
        price = row['close']
        timestamp = row['timestamp']
        
        # Zone checks (within buffer_pct)
        # Support: PDL, VAL, POC
        # Resistance: PDH, VAH, POC
        support_levels = [row['pdl'], row['val'], row['poc']]
        resistance_levels = [row['pdh'], row['vah'], row['poc']]
        
        in_support_zone = any(abs(price - lvl) / lvl <= buffer_pct for lvl in support_levels)
        in_resistance_zone = any(abs(price - lvl) / lvl <= buffer_pct for lvl in resistance_levels)

        is_eod = (timestamp.hour == 15 and timestamp.minute >= 15)
        
        if current_position == 1:
            peak_price = max(peak_price, row['high'])
        elif current_position == -1:
            peak_price = min(peak_price, row['low'])

        # Exit logic
        if current_position != 0:
            exit_reason = None
            if current_position == 1 and (st_dir == -1 or is_eod):
                exit_reason = "ST Flip" if st_dir == -1 else "EOD"
            elif current_position == -1 and (st_dir == 1 or is_eod):
                exit_reason = "ST Flip" if st_dir == 1 else "EOD"
            
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
                    'pnl': pnl,
                    'peak_pts': pts,
                    'reason': exit_reason
                })
                current_position = 0
                
        # Entry logic with Zone Filter
        if current_position == 0 and not is_eod and timestamp.hour < 15:
            if st_sig == 1 and in_support_zone:
                current_position = 1
                entry_price = price
                entry_time = timestamp
                peak_price = row['high']
            elif st_sig == -1 and in_resistance_zone:
                current_position = -1
                entry_price = price
                entry_time = timestamp
                peak_price = row['low']
                
    # 5. Report Results
    if not trades:
        logger.warning("No trades generated with current zone filters.")
        return

    results_df = pd.DataFrame(trades)
    win_rate = (results_df['pnl'] > 0).mean()
    
    print("\n" + "="*60)
    print(f"🏰 ZONE-FILTERED SUPERTREND BACKTEST (NIFTY 50)")
    print("="*60)
    print(f"Strategy:       ST(7,3) + Previous Day Support/Resistance Zones")
    print(f"Buffer:         {buffer_pct*100:.2f}% around zones")
    print(f"Total Trades:   {len(results_df)} (vs 61 original)")
    print(f"Win Rate:       {win_rate:.1%} (vs 50.8% original)")
    print(f"Total PnL %:    {results_df['pnl'].sum():.2%}")
    print(f"Profit Factor:  {abs(results_df[results_df['pnl']>0]['pnl'].sum() / results_df[results_df['pnl']<0]['pnl'].sum()):.2f}")
    print(f"Avg Peak Pts:   {results_df['peak_pts'].mean():.1f}")
    print("="*60)
    
    report_path = Path("reports/supertrend_3m/nifty50_zone_filtered_st.csv")
    results_df.to_csv(report_path, index=False)
    print(f"Detailed log saved to: {report_path}")

if __name__ == "__main__":
    run_zone_filtered_backtest("data/fo_historical/NSE_NIFTY50-INDEX_3min_historical.csv")
