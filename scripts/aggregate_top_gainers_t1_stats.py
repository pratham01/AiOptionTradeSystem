import sqlite3
import pandas as pd
import numpy as np

con = sqlite3.connect('data/trade_system.db')

# Pull sample top gainers dates and symbols
query = """
SELECT symbol, date(timestamp) as gainer_date, change_pct 
FROM top_gainers_stocks 
WHERE change_pct >= 5.0 
ORDER BY timestamp DESC
LIMIT 500
"""
top_df = pd.read_sql_query(query, con)
print(f"Loaded {len(top_df)} top gainer instances with change_pct >= 5.0%")

t1_stats = []

for _, row in top_df.iterrows():
    sym = row['symbol']
    g_date = row['gainer_date']
    
    # Get daily bars around this date
    q_bars = f"""
    SELECT timestamp, open, high, low, close, volume 
    FROM ohlcv_daily 
    WHERE symbol = '{sym}' AND date(timestamp) <= '{g_date}'
    ORDER BY timestamp DESC
    LIMIT 25
    """
    bars = pd.read_sql_query(q_bars, con)
    if len(bars) < 22:
        continue
    
    bars = bars.sort_values('timestamp').reset_index(drop=True)
    row_t = bars.iloc[-1]
    row_t1 = bars.iloc[-2]
    
    # 20 SMA volume
    vol_sma = bars['volume'].iloc[-22:-2].mean()
    if vol_sma == 0 or pd.isna(vol_sma):
        continue
    vol_ratio_t1 = row_t1['volume'] / vol_sma
    
    # ATR 14
    tr = np.maximum(
        bars['high'] - bars['low'],
        np.maximum(
            (bars['high'] - bars['close'].shift(1)).abs(),
            (bars['low'] - bars['close'].shift(1)).abs()
        )
    )
    atr = tr.iloc[-16:-2].mean()
    if atr == 0 or pd.isna(atr):
        continue
    range_t1 = row_t1['high'] - row_t1['low']
    range_ratio_t1 = range_t1 / atr
    
    # T-1 price gain
    t1_prev = bars['close'].iloc[-3]
    t1_gain = ((row_t1['close'] - t1_prev) / t1_prev) * 100.0 if t1_prev > 0 else 0
    
    # Inside bar & NR7
    is_inside = (row_t1['high'] < bars['high'].iloc[-3]) and (row_t1['low'] > bars['low'].iloc[-3])
    recent_ranges = (bars['high'] - bars['low']).iloc[-8:-1]
    is_nr7 = range_t1 == recent_ranges.min()
    
    # 20 EMA & 50 EMA
    ema20 = bars['close'].iloc[-22:-1].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = bars['close'].iloc[-22:-1].ewm(span=50, adjust=False).mean().iloc[-1]
    
    t1_stats.append({
        "symbol": sym,
        "date": g_date,
        "t_gain": row['change_pct'],
        "t1_gain": t1_gain,
        "vol_ratio_t1": vol_ratio_t1,
        "range_ratio_t1": range_ratio_t1,
        "is_inside": is_inside,
        "is_nr7": is_nr7,
        "above_20ema": row_t1['close'] > ema20,
        "above_50ema": row_t1['close'] > ema50,
        "tight_consolidation": range_ratio_t1 < 0.85,
        "volume_dryup": vol_ratio_t1 < 0.85
    })

stat_df = pd.DataFrame(t1_stats)
print(f"\nAnalyzed {len(stat_df)} instances across multiple top gainers:")
print(f"• Tight Range Coiling on T-1 (Range < 0.85 ATR): {stat_df['tight_consolidation'].mean()*100:.1f}%")
print(f"• Volume Dry-up on T-1 (Vol < 0.85x 20-DMA): {stat_df['volume_dryup'].mean()*100:.1f}%")
print(f"• Inside Bar or NR7 on T-1: {((stat_df['is_inside'] | stat_df['is_nr7'])).mean()*100:.1f}%")
print(f"• Price Above 20 EMA on T-1: {stat_df['above_20ema'].mean()*100:.1f}%")
print(f"• Median T-1 Price Change: {stat_df['t1_gain'].median():+.2f}%")
print(f"• Median T-1 Volume Ratio: {stat_df['vol_ratio_t1'].median():.2f}x")
print(f"• Median T-1 Range/ATR Ratio: {stat_df['range_ratio_t1'].median():.2f}")

con.close()
