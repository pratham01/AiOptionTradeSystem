import pandas as pd
from sqlalchemy import text
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

def main():
    engine = get_engine()
    
    # 1. Fetch all 15m candle records for F&O symbols (excluding indices) from the last 15 days
    query = text("""
        SELECT symbol, date(timestamp) as trade_date, SUM(volume) as daily_volume
        FROM ohlcv_15m
        WHERE symbol LIKE '%-EQ%'
        GROUP BY symbol, trade_date
        ORDER BY symbol, trade_date
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    if df.empty:
        print("No F&O stock volume data found in the database.")
        return
        
    df['daily_volume'] = df['daily_volume'].astype(float)
    today_str = "2026-06-29"
    df_history = df[df['trade_date'] < today_str]
    df_today = df[df['trade_date'] == today_str]
    
    if df_today.empty:
        print(f"No volume records found in the database for today ({today_str}).")
        return
        
    baseline = df_history.groupby('symbol')['daily_volume'].mean().reset_index()
    baseline.rename(columns={'daily_volume': 'avg_daily_volume'}, inplace=True)
    
    merged = pd.merge(df_today, baseline, on='symbol')
    merged = merged[merged['avg_daily_volume'] > 1000]
    merged['volume_ratio'] = merged['daily_volume'] / merged['avg_daily_volume']
    
    # 2. Fetch current close price (LTP) for today
    price_query = text("""
        SELECT symbol, close as ltp, timestamp
        FROM ohlcv_15m
        WHERE (symbol, timestamp) IN (
            SELECT symbol, MAX(timestamp)
            FROM ohlcv_15m
            WHERE date(timestamp) = :today
            GROUP BY symbol
        )
    """)
    with engine.connect() as conn:
        df_prices = pd.read_sql(price_query, conn, params={"today": today_str})
        
    # 3. Fetch previous day's close price
    prev_close_query = text("""
        SELECT symbol, close as prev_close
        FROM ohlcv_15m
        WHERE (symbol, timestamp) IN (
            SELECT symbol, MAX(timestamp)
            FROM ohlcv_15m
            WHERE date(timestamp) < :today
            GROUP BY symbol
        )
    """)
    with engine.connect() as conn:
        df_prev_close = pd.read_sql(prev_close_query, conn, params={"today": today_str})
        
    # Merge all data
    merged = pd.merge(merged, df_prices, on='symbol', how='left')
    merged = pd.merge(merged, df_prev_close, on='symbol', how='left')
    
    # Calculate price change percent
    merged['price_change'] = ((merged['ltp'] - merged['prev_close']) / merged['prev_close']) * 100
    
    # Sort by volume ratio
    top_volume_surge = merged.sort_values(by='volume_ratio', ascending=False).head(15)
    
    print("\n" + "="*95)
    print(f"🔥 TOP 15 F&O STOCKS WITH HIGHEST VOLUME SURGE (Today vs 10-Day Avg)")
    print(f"Date: {today_str} | Current Time: 13:42 IST")
    print("="*95)
    print(f"{'Symbol':<15} | {'LTP (Rs.)':<10} | {'Change %':<10} | {'Today Vol':<12} | {'Avg Daily Vol':<14} | {'Surge %':<8}")
    print("-"*95)
    for _, row in top_volume_surge.iterrows():
        symbol_clean = row['symbol'].replace("NSE:", "").replace("-EQ", "")
        ltp_str = f"{row['ltp']:.2f}" if pd.notna(row['ltp']) else "N/A"
        
        # Color or format price change direction
        change_val = row['price_change']
        if pd.isna(change_val):
            change_str = "N/A"
        else:
            sign = "+" if change_val >= 0 else ""
            change_str = f"{sign}{change_val:.2f}%"
            
        surge_pct = f"{row['volume_ratio'] * 100:.1f}%"
        print(f"{symbol_clean:<15} | {ltp_str:<10} | {change_str:<10} | {int(row['daily_volume']):<12,} | {int(row['avg_daily_volume']):<14,} | {surge_pct:<8}")
    print("="*95)

if __name__ == "__main__":
    main()
