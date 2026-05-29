import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from datetime import date, datetime
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.data.fo_universe import get_sector_mapping
from trade_system.core.ports.broker import MarketQuote
from sqlalchemy import text

# Mimic fetch_available_dates and fetch_target_date_and_data
def run_verification():
    print("=== Testing Dashboard Fallback Logic ===")
    
    # 1. Fetch available dates
    engine = get_engine()
    query = text("SELECT DISTINCT date(timestamp) as d FROM ohlcv_15m WHERE symbol != 'NSE:NIFTY50-INDEX' ORDER BY d DESC")
    with engine.connect() as conn:
        result = conn.execute(query).fetchall()
    available_dates = [row[0] for row in result if row[0] is not None]
    print(f"Available dates in database: {available_dates[:5]} ... total {len(available_dates)}")

    # Inject today's date (today is 2026-05-29)
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Today is: {today_str}")
    if today_str not in available_dates:
        available_dates.insert(0, today_str)
        print("Injected today's date into available_dates.")

    selected_date_str = available_dates[0]
    print(f"Selected analysis date: {selected_date_str}")

    # Fetch last 10 days of data around latest date
    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE timestamp >= date(:latest_date, '-10 days') AND timestamp <= date(:latest_date, '+1 day')
        ORDER BY symbol, timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"latest_date": selected_date_str})
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    print(f"Fetched {len(df)} candles from database.")

    target_date = pd.to_datetime(selected_date_str).date()
    fo_metadata = get_sector_mapping()
    df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))

    # Test Fallback Resolution
    unique_db_dates = sorted(list(df['timestamp'].dt.date.unique()))
    print(f"Unique dates in fetched dataframe: {unique_db_dates}")
    
    db_target_date = target_date
    if target_date not in unique_db_dates and unique_db_dates:
        db_target_date = unique_db_dates[-1]
    print(f"db_target_date resolved: {db_target_date}")

    is_today = (target_date == date.today())
    print(f"is_today originally: {is_today}")

    # Simulate quotes fetch failure (empty)
    quotes = {}
    print("Simulating empty quotes fetch (Fyers 429 rate limit)...")

    # Run our fallback logic
    if is_today and not quotes and target_date not in unique_db_dates:
        print(f"FALLBACK TRIGGERED: target_date changed to {db_target_date}")
        target_date = db_target_date
        is_today = False

    # Now calculate today_df and prev_df
    today_df = df[df['timestamp'].dt.date == target_date]
    prev_df = df[df['timestamp'].dt.date < target_date]
    
    print(f"today_df count: {len(today_df)}, prev_df count: {len(prev_df)}")

    latest_closes = today_df[~today_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()
    prev_closes = prev_df[~prev_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()
    
    merged_closes = pd.merge(latest_closes, prev_closes, on=['symbol', 'sector'], suffixes=('_last', '_prev'))
    print(f"merged_closes count (from DB fallback): {len(merged_closes)}")
    if not merged_closes.empty:
        merged_closes['pChange'] = ((merged_closes['close_last'] - merged_closes['close_prev']) / merged_closes['close_prev']) * 100
        print("Calculated pChange for fallback closes.")
        print(merged_closes[['symbol', 'sector', 'close_last', 'close_prev', 'pChange']].head(3))
    
    print("--------------------------------------------------")
    
    # Simulate quotes fetch SUCCESS
    print("Simulating successful quotes fetch...")
    target_date = pd.to_datetime(selected_date_str).date()
    is_today = (target_date == date.today())
    
    # Mock some quotes
    mock_quotes = {
        "NSE:SBIN-EQ": MarketQuote(
            symbol="NSE:SBIN-EQ", exchange="NSE", last_price=850.5, open=840.0,
            high=855.0, low=839.0, close=850.5, previous_close=838.0, volume=10000,
            change=12.5, change_percent=1.49, timestamp=datetime.now()
        ),
        "NSE:RELIANCE-EQ": MarketQuote(
            symbol="NSE:RELIANCE-EQ", exchange="NSE", last_price=2450.0, open=2420.0,
            high=2460.0, low=2415.0, close=2450.0, previous_close=2420.0, volume=20000,
            change=30.0, change_percent=1.24, timestamp=datetime.now()
        )
    }

    # Run fallback check again (should not trigger fallback since quotes exist)
    if is_today and not mock_quotes and target_date not in unique_db_dates:
        print("WARNING: Fallback triggered unexpectedly!")
        target_date = db_target_date
        is_today = False
    else:
        print("Fallback not triggered (quotes present). target_date remains:", target_date)

    today_df = df[df['timestamp'].dt.date == target_date]  # empty
    prev_df = df[df['timestamp'].dt.date < target_date]
    latest_closes = today_df[~today_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()  # empty
    prev_closes = prev_df[~prev_df['symbol'].str.contains("INDEX")].groupby('symbol').last().reset_index()
    
    merged_closes = pd.merge(latest_closes, prev_closes, on=['symbol', 'sector'], suffixes=('_last', '_prev'))  # empty
    print(f"merged_closes count (from DB today): {len(merged_closes)}")

    # Update/enrich merged_closes using the successfully fetched live quotes
    if is_today and mock_quotes:
        live_rows = []
        for symbol, quote in mock_quotes.items():
            sector = fo_metadata.get(symbol, "UNKNOWN")
            ltp = quote.last_price or quote.close or quote.open
            prev_close = quote.previous_close
            if prev_close > 0 and ltp > 0:
                pchange = ((ltp - prev_close) / prev_close) * 100
                live_rows.append({
                    "symbol": symbol,
                    "sector": sector,
                    "close_last": ltp,
                    "close_prev": prev_close,
                    "pChange": pchange
                })
        
        if live_rows:
            live_df = pd.DataFrame(live_rows)
            if merged_closes.empty:
                merged_closes = live_df
            else:
                merged_closes = merged_closes.set_index('symbol')
                live_df = live_df.set_index('symbol')
                merged_closes.update(live_df)
                new_symbols = live_df.index.difference(merged_closes.index)
                if not new_symbols.empty:
                    merged_closes = pd.concat([merged_closes, live_df.loc[new_symbols]])
                merged_closes = merged_closes.reset_index()
            print(f"merged_closes count after applying live quotes: {len(merged_closes)}")
            print(merged_closes[['symbol', 'sector', 'close_last', 'close_prev', 'pChange']].head(3))

if __name__ == "__main__":
    run_verification()
