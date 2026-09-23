import pandas as pd
from datetime import time as dt_time, date
from trade_system.interfaces.dashboard.sector_scope_dashboard import fetch_target_date_and_data, get_sector_mapping

def test_all_lookbacks():
    latest_date_str, df, last_candle_ts = fetch_target_date_and_data("2026-09-22")
    assert not df.empty, "Dataframe should not be empty"
    target_date = pd.to_datetime(latest_date_str).date()
    fo_metadata = get_sector_mapping()
    df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))

    df_filtered = df[df['timestamp'].dt.date <= target_date].copy()
    df_filtered['trade_date'] = df_filtered['timestamp'].dt.date
    df_filtered['trade_time'] = df_filtered['timestamp'].dt.time

    lookbacks = [
        ("🌅 Full Day", None),
        ("⚡ Last 15 Mins", 1),
        ("🚀 Last 30 Mins", 2),
        ("⏱️ Last 1 Hour", 4),
        ("⏱️ Last 2 Hours", 8),
        ("🍱 Since 12:30 PM (Midday)", "midday_1230"),
    ]

    results = {}

    for label, step in lookbacks:
        rows = []
        grouped = df_filtered[~df_filtered['symbol'].str.contains("INDEX")].groupby('symbol')

        for symbol, grp in grouped:
            grp_sorted = grp.sort_values('timestamp')
            today_candles = grp_sorted[grp_sorted['trade_date'] == target_date]
            if today_candles.empty:
                continue

            close_last = float(today_candles.iloc[-1]['close'])
            close_prev = None
            pchange = None
            window_candles = today_candles

            if step == "midday_1230":
                midday_bars = today_candles[today_candles['trade_time'] >= dt_time(12, 30)]
                if not midday_bars.empty:
                    ref_bar = midday_bars.iloc[0]
                    close_prev = float(ref_bar['open'])
                    window_candles = today_candles[today_candles['timestamp'] >= ref_bar['timestamp']]
                else:
                    ref_bar = today_candles.iloc[-1]
                    close_prev = float(ref_bar['open'])
                    window_candles = today_candles.tail(1)
                if close_prev > 0 and close_last > 0:
                    pchange = ((close_last - close_prev) / close_prev) * 100
            elif step is not None:
                if len(today_candles) > step:
                    ref_idx = -step - 1
                    close_prev = float(today_candles.iloc[ref_idx]['close'])
                else:
                    close_prev = float(today_candles.iloc[0]['open'])
                window_candles = today_candles.tail(step + 1)
                if close_prev > 0 and close_last > 0:
                    pchange = ((close_last - close_prev) / close_prev) * 100
            else:
                prev_candles = grp_sorted[grp_sorted['timestamp'].dt.date < target_date]
                if not prev_candles.empty:
                    close_prev = float(prev_candles.iloc[-1]['close'])
                else:
                    close_prev = float(grp_sorted.iloc[0]['close'])
                if close_prev > 0 and close_last > 0:
                    pchange = ((close_last - close_prev) / close_prev) * 100
                window_candles = today_candles

            if pchange is not None:
                w_high = float(window_candles['high'].max())
                w_low = float(window_candles['low'].min())
                move_from_low = ((close_last - w_low) / w_low * 100) if w_low > 0 else 0.0
                drop_from_high = ((w_high - close_last) / w_high * 100) if w_high > 0 else 0.0

                rows.append({
                    "symbol": symbol,
                    "close_last": close_last,
                    "close_prev": close_prev,
                    "pchange": pchange,
                    "move_from_low": move_from_low,
                    "drop_from_high": drop_from_high,
                    "window_bars": len(window_candles)
                })

        res_df = pd.DataFrame(rows)
        results[label] = res_df
        top_gainers = res_df.sort_values('pchange', ascending=False).head(3)
        top_losers = res_df.sort_values('pchange', ascending=True).head(3)

        print(f"\n=======================================================")
        print(f"📊 LOOKBACK: {label} (Evaluated {len(res_df)} symbols)")
        print(f"=======================================================")
        print("  🟢 Top 3 Gainers:")
        for _, r in top_gainers.iterrows():
            print(f"     {r['symbol']:<15} Return: {r['pchange']:+6.2f}% | Prev: {r['close_prev']:.2f} -> Last: {r['close_last']:.2f} | Bars: {r['window_bars']}")
        print("  🔴 Top 3 Losers:")
        for _, r in top_losers.iterrows():
            print(f"     {r['symbol']:<15} Return: {r['pchange']:+6.2f}% | Prev: {r['close_prev']:.2f} -> Last: {r['close_last']:.2f} | Bars: {r['window_bars']}")

    # Validation assertions across windows
    # Verify that returns change across windows
    full_day_df = results["🌅 Full Day"].set_index("symbol")
    m15_df = results["⚡ Last 15 Mins"].set_index("symbol")
    m30_df = results["🚀 Last 30 Mins"].set_index("symbol")
    h1_df = results["⏱️ Last 1 Hour"].set_index("symbol")
    midday_df = results["🍱 Since 12:30 PM (Midday)"].set_index("symbol")

    # Pick common symbol
    sample_sym = "NSE:RBLBANK-EQ"
    print("\n-------------------------------------------------------")
    print(f"🔍 Comparative Deep-Dive for {sample_sym}:")
    print(f"   Full Day Return:          {full_day_df.loc[sample_sym, 'pchange']:+6.2f}% (Window bars: {full_day_df.loc[sample_sym, 'window_bars']})")
    print(f"   Since 12:30 PM:           {midday_df.loc[sample_sym, 'pchange']:+6.2f}% (Window bars: {midday_df.loc[sample_sym, 'window_bars']})")
    print(f"   Last 2 Hours Return:      {results['⏱️ Last 2 Hours'].set_index('symbol').loc[sample_sym, 'pchange']:+6.2f}%")
    print(f"   Last 1 Hour Return:       {h1_df.loc[sample_sym, 'pchange']:+6.2f}% (Window bars: {h1_df.loc[sample_sym, 'window_bars']})")
    print(f"   Last 30 Mins Return:      {m30_df.loc[sample_sym, 'pchange']:+6.2f}% (Window bars: {m30_df.loc[sample_sym, 'window_bars']})")
    print(f"   Last 15 Mins Return:      {m15_df.loc[sample_sym, 'pchange']:+6.2f}% (Window bars: {m15_df.loc[sample_sym, 'window_bars']})")
    print("-------------------------------------------------------")

    assert m15_df.loc[sample_sym, 'window_bars'] == 2, "15m should have 2 bars (t-1 and t)"
    assert m30_df.loc[sample_sym, 'window_bars'] == 3, "30m should have 3 bars"
    assert h1_df.loc[sample_sym, 'window_bars'] == 5, "1h should have 5 bars"
    assert midday_df.loc[sample_sym, 'window_bars'] == 12, "12:30 PM should have 12 afternoon bars (12:30 to 15:15)"
    print("✅ All lookback window assertions verified successfully!")

if __name__ == "__main__":
    test_all_lookbacks()
