import pandas as pd
import numpy as np
from trade_system.interfaces.dashboard.sector_scope_dashboard import fetch_target_date_and_data, get_sector_mapping

def test_rrg_calculation():
    latest_date_str, df, last_candle_ts = fetch_target_date_and_data("2026-09-22")
    fo_metadata = get_sector_mapping()
    df['sector'] = df['symbol'].apply(lambda s: fo_metadata.get(s, "UNKNOWN"))

    target_date = pd.to_datetime(latest_date_str).date()
    today_df = df[df['timestamp'].dt.date == target_date].sort_values('timestamp')

    # Benchmark: NIFTY 50
    bench_df = today_df[today_df['symbol'].str.contains("NIFTY50|NIFTY 50", case=False)]
    if bench_df.empty:
        print("Nifty 50 not found directly, using composite mean")
        bench_ts = today_df.groupby('timestamp')['close'].pct_change().fillna(0)
    else:
        bench_df = bench_df.sort_values('timestamp')
        bench_base = bench_df.iloc[0]['open']
        bench_ret = (bench_df.set_index('timestamp')['close'] - bench_base) / bench_base * 100

    # Sector time series
    sectors = [s for s in today_df['sector'].unique() if s != "UNKNOWN"]
    print(f"Testing RRG for {len(sectors)} sectors against benchmark...")

    results = []
    for sec in sectors:
        sec_df = today_df[today_df['sector'] == sec]
        # Aggregate sector close return per timestamp
        sec_piv = sec_df.groupby('timestamp')['close'].mean()
        sec_base = sec_df.groupby('timestamp')['open'].mean().iloc[0]
        sec_ret = (sec_piv - sec_base) / sec_base * 100

        # Relative Strength = Sector Return - Benchmark Return
        aligned = pd.DataFrame({'sector': sec_ret, 'bench': bench_ret}).dropna()
        if len(aligned) < 3:
            continue

        # RS-Ratio: normalized around 100
        # StockMojo / JdK: RS-Ratio = 100 + (RS - SMA(RS)) / Std(RS) * factor
        # For intraday: RS-Ratio = 100 + (sector_ret - bench_ret) * 2.5
        rs_diff = aligned['sector'] - aligned['bench']
        rs_ratio_series = 100 + rs_diff * 2.0
        
        # RS-Momentum: Rate of change of RS-Ratio
        rs_mom_series = 100 + (rs_ratio_series - rs_ratio_series.shift(2)).fillna(0) * 1.5

        latest_ratio = round(rs_ratio_series.iloc[-1], 2)
        latest_mom = round(rs_mom_series.iloc[-1], 2)

        if latest_ratio >= 100 and latest_mom >= 100:
            quadrant = "LEADING"
        elif latest_ratio >= 100 and latest_mom < 100:
            quadrant = "WEAKENING"
        elif latest_ratio < 100 and latest_mom < 100:
            quadrant = "LAGGING"
        else:
            quadrant = "IMPROVING"

        results.append({
            "sector": sec,
            "rs_ratio": latest_ratio,
            "rs_momentum": latest_mom,
            "quadrant": quadrant,
            "sector_ret": round(aligned['sector'].iloc[-1], 2),
            "bench_ret": round(aligned['bench'].iloc[-1], 2),
        })

    res_df = pd.DataFrame(results).sort_values('rs_ratio', ascending=False)
    print("\n--- SECTOR RRG CLASSIFICATION ---")
    print(res_df.to_string())

if __name__ == "__main__":
    test_rrg_calculation()
