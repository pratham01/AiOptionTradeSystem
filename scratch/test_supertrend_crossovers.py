import pandas as pd
from trade_system.application.indicators import calculate_supertrend
from trade_system.interfaces.live.collector import resample_to_timeframe

def main():
    # Load 3-minute Nifty 50 bars
    csv_path = "data/fo_historical/NSE_NIFTY50-INDEX_3min_2026.csv"
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    
    # 1. 3m Supertrend (Trend timeframe)
    st_3m = calculate_supertrend(df, period=7, multiplier=3)
    
    # 2. 15m resampled Supertrend (Confirmed timeframe)
    df_15m = resample_to_timeframe(df, 15)
    st_15m = calculate_supertrend(df_15m, period=7, multiplier=3)
    
    print("--- 3m Supertrend crossovers today ---")
    st_3m["prev_direction"] = st_3m["supertrend_direction"].shift(1)
    crossovers_3m = st_3m[(st_3m["supertrend_direction"] != st_3m["prev_direction"]) & (st_3m.index.date == pd.Timestamp("2026-05-29").date())]
    if crossovers_3m.empty:
        print("No 3m crossovers today.")
    else:
        for idx, row in crossovers_3m.iterrows():
            print(f"3m Crossover at {idx}: Flipped to {row['supertrend_direction']} | Close: {row['close']} | ST: {row['supertrend']}")
            
    print("\n--- 15m Supertrend crossovers today ---")
    st_15m["prev_direction"] = st_15m["supertrend_direction"].shift(1)
    crossovers_15m = st_15m[(st_15m["supertrend_direction"] != st_15m["prev_direction"]) & (st_15m.index.date == pd.Timestamp("2026-05-29").date())]
    if crossovers_15m.empty:
        print("No 15m crossovers today.")
    else:
        for idx, row in crossovers_15m.iterrows():
            print(f"15m Crossover at {idx}: Flipped to {row['supertrend_direction']} | Close: {row['close']} | ST: {row['supertrend']}")

if __name__ == "__main__":
    main()
