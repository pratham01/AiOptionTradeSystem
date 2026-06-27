import pandas as pd
from trade_system.application.indicators.supertrend import calculate_supertrend

def check_csv():
    df = pd.read_csv("data/NSE_NIFTY50-INDEX_3min_2026.csv", parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    today = "2026-05-27"
    df_today = df[df.index.date == pd.Timestamp(today).date()]
    print(f"Total bars today in CSV: {len(df_today)}")
    
    st_df = calculate_supertrend(df_today, period=7, multiplier=3)
    print("\nBars around 10:15:")
    print(st_df[["open", "high", "low", "close", "supertrend", "supertrend_direction"]].loc["2026-05-27 10:00:00":"2026-05-27 10:30:00"])

if __name__ == "__main__":
    check_csv()
