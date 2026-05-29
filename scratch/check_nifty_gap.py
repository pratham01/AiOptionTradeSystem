import pandas as pd
from trade_system.application.indicators.supertrend import calculate_supertrend
from trade_system.interfaces.live.helpers import get_gap_adjusted_data

def check_gap():
    df = pd.read_csv("data/NSE_NIFTY50-INDEX_3min_2026.csv", parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    today = pd.Timestamp("2026-05-27").date()
    
    # Calculate Supertrend WITH gap adjustment (like collector.py does)
    adjusted = get_gap_adjusted_data(df, 7, today)
    st_df_adj = calculate_supertrend(adjusted, period=7, multiplier=3)
    today_adj = st_df_adj[st_df_adj.index.date == today]
    
    # Calculate Supertrend WITHOUT gap adjustment (like our check_nifty_trend.py did)
    df_today = df[df.index.date == today]
    st_df_noadj = calculate_supertrend(df_today, period=7, multiplier=3)
    
    print("\n--- WITH GAP ADJUSTMENT ---")
    print(today_adj[["open", "high", "low", "close", "supertrend", "supertrend_direction"]].loc["2026-05-27 10:09:00":"2026-05-27 10:21:00"])
    
    print("\n--- WITHOUT GAP ADJUSTMENT ---")
    print(st_df_noadj[["open", "high", "low", "close", "supertrend", "supertrend_direction"]].loc["2026-05-27 10:09:00":"2026-05-27 10:21:00"])

if __name__ == "__main__":
    check_gap()
