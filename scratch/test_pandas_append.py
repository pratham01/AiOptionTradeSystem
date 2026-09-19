import pandas as pd

df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=[pd.Timestamp("2026-09-18 11:00"), pd.Timestamp("2026-09-18 11:15")])
live_pchanges = pd.Series({"A": 10, "B": 20, "C": 30})

live_time = pd.Timestamp.now().round("T")
aligned_live = live_pchanges.reindex(df.columns).fillna(0.0)
df.loc[live_time] = aligned_live

print(df)
