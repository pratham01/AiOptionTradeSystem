import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.advisory.application.agent.candlestick_pattern_agent import CandlestickPatternAgent

def main():
    engine = get_engine()
    agent = CandlestickPatternAgent()
    
    # 1. Fetch all distinct symbols in the daily database
    print("📥 Fetching active symbols from the database...")
    with engine.connect() as conn:
        symbols_df = pd.read_sql(text("""
            SELECT DISTINCT symbol 
            FROM ohlcv_daily 
            WHERE timestamp >= date('now', '-60 days')
            ORDER BY symbol ASC
        """), conn)
    
    symbols = symbols_df["symbol"].tolist()
    print(f"Loaded {len(symbols)} symbols. Starting pattern sweeps...")
    
    all_pattern_results = []
    
    # 2. Loop through symbols and calculate pattern stats
    processed = 0
    for sym in symbols:
        clean_sym = sym.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
        
        try:
            probs = agent.calculate_probabilities(sym)
            for pat_name, stats in probs.items():
                count = stats.get("count", 0)
                if count >= 3: # Only consider patterns that occurred at least 3 times in history
                    # Identify direction based on pattern name
                    direction = "LONG" if pat_name in ["Bullish Engulfing", "Hammer", "Morning Star"] else "SHORT" if pat_name in ["Bearish Engulfing", "Shooting Star", "Evening Star"] else "NEUTRAL"
                    
                    all_pattern_results.append({
                        "Symbol": clean_sym,
                        "Pattern": pat_name,
                        "Direction": direction,
                        "Occurrences": count,
                        "Win Rate": stats.get("win_rate", 0.0),
                        "Avg Next-Day Return": stats.get("avg_pnl", 0.0),
                        "t-stat": stats.get("t_stat", 0.0)
                    })
        except Exception as e:
            print(f"Error calculating stats for {clean_sym}: {e}")
            
        processed += 1
        if processed % 30 == 0:
            print(f"Processed {processed}/{len(symbols)} symbols...")
            
    print(f"Pattern sweep complete! Total valid stock-pattern setups: {len(all_pattern_results)}")
    
    if not all_pattern_results:
        print("No pattern setups met the criteria.")
        return
        
    df_res = pd.DataFrame(all_pattern_results)
    
    # 3. Filter and Sort to find the highest conviction edges
    # Standard statistical significance threshold is t-stat >= 2.0 (95% confidence)
    # We sort by win rate and t-statistic
    df_sig = df_res[df_res["t-stat"].abs() >= 1.5].copy() # Moderate to strong significance
    df_sig_sorted = df_sig.sort_values(by=["Win Rate", "t-stat"], ascending=[False, False])
    
    # Group results by symbol for the catalog section
    catalog_md = ""
    for symbol, grp in df_res.groupby("Symbol"):
        catalog_md += f"### {symbol}\n\n"
        catalog_md += "| Pattern | Direction | Occurrences | Win Rate | Avg Next-Day Return | t-statistic | Verdict |\n"
        catalog_md += "|:--------|:----------|:------------|:---------|:--------------------|:------------|:--------|\n"
        for _, row in grp.iterrows():
            is_sig = abs(row["t-stat"]) >= 2.0
            verdict = "✅ Significant Edge" if is_sig else "❌ Noise"
            # Format returns
            avg_ret = f"{row['Avg Next-Day Return']:+.2f}%"
            catalog_md += f"| {row['Pattern']} | {row['Direction']} | {row['Occurrences']} | {row['Win Rate']:.1f}% | {avg_ret} | {row['t-stat']:.3f} | {verdict} |\n"
        catalog_md += "\n"

    # 4. Write full report to markdown file
    markdown_report = f"""# 🕯️ F&O Candlestick Pattern Edge & Probability Catalog

This report cataloges the empirical win rates and mathematical edge for candlestick patterns across all F&O stocks in the database. 

* **Win Rate** indicates the % of times the price moved in the expected direction on the next trading day.
* **t-statistic** measures statistical significance. A value of **|t| >= 2.0** indicates a statistically verified edge (95%+ confidence), whereas lower values suggest the results could be random noise.

---

## 🏆 Top 25 Highest Conviction Candlestick Edges
Below are the top candlestick setups sorted by historical win rate, filtered for moderate-to-strong statistical significance (|t| >= 1.5):

| Symbol | Pattern | Direction | Occurrences | Win Rate | Avg Next-Day Return | t-statistic |
|:-------|:--------|:----------|:------------|:---------|:--------------------|:------------|
"""
    
    # Append top 25 setups
    for _, row in df_sig_sorted.head(25).iterrows():
        avg_ret = f"{row['Avg Next-Day Return']:+.2f}%"
        markdown_report += f"| {row['Symbol']} | {row['Pattern']} | {row['Direction']} | {row['Occurrences']} | {row['Win Rate']:.1f}% | {avg_ret} | {row['t-stat']:.3f} |\n"
        
    markdown_report += f"""
---

## 📚 Complete F&O Stock Candlestick Catalog
Below is the stock-by-stock breakdown of all candlestick pattern probabilities where the pattern has occurred at least 3 times in history:

{catalog_md}
"""
    
    report_path = "scratch/fo_candlestick_probabilities_report.md"
    with open(report_path, "w") as f:
        f.write(markdown_report)
        
    print(f"\n📝 Detailed F&O catalog report written to: {report_path}")

if __name__ == "__main__":
    main()
