import sys, os
sys.path.insert(0, 'src')

import asyncio
import pandas as pd
from trade_system.application.agent.option_research_agent import OptionResearchAgent

async def main():
    agent = OptionResearchAgent()
    symbols = [
        "NSE:NIFTY50-INDEX",
        "NSE:HDFCBANK-EQ",
        "NSE:ICICIBANK-EQ",
        "NSE:AXISBANK-EQ",
        "NSE:HCLTECH-EQ"
    ]
    
    # Define backtest configurations: (indicator, threshold)
    configs = [
        ("Momentum Surge", 1.5),
        ("Volume Ratio", 2.0),
        ("RSI Oversold", 35.0),
        ("RSI Overbought", 65.0),
        ("High-Volume Breakout", 2.0)
    ]
    
    results = []
    for symbol in symbols:
        for indicator, threshold in configs:
            print(f"Running {indicator} (threshold={threshold}) for {symbol}...")
            res = await agent.run_predictability_backtest(symbol, indicator, threshold)
            if res["success"]:
                results.append({
                    "Symbol": symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", ""),
                    "Indicator": indicator,
                    "Threshold": f"{threshold}",
                    "Total Days": res["total_days_analyzed"],
                    "Signals Triggered": res["condition_met_count"],
                    "Win Rate (Signal)": f"{res['win_rate_when_signal']*100:.1f}%",
                    "Win Rate (Baseline)": f"{res['win_rate_baseline']*100:.1f}%",
                    "Avg Return (Signal)": f"{res['avg_return_when_signal']:+.3f}%",
                    "Avg Return (Baseline)": f"{res['avg_return_baseline']:+.3f}%",
                    "t-stat": f"{res['t_stat']:.3f}",
                    "p-value": f"{res['p_value']:.4f}",
                    "Predictable Edge": "✅ YES" if res["is_significant"] else "❌ NO"
                })
                
    df = pd.DataFrame(results)
    
    # Save report
    md_content = f"""# 📊 Quantitative Return Predictability Backtest Report
    
**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Data Universe**: Selected Index and Major F&O stocks from the SQLite database.  

This backtest evaluates whether the close-to-close direction of the next trading day is statistically predictable based on different technical indicator signals today. We run a two-sample independent t-test to compare the mean return on signal days versus baseline days.

## 🔬 Statistical Edge Results

{df.to_markdown(index=False)}

---
## 🧠 Key Takeaways
1. **t-statistic & p-value**: A t-statistic with absolute value >= 2.0 (or p-value < 0.05) indicates that the difference in mean returns is statistically significant and not due to random walk.
2. **RSI Oversold / Overbought**: These reversion conditions often show the most significant predictability on indices (like NIFTY50) due to mean-reverting index dynamics.
3. **Volume Breakouts**: High volume ratios usually represent smart money entry. Comparing next-day returns after volume surges helps confirm if they represent follow-through or exhaustion.
"""
    
    report_path = "scratch/predictability_backtest_report.md"
    with open(report_path, "w") as f:
        f.write(md_content)
    print(f"📝 Report saved successfully to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
