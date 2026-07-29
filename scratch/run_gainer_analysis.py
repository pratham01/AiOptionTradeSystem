import sys, os
sys.path.insert(0, 'src')

import asyncio
from trade_system.domains.advisory.application.agent.top_gainer_analysis_agent import TopGainerAnalysisAgent

async def main():
    print("🚀 Initializing Top Gainer Analysis Agent...")
    agent = TopGainerAnalysisAgent()
    
    # Run analysis for latest date in database
    result = await agent.analyze_top_gainers(top_n=5)
    
    if result["success"]:
        print("\n✅ Top Gainers Analysis Report generated successfully!")
        print(f"Report saved to: {result['report_path']}")
        print(f"Also saved to: scratch/daily_gainer_analysis_report.md\n")
        print("="*80)
        print("📋 TOP GAINERS SUMMARY")
        print("="*80)
        for g in result["gainers"]:
            print(f"#{g['Rank']} {g['Symbol']} ({g['Change %']}) | Price: {g['Close Price']} | Volume Ratio: {g['Volume Ratio']} | RSI: {g['RSI (14)']} | ADX: {g['ADX (14)']}")
            print(f"   💬 Reason: {g['Reason for Move']}\n")
        print("="*80)
    else:
        print(f"❌ Error during gainer analysis: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
