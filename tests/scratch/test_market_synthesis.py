import asyncio
import logging
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.advisory.application.agent.market_synthesis_agent import MarketSynthesisAgent
from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
from trade_system.shared.config import Settings

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("Testing MarketSynthesisAgent...")
    
    settings = Settings.load()
    llm_client = LlmAdvisorClient()
    
    agent = MarketSynthesisAgent(llm_client=llm_client, settings=settings)
    
    dummy_stats = {
        "total": 5,
        "wins": 3,
        "losses": 2
    }
    
    dummy_missed = [
        "Missed 3.2% rally on RELIANCE due to tight daily SL.",
        "Missed short trigger on INFY because volume delta was negative."
    ]
    
    dummy_correlation = "VIX increased by 4.2% while sector NIFTY_IT showed strong positive correlation (0.85) with NIFTY."
    
    dummy_skills = [
        "Dynamic ATR multiplier adaptation during high VIX regimes.",
        "Increased weight on CVD confirmation during opening range breakouts."
    ]
    
    dummy_patterns = [
        {"symbol": "NSE:RELIANCE-EQ", "patterns": ["Bullish Engulfing", "Inside Bar"]},
        {"symbol": "NSE:TCS-EQ", "patterns": ["Hammer"]}
    ]
    
    dummy_st_touches = {
        "15m": type('Obj', (object,), {'symbol': 'NSE:SBIN-EQ'})(),
        "Daily": type('Obj', (object,), {'symbol': 'NSE:HDFCBANK-EQ'})()
    }
    # Convert st_touches to lists of objects
    dummy_st_touches_processed = {
        "15m": [dummy_st_touches["15m"]],
        "Daily": [dummy_st_touches["Daily"]]
    }
    
    dummy_rotation = pd.DataFrame([
        {"Sector": "NIFTY IT", "Status": "LEADING"},
        {"Sector": "NIFTY FMCG", "Status": "LAGGING"}
    ])
    
    dummy_vcp = [
        {"symbol": "NSE:BHARTIAIRTEL-EQ", "bbw": 0.045, "close": 1150.0}
    ]
    
    report = await agent.generate_daily_synthesis(
        stats=dummy_stats,
        missed=dummy_missed,
        correlation=dummy_correlation,
        skills=dummy_skills,
        patterns=dummy_patterns,
        st_touches=dummy_st_touches_processed,
        rotation_df=dummy_rotation,
        vcp=dummy_vcp
    )
    
    print("\n" + "="*50)
    print("AI SWARM SYNTHESIS REPORT OUTPUT:")
    print("="*50)
    print(report)
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
