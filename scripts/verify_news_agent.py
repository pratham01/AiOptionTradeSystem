"""
Direct verification of the PreMarketNewsAgent with the new MoneyControl scraper.
"""
import asyncio
import logging
from trade_system.domains.advisory.application.agent.premarket_news_agent import PreMarketNewsAgent

logging.basicConfig(level=logging.INFO)

async def verify_agent():
    print("🚀 Initializing PreMarketNewsAgent with Scraper...")
    agent = PreMarketNewsAgent()
    
    print("🔍 Running analysis (Scraping MoneyControl & ET headlines)...")
    # Pass dummy numbers but no news to force scraping
    brief = await agent.analyze(usd_inr=83.5, crude_oil=82.1, fii_dii_net=-450.0)
    
    print("\n" + "="*50)
    print("📡 SCRAPED HEADLINES DETECTED:")
    print("="*50)
    for i, news in enumerate(brief.top_news[:10], 1):
        print(f"{i}. {news}")
    
    print("\n" + "="*50)
    print("🤖 AI SYNTHESIS (FALLBACK IF NO LLM):")
    print("="*50)
    print(f"Sentiment: {brief.overall_sentiment}")
    print(f"Summary: {brief.agent_summary}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(verify_agent())
