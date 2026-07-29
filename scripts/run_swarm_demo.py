import asyncio
import logging
from trade_system.domains.advisory.application.agent.orchestrator import TradeOrchestrator
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.shared.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

async def run_deliberation():
    print("="*60)
    print("🐝 EXECUTING LIVE SWARM DELIBERATION: NIFTY 50")
    print("="*60)
    
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.read_cached_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator
    )
    
    orc = TradeOrchestrator(broker=broker)
    
    print("📡 Swarm Agents are communicating in parallel...")
    print("   [Agent 1] PreMarketNewsAgent: Scraping business sentiment...")
    print("   [Agent 2] OptionChainAgent: Calculating strike-wise OI intensity...")
    print("   [Agent 3] CandidateScreener: Building technical structure (VWAP/ST/Fib)...")
    print("   [Agent 4] TradeOrchestrator: Synthesizing verdict...")
    print("-" * 60)
    
    # Run the orchestrator session
    plan = await orc.run_session()
    
    # Display Results
    print("\n🏁 SWARM VERDICT FINALIZED")
    print("="*60)
    
    # Get Nifty suggestion
    nifty_plan = next((s for s in plan.all_suggestions() if "NIFTY50" in s.symbol), None)
    
    if nifty_plan:
        print(f"📊 SYMBOL: {nifty_plan.symbol}")
        print(f"🎯 DIRECTION: {nifty_plan.direction.value}")
        print(f"🔥 CONFIDENCE: {nifty_plan.confidence:.0%}")
        print(f"📖 RATIONALE: {nifty_plan.narrative}")
        print(f"🛠️ TAGS: {', '.join(nifty_plan.tags) if nifty_plan.tags else 'N/A'}")
    else:
        print("❌ SWARM VERDICT: 'NO TRADE' for NIFTY at this moment.")
        print("   Reason: Insufficient confluence between News, OI, and Technicals.")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_deliberation())
