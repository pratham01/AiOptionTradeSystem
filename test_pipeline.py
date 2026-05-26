import asyncio
import logging
from trade_system.application.agent.orchestrator import TradeOrchestrator
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.config import Settings

logging.basicConfig(level=logging.INFO)

async def test_full_pipeline():
    print("🚀 Running Integrated Pipeline (ST + RSI + Sector Rotation)...")
    
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.read_cached_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id
    )
    
    orc = TradeOrchestrator(broker=broker)
    # Trigger a session run (without notifying telegram for test)
    orc.notifier = None 
    
    print("🧠 Swarm is thinking...")
    plan = await orc.run_session()
    
    print("\n" + "="*50)
    print("📋 TOP AI SUGGESTIONS (RANKED BY CONFLUENCE)")
    print("="*50)
    for i, s in enumerate(plan.all_suggestions(), 1):
        print(f"{i}. {s.symbol} | Dir: {s.direction.value} | Conf: {s.confidence:.0%}")
        print(f"   Note: {s.narrative}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
