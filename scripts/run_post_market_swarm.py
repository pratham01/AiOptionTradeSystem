"""
Script to manually trigger the fully autonomous AI-Native Post-Market Swarm ritual.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.advisory.application.agent.postmarket_improver_agent import PostMarketImproverAgent
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.shared.config import Settings

async def run_swarm_ritual():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("🚀 Triggering AI-Native Post-Market Swarm Ritual...")
    
    settings = Settings.load()
    auth_service = FyersAuthService(settings)
    token = auth_service.get_valid_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth_service.authenticator
    )
    
    orchestrator = PostMarketImproverAgent(broker=broker)
    
    # Run the ritual
    result = await orchestrator.run_post_market_analysis()
    
    print("\n" + "="*50)
    print("🏁 SWARM RITUAL COMPLETE")
    print("="*50)
    print(f"Stats: {result['stats']}")
    print(f"Missed: {len(result['missed'])} setups identified.")
    print(f"Skills: {len(result['skills'])} new skills created.")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_swarm_ritual())
