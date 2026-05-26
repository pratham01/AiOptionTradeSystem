import asyncio
import logging
from trade_system.application.agent.orchestrator import TradeOrchestrator

# Setup basic logging to see agent activity
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

async def main():
    print("🚀 Initializing TradeOrchestrator for Verification...")
    
    # We pass broker=None to see if it can initialize without a live connection
    # Most agents should initialize, but might fail during run_session if they need live data.
    try:
        orc = TradeOrchestrator(broker=None)
        print("✅ TradeOrchestrator initialized successfully.")
        
        # Test a small piece of the pipeline if possible
        # For example, check if agents are present
        print(f"Agents in Swarm: {[type(a).__name__ for a in [orc.premarket_agent, orc.market_context_agent, orc.nifty_agent]]}")
        
    except Exception as e:
        print(f"❌ Failed to initialize TradeOrchestrator: {e}")

if __name__ == "__main__":
    asyncio.run(main())
