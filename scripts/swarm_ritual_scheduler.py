"""
Swarm Ritual Scheduler — Orchestrates the Post-Market Evolution.
Triggers the PostMarketImproverAgent every day at 15:45 IST.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

from trade_system.domains.advisory.application.agent.postmarket_improver_agent import PostMarketImproverAgent
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.shared.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/swarm_scheduler.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger("swarm_scheduler")

class SwarmRitualScheduler:
    def __init__(self, target_time: str = "15:45"):
        self.settings = Settings.load()
        self.auth_service = FyersAuthService(self.settings)
        self.target_time = dt_time.fromisoformat(target_time)
        self.is_running = True

    async def run(self):
        LOGGER.info(f"🚀 Swarm Ritual Scheduler Active. Target Time: {self.target_time.strftime('%H:%M')} IST")
        
        while self.is_running:
            try:
                now = datetime.now()
                
                # Check if it's a weekend
                if now.weekday() >= 5:
                    LOGGER.info("Weekend detected. Sleeping until Monday...")
                    # Sleep for an hour and check again
                    await asyncio.sleep(3600)
                    continue

                current_time = now.time()
                
                # Trigger window: if current time is >= 15:45 and we haven't run today
                if current_time >= self.target_time:
                    today_str = now.strftime('%Y%m%d')
                    lock_file = self.settings.data_dir / "locks" / f"swarm_ritual_done_{today_str}.txt"
                    
                    if not lock_file.exists():
                        LOGGER.info("🔔 Triggering Daily Post-Market Swarm Ritual...")
                        await self._execute_ritual()
                        
                        # Create persistent lock
                        lock_file.parent.mkdir(parents=True, exist_ok=True)
                        lock_file.write_text(f"Executed at {now.isoformat()}")
                        LOGGER.info("✅ Daily Ritual successfully completed and locked.")
                    else:
                        LOGGER.debug("Ritual already completed for today.")
                
                # Check every 60 seconds
                await asyncio.sleep(60)

            except Exception as e:
                LOGGER.exception(f"Scheduler encountered an error: {e}")
                await asyncio.sleep(300) # Wait 5 mins on error

    async def _execute_ritual(self):
        """Initializes and runs the post-market improver agent."""
        try:
            # 1. Ensure fresh token
            token = self.auth_service.get_valid_token()
            broker = FyersBroker(
                client_id=self.settings.fyers.client_id,
                access_token=token,
                user_id=self.settings.fyers.user_id,
                authenticator=self.auth_service.authenticator
            )
            
            # 2. Run Analysis
            orchestrator = PostMarketImproverAgent(broker=broker)
            await orchestrator.run_post_market_analysis()
            
        except Exception as e:
            LOGGER.error(f"Ritual execution failed: {e}")

if __name__ == "__main__":
    # Create logs dir if not exists
    Path("logs").mkdir(exist_ok=True)
    scheduler = SwarmRitualScheduler(target_time="15:45")
    asyncio.run(scheduler.run())
