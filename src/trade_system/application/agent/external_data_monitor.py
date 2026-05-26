from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from trade_system.application.agent.website_dashboard_agent import WebsiteDashboardAgent, load_config
from trade_system.notifications.telegram import TelegramNotifier
from trade_system.config import Settings

LOGGER = logging.getLogger(__name__)

class ExternalDataMonitorAgent:
    """
    Agent that continuously monitors an external website and shares its findings
    with the rest of the agentic system via Shared Memory.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.storage_path = Path("data/external_signals.json")
        self.settings = Settings.load()
        self.notifier = TelegramNotifier(self.settings.telegram.bot_token, self.settings.telegram.chat_id)
        self.shutdown = False

    async def run_forever(self, interval_seconds: int = 300):
        """Main loop that polls the website periodically."""
        LOGGER.info(f"Starting continuous website monitor for {self.config_path}")
        
        while not self.shutdown:
            try:
                # 1. Run the Scraping Agent
                config = load_config(self.config_path)
                agent = WebsiteDashboardAgent(config)
                
                LOGGER.info(f"Polling {config.url}...")
                result = agent.run()
                
                if result.get("status") == "ok":
                    # 2. Update Shared Memory
                    self._update_shared_memory(result["extracted"])
                    LOGGER.info("Shared memory updated with external signals.")
                else:
                    LOGGER.error(f"Failed to fetch external data: {result.get('error')}")

            except Exception as e:
                LOGGER.error(f"Error in external monitor loop: {e}")

            # Wait for next interval
            await asyncio.sleep(interval_seconds)

    def _update_shared_memory(self, data: dict[str, str]):
        """Persists extracted data to the JSON signal store."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        existing = {}
        if self.storage_path.exists():
            try:
                existing = json.loads(self.storage_path.read_text())
            except:
                pass
        
        # Merge new data with timestamp
        payload = {
            "last_updated": datetime.now().isoformat(),
            "signals": data
        }
        
        # Merge if multiple sources exist (keyed by URL/Config)
        source_key = self.config_path.stem
        existing[source_key] = payload
        
        self.storage_path.write_text(json.dumps(existing, indent=2))

    def stop(self):
        self.shutdown = True

if __name__ == "__main__":
    # Example usage (standalone)
    logging.basicConfig(level=logging.INFO)
    # This would be triggered by a specific config file for tradefinder.in etc.
    # monitor = ExternalDataMonitorAgent("config/tradefinder_monitor.json")
    # asyncio.run(monitor.run_forever())
    pass
