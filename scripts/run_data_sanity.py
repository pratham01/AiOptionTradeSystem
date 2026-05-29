import sys
from pathlib import Path
import logging

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.agent.data_sanity_agent import DataSanityAgent
from trade_system.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)

def main():
    LOGGER.info("Starting database market data sanity check and heal...")
    settings = Settings.load()
    agent = DataSanityAgent(settings)
    
    # We require 100 days of history for Bollinger Bands and ATR indicators
    result = agent.ensure_data_sanity(min_candles=100)
    
    healed = result.get('healed_daily', [])
    LOGGER.info(f"Sanity check and heal complete. Total symbols healed: {len(healed)}")
    if healed:
        LOGGER.info(f"Healed symbols: {healed}")

if __name__ == "__main__":
    main()
