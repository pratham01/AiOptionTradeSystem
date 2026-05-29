import sys
from pathlib import Path
import logging

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.application.agent.consolidation_screener_agent import ConsolidationScreenerAgent
from trade_system.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
LOGGER = logging.getLogger(__name__)

def main():
    LOGGER.info("Starting Volatility Contraction / Consolidation scan...")
    settings = Settings.load()
    agent = ConsolidationScreenerAgent(settings)
    # 5% percentile is quite strict, should yield a manageable watchlist of highly squeezed stocks
    agent.scan(bbw_percentile=0.05)
    LOGGER.info("Consolidation scan complete.")

if __name__ == "__main__":
    main()
