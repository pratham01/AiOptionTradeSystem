import sys
import logging
import asyncio
from pathlib import Path
from datetime import date, timedelta

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.infrastructure.data.history import HistoricalDataService
from trade_system.infrastructure.data.storage import CsvDataCatalog
from trade_system.infrastructure.data.fo_universe import get_fo_universe

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

async def update_all_history():
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.read_cached_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id
    )
    
    catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
    history_service = HistoricalDataService(broker, catalog)
    
    universe = get_fo_universe()
    LOGGER.info(f"Updating history for {len(universe)} F&O stocks...")
    
    # Update Daily and 5m data for the last 5 days to ensure fresh context
    end_date = date.today()
    start_date = end_date - timedelta(days=5)
    
    for symbol in universe:
        try:
            LOGGER.info(f"Updating {symbol}...")
            # Daily
            history_service.collect(symbol, "D", start_date, end_date)
            # 15-minute
            history_service.collect(symbol, "15", start_date, end_date)
            # 5-minute
            history_service.collect(symbol, "5", start_date, end_date)
            # Sleep briefly to respect rate limits
            await asyncio.sleep(0.2)
        except Exception as e:
            LOGGER.error(f"Failed to update {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(update_all_history())
