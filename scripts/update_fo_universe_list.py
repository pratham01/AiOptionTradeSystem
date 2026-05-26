import sys
import json
import logging
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def update_fo_universe():
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.read_cached_token()
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id
    )
    
    LOGGER.info("Fetching F&O universe from Fyers...")
    # Logic to get all F&O stocks. 
    # Usually we can get this by checking the symbol CSV or a specific API call.
    # For now, let's manually add some major ones and then look for a way to automate.
    
    config_path = root_path / "config" / "fo_universe.json"
    with open(config_path, "r") as f:
        universe = json.load(f)
        
    # Add SAMMAANCAP
    if "NSE:SAMMAANCAP-EQ" not in universe:
        universe["NSE:SAMMAANCAP-EQ"] = "FINANCE"
        LOGGER.info("Added SAMMAANCAP to universe.")

    # More missing ones
    # ...
    
    with open(config_path, "w") as f:
        json.dump(universe, f, indent=4)
    
    LOGGER.info(f"Updated F&O universe to {len(universe)} stocks.")

if __name__ == "__main__":
    update_fo_universe()
