import sys
import logging
import asyncio
from pathlib import Path
from datetime import date

# Add src to path
root_path = Path(__file__).parent.parent
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient

logging.basicConfig(level=logging.INFO)

async def main():
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
    )
    
    # Try fetching KALYANKJIL for different dates and resolutions
    print("--- Fetching daily for KALYANKJIL ---")
    try:
        df_d = broker.fetch_history(
            symbol="NSE:KALYANKJIL-EQ",
            resolution="D",
            range_from="2026-06-12",
            range_to="2026-06-21",
        )
        print("Daily results shape:", df_d.shape)
        if not df_d.empty:
            print(df_d.tail(5))
    except Exception as e:
        print("Daily error:", e)
        
    print("\n--- Fetching 15m for KALYANKJIL ---")
    try:
        df_15 = broker.fetch_history(
            symbol="NSE:KALYANKJIL-EQ",
            resolution="15",
            range_from="2026-06-12",
            range_to="2026-06-21",
        )
        print("15m results shape:", df_15.shape)
        if not df_15.empty:
            print(df_15.tail(5))
    except Exception as e:
        print("15m error:", e)

if __name__ == "__main__":
    asyncio.run(main())
