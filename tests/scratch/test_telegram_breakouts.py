import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging
logging.basicConfig(level=logging.INFO)

from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
from trade_system.interfaces.live import LiveMarketDataService
from trade_system.domains.market_data.infrastructure.data import CsvDataCatalog

def test_telegram():
    print("Loading settings...")
    settings = Settings.load()
    
    # Initialize broker and catalog
    from trade_system.domains.trading.infrastructure.brokers.legacy import FyersBrokerClient
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=settings.fyers.access_token,
        user_id=settings.fyers.user_id
    )
    catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
    
    print("Initializing LiveMarketDataService...")
    service = LiveMarketDataService(
        broker=broker,
        catalog=catalog,
        symbols=["NSE:NIFTY50-INDEX"],
        settings=settings
    )
    
    # Clear the last_breakout_alerts cache to force sending
    service.last_breakout_alerts.clear()
    
    # Mock 12 breakouts
    mock_breakouts = []
    for i in range(12):
        direction = "LONG" if i % 2 == 0 else "SHORT"
        mock_breakouts.append({
            "symbol": f"NSE:MOCKSTOCK{i}-EQ",
            "sector": "MOCK_SECTOR",
            "direction": direction,
            "close": 1000.0 + i * 10,
            "orb_high": 990.0,
            "orb_low": 980.0,
            "volume": 50000,
            "vol_sma": 10000
        })
        
    print(f"Calling _send_breakout_alerts with {len(mock_breakouts)} mocked alerts...")
    try:
        service._send_breakout_alerts(mock_breakouts)
        print("Alerts function finished execution. Check your Telegram group to verify receipt of multiple small messages!")
    except Exception as e:
        print(f"ERROR: Failed to send alerts: {e}")

if __name__ == "__main__":
    test_telegram()
