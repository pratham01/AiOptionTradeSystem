import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import os
import time
import logging
from datetime import datetime, time as dt_time

from trade_system.config import Settings
from trade_system.utils.logging_utils import configure_logging
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.brokers.factory import get_broker_manager
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.infrastructure.data.manager import DataManager
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.interfaces.live.market import MarketStatus
from trade_system.interfaces.live.engine import TradeEngine

def wait_for_market_hours():
    """Waits until it's 9:00 AM on a weekday before starting the main system."""
    logger = logging.getLogger("AutomationWrapper")
    
    while True:
        now = datetime.now()
        
        # 1. Skip weekends
        if now.weekday() >= 5:
            logger.info("Today is a weekend. Skipping.")
            return False

        # 2. Check market window (9:00 AM start, 3:30 PM end)
        start_time = dt_time(9, 0)
        end_time = dt_time(15, 30)
        current_time = now.time()

        if current_time > end_time:
            logger.info(f"Market session already ended for today ({current_time}). Terminating.")
            return False
            
        if current_time >= start_time:
            logger.info(f"Within market hours ({current_time}). Starting system...")
            return True

        # 3. Wait until 9:00 AM
        logger.info(f"Current time is {current_time}. Waiting until 9:00 AM...")
        time.sleep(60) # Check every minute

def     start_automated_system():
    # 1. Setup logging and settings
    settings = Settings.load()
    configure_logging(settings.log_level)
    logger = logging.getLogger("AutomationWrapper")

    # 2. Enforce start window
    if not wait_for_market_hours():
        return

    # 3. Initialize components
    authenticator = FyersAuthenticator(settings)
    broker = FyersBroker(
        client_id=settings.fyers_client_id,
        access_token=settings.fyers_access_token,
        user_id=settings.fyers_user_id,
        authenticator=authenticator
    )
    
    # 4. Market Status Check
    market_checker = MarketStatus(broker)
    logger.info("Market status check initiated.")
    
    if not market_checker.wait_for_market_open():
        logger.warning("Market did not open within time limits (possibly a holiday). Terminating.")
        return

    # 5. Initialize Engine Components
    supertrend = SupertrendIndicator(
        period=settings.indicator_config.supertrend_period,
        multiplier=settings.indicator_config.supertrend_multiplier
    )
    data_manager = DataManager(settings.data_dir)
    notifier = TelegramNotifier(
        token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id
    )

    # 6. Initialize and Run Engine
    engine = TradeEngine(
        broker=broker,
        indicators=[supertrend],
        data_manager=data_manager,
        notifier=notifier,
        settings=settings
    )

    logger.info("Executing Trade System Main Loop...")
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Stopping...")
        engine.stop()
    except Exception as e:
        logger.error(f"Automation Wrapper caught an error: {e}", exc_info=True)

if __name__ == "__main__":
    start_automated_system()
