import logging
import sys
from .config import Settings
from .logging_utils import configure_logging
from .brokers.fyers import FyersBroker
from .brokers.fyers_auth import FyersAuthenticator
from .indicators.supertrend import SupertrendIndicator
from .data.manager import DataManager
from .notifications.telegram import TelegramNotifier
from .live.engine import TradeEngine

def main():
    # 1. Load configuration and setup logging
    settings = Settings.load()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Initializing Trade System (Industry Grade)...")

    # 2. Initialize components
    # Add authenticator for auto-token refresh
    authenticator = FyersAuthenticator(settings)
    
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=settings.fyers.access_token,
        user_id=settings.fyers.user_id,
        authenticator=authenticator
    )

    # Use settings for Supertrend
    supertrend = SupertrendIndicator(
        period=settings.indicator_config.supertrend_period,
        multiplier=settings.indicator_config.supertrend_multiplier
    )

    data_manager = DataManager(settings.data_dir)

    notifier = TelegramNotifier(
        token=settings.telegram.bot_token,
        chat_id=settings.telegram.chat_id
    )

    # 3. Initialize engine
    engine = TradeEngine(
        broker=broker,
        indicators=[supertrend],
        data_manager=data_manager,
        notifier=notifier,
        settings=settings
    )

    # 4. Run the engine
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Stopping...")
        engine.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
