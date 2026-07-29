import logging
from datetime import datetime, time
import time as sleep_timer
from zoneinfo import ZoneInfo
from trade_system.domains.trading.domain.ports.broker import Broker as BaseBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import _parse_nse_market_status

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

class MarketStatus:
    def __init__(self, broker: BaseBroker):
        self.broker = broker

    def is_market_open(self) -> bool:
        """
        Checks if the market is currently open for trading.
        Considers weekends, time of day, and Fyers API status.
        """
        now = datetime.now(IST)
        
        # 1. Check if it's a weekend (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            logger.info("Market is closed (Weekend).")
            return False

        # 2. Check Time Window (9:15 AM to 3:30 PM)
        market_start = time(9, 15)
        market_end = time(15, 30)
        current_time = now.time()

        if current_time < market_start or current_time > market_end:
            logger.info(f"Market is closed (Current time: {current_time}). Trading window: 09:15 - 15:30.")
            return False

        # 3. Verify with Fyers API (handles holidays)
        # We only check API if we are within the time window to save API calls
        try:
            # Re-using the logic from market_status_checker.py but adapted for the broker
            # Note: self.broker.fyers must be initialized (authenticated)
            if not self.broker.fyers:
                if not self.broker.authenticate():
                    logger.error("Could not authenticate to check market status via API.")
                    return False

            response = self.broker.fyers.market_status()
            if response and response.get('s') == 'ok' and 'marketStatus' in response:
                market_open, detail = _parse_nse_market_status(response["marketStatus"])
                if market_open:
                    return True
                logger.info("Fyers API indicates relevant market segments are closed (Holiday?).")
                logger.debug("FYERS market status detail: %s", detail)
                return False
        except Exception as e:
            logger.error(f"Error checking Fyers market status API: {e}")
            # If API fails but we are in time window, we might want to continue or stop.
            # Usually safer to stop if API is down.
            return False

        return True

    def wait_for_market_open(self):
        """Blocks until the market opens or a timeout occurs."""
        while not self.is_market_open():
            now = datetime.now(IST)
            # If it's after 3:30 PM, we stop waiting for today
            if now.time() > time(15, 30):
                logger.info("Market session ended for today. Exiting.")
                return False
            
            logger.info("Waiting for market to open... (Checking again in 5 minutes)")
            sleep_timer.sleep(300) # Check every 5 minutes
        
        logger.info("Market is now OPEN!")
        return True
