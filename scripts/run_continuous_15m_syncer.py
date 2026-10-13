"""Continuous 15-Minute Candle Syncer — Automatically fetches completed 15m candles
across the entire F&O universe during NSE market hours (09:15 - 15:35 IST).
"""
import logging
import os
import sys
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.sync_today_15m import sync_fo_15m_candles

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/15m_syncer.log"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger("15m_syncer")
IST = ZoneInfo("Asia/Kolkata")


def is_market_hours() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    t = now.time()
    return dt_time(9, 15) <= t <= dt_time(15, 35)


def run_continuous_syncer():
    LOGGER.info("🚀 Starting Continuous 15m Candle Syncer for F&O Universe...")
    while True:
        try:
            if not is_market_hours():
                LOGGER.info("Market is closed. Sleeping for 5 minutes...")
                time.sleep(300)
                continue

            LOGGER.info("Market open: Running scheduled 15m candle sync...")
            count = sync_fo_15m_candles()
            LOGGER.info(f"Sync completed successfully with {count} candles.")

            # Sleep 15 minutes between sync cycles
            LOGGER.info("Standing by for next 15-minute sync window (sleeping 900s)...")
            time.sleep(900)
        except Exception as e:
            LOGGER.error(f"Error in continuous syncer cycle: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    run_continuous_syncer()
