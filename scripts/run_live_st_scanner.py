"""
Continuous Supertrend Touch Scanner — Runs every 15 minutes during market hours.
Autonomously monitors the entire F&O universe for institutional re-entry signals.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, time as dt_time, date
from pathlib import Path

import pandas as pd

from trade_system.domains.advisory.application.agent.supertrend_touch_agent import SupertrendTouchAgent
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.shared.notifications.telegram import TelegramNotifier
from trade_system.shared.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/st_scanner.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger("st_scanner")

class ContinuousStScanner:
    def __init__(self):
        self.settings = Settings.load()
        self.auth_service = FyersAuthService(self.settings)
        bot_token = self.settings.st_confirmed_telegram.bot_token or self.settings.telegram.bot_token
        chat_id = self.settings.st_confirmed_telegram.chat_id or self.settings.telegram.chat_id
        self.notifier = TelegramNotifier(bot_token, chat_id)
        self.last_alert_sent: dict[str, datetime] = {}
        self.scan_interval_seconds = 3 * 60 # 3 minutes (aligned with db sync)
        self.is_running = True

    def _is_market_hours(self) -> bool:
        """Check if current time is within Indian market hours (09:15 - 15:30 IST)."""
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if now.weekday() >= 5: # Weekend
            return False
        
        current_time = now.time()
        start_time = dt_time(9, 15)
        end_time = dt_time(15, 30)
        return start_time <= current_time <= end_time

    def run(self):
        LOGGER.info("🚀 Starting Continuous Supertrend Touch Scanner...")
        
        while self.is_running:
            try:
                if not self._is_market_hours():
                    # If before market, wait until 9:15
                    # If after market, wait until next day
                    # For simplicity, we just sleep and check periodically
                    LOGGER.info("Market is closed. Standing by...")
                    time.sleep(300) # Check every 5 mins
                    continue

                LOGGER.info("Starting new 15-minute scan cycle...")
                
                # 1. Ensure fresh token
                token = self.auth_service.get_valid_token()
                broker = FyersBroker(
                    client_id=self.settings.fyers.client_id,
                    access_token=token,
                    user_id=self.settings.fyers.user_id,
                    authenticator=self.auth_service.authenticator
                )
                
                if not broker.authenticate():
                    LOGGER.error("Failed to authenticate broker. Retrying in 60s...")
                    time.sleep(60)
                    continue

                # 2. Run Scan
                agent = SupertrendTouchAgent(broker=broker)
                universe = self.settings.index_symbols
                
                # Scan 15m resolution using database cache (Only applicable to indices)
                touches = asyncio.run(agent.scan_for_touches(universe, resolution="15", use_cache=True))
                
                # 3. Filter and Alert
                if touches:
                    filtered_touches = []
                    for s in touches:
                        # Debounce: Only alert once per 15 mins for the same symbol
                        last_time = self.last_alert_sent.get(s.symbol)
                        if last_time and (datetime.now() - last_time).total_seconds() < 900:
                            continue
                            
                        filtered_touches.append(s)
                        self.last_alert_sent[s.symbol] = datetime.now()

                    if filtered_touches:
                        self._send_report(filtered_touches)

                LOGGER.info(f"Scan cycle complete. Next scan in {self.scan_interval_seconds // 60} minutes.")
                time.sleep(self.scan_interval_seconds)

            except Exception as e:
                LOGGER.exception(f"Continuous scanner encountered an error: {e}")
                time.sleep(60) # Wait before retry

    def _send_report(self, suggestions):
        today_str = date.today().strftime("%d %b %Y")
        now_str = datetime.now().strftime("%H:%M")
        
        report = [
            f"🧭 <b>Live ST-Touch Alert ({now_str})</b>",
            f"<i>Institutional re-entry zones detected on 15m chart.</i>",
            ""
        ]
        
        for s in suggestions:
            dir_icon = "🟢" if s.direction.value == "CALL" else "🔴"
            report.append(
                f" • {dir_icon} <b>{s.symbol.split(':')[-1]}</b> ({s.direction.value})\n"
                f"   Conf: {s.confidence:.0%} | {s.sector}"
            )
        
        report.append("\n<i>Strategy: Price touching Supertrend line with volume confirmation.</i>")
        self.notifier.send("\n".join(report))
        LOGGER.info(f"Sent live alert for {len(suggestions)} stocks.")

if __name__ == "__main__":
    scanner = ContinuousStScanner()
    scanner.run()
