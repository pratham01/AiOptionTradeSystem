import time
import logging
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Any

from trade_system.core.ports.broker import Broker as BaseBroker
from trade_system.application.indicators.base import BaseIndicator
from trade_system.infrastructure.data.manager import DataManager
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.config import Settings
from .market import MarketStatus

logger = logging.getLogger(__name__)

class TradeEngine:
    def __init__(self, broker: BaseBroker, indicators: List[BaseIndicator], data_manager: DataManager, notifier: TelegramNotifier, settings: Settings):
        self.broker = broker
        self.indicators = indicators
        self.data_manager = data_manager
        self.notifier = notifier
        self.settings = settings
        self.is_running = False
        
        # Initialize MarketStatus checker
        self.market_status = MarketStatus(self.broker)
        
        # To keep track of the last direction of Supertrend
        self.last_directions = {} # symbol -> direction

    def run(self):
        """Starts the live trading engine."""
        logger.info("Starting Trade Engine (Automated Mode)...")
        
        # 1. Immediate termination if market is closed (holidays/weekends)
        if not self.market_status.is_market_open():
            logger.warning("Market is closed. System terminating immediately.")
            return

        # 2. Authenticate
        if not self.broker.authenticate():
            logger.error("Failed to authenticate broker. Exiting.")
            return

        self.is_running = True
        
        # 3. Initial data fetch and direction check
        for symbol in self.settings.live_symbols:
            self._update_direction(symbol)

        # 4. Main Loop (Runs 9:15 AM to 3:30 PM)
        while self.is_running:
            try:
                # Periodic Market Status Check (Ensures termination at 3:30 PM)
                if not self.market_status.is_market_open():
                    logger.info("Market is now closed. Shutting down system for today.")
                    self.stop()
                    break

                for symbol in self.settings.live_symbols:
                    self._process_symbol(symbol)
                
                # Check every minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in Trade Engine loop: {e}")
                time.sleep(10)

    def _process_symbol(self, symbol: str):
        """Processes a single symbol: fetch, resample, calculate indicators, notify."""
        logger.info(f"Processing symbol {symbol}...")
        
        # 1. Fetch 1-minute data (last few days to ensure enough data for indicators)
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        
        df_1m = self.broker.get_historical_data(symbol, "1", from_date, to_date)
        if df_1m is None or df_1m.empty:
            logger.warning(f"No data fetched for {symbol}.")
            return

        # 2. Resample to target timeframe
        timeframe = f"{self.settings.indicator_config.trend_timeframe_minutes}min"
        df_resampled = self.data_manager.resample_data(df_1m, timeframe)
        
        # 3. Save data
        self.data_manager.save_data(df_resampled, symbol, timeframe)
        
        # 4. Calculate Indicators (Supertrend)
        for indicator in self.indicators:
            df_resampled = indicator.calculate(df_resampled)
            
        # 5. Check for Signal (Direction Change)
        if 'supertrend_direction' in df_resampled.columns:
            latest_direction = df_resampled.iloc[-1]['supertrend_direction']
            
            # Check if direction changed since last check (or since previous candle)
            stored_direction = self.last_directions.get(symbol)
            
            if stored_direction is not None and latest_direction != stored_direction:
                self._notify_direction_change(symbol, latest_direction, df_resampled.iloc[-1])
            
            self.last_directions[symbol] = latest_direction

    def _update_direction(self, symbol: str):
        """Initial check of Supertrend direction."""
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        df_1m = self.broker.get_historical_data(symbol, "1", from_date, to_date)
        if df_1m is not None and not df_1m.empty:
            timeframe = f"{self.settings.indicator_config.trend_timeframe_minutes}min"
            df_resampled = self.data_manager.resample_data(df_1m, timeframe)
            for indicator in self.indicators:
                df_resampled = indicator.calculate(df_resampled)
            if 'supertrend_direction' in df_resampled.columns:
                self.last_directions[symbol] = df_resampled.iloc[-1]['supertrend_direction']
                logger.info(f"Initial direction for {symbol}: {self.last_directions[symbol]}")

    def _notify_direction_change(self, symbol: str, direction: int, candle: pd.Series):
        """Sends a Telegram notification about direction change."""
        dir_str = "🟢 BULLISH (Up)" if direction == 1 else "🔴 BEARISH (Down)"
        message = (
            f"<b>⚡ Supertrend Direction Change!</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>New Trend:</b> {dir_str}\n"
            f"<b>Time:</b> {candle['timestamp']}\n"
            f"<b>Price:</b> {candle['close']:.2f}\n"
        )
        logger.info(f"SIGNAL: {symbol} trend changed to {dir_str}")
        self.notifier.send_message(message)

    def stop(self):
        """Stops the trade engine."""
        self.is_running = False
        logger.info("Trade Engine stopped.")
