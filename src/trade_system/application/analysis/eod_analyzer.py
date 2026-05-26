import logging
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient as FyersBroker
from trade_system.application.strategies.rvol_trend import RvolTrendStrategy
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.config import Settings

logger = logging.getLogger(__name__)

class EODAnalyzer:
    def __init__(self, broker: FyersBroker, notifier: TelegramNotifier, settings: Settings):
        self.broker = broker
        self.notifier = notifier
        self.settings = settings
        self.strategy = RvolTrendStrategy(
            rvol_threshold=2.0, # High conviction: 2x average volume
            st_period=7,
            st_multiplier=3
        )
        # Fyers API Limit: ~10 requests per second. 
        # Using a semaphore and small delay to stay safe.
        self.rate_limiter = Semaphore(10)

    def _scan_single_symbol(self, name: str, today_str: str) -> list:
        """Helper to scan one symbol, used in thread pool."""
        with self.rate_limiter:
            # Construct Symbol
            if name in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                if name == "NIFTY": symbol = "NSE:NIFTY50-INDEX"
                elif name == "BANKNIFTY": symbol = "NSE:NIFTYBANK-INDEX"
                elif name == "FINNIFTY": symbol = "NSE:FINNIFTY-INDEX"
                elif name == "MIDCPNIFTY": symbol = "NSE:MIDCPNIFTY-INDEX"
            else:
                symbol = f"NSE:{name}-EQ"

            try:
                # Fetch last 30 days to have enough data for 20-day Avg Vol and Supertrend
                from_date = (datetime.now() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
                df = self.broker.get_historical_data(symbol, "D", from_date, today_str)
                
                if df is None or df.empty:
                    return []

                # Run Strategy
                df_results = self.strategy.generate_signals(df)
                if df_results.empty:
                    return []

                # Check if the LATEST row has a signal
                latest_row = df_results.iloc[-1]
                
                if latest_row['final_signal'] != 0:
                    logger.info(f"SIGNAL DETECTED: {symbol} at {latest_row['close']}")
                    return [{
                        'symbol': symbol,
                        'signal': latest_row['final_signal'],
                        'price': latest_row['close'],
                        'rvol': latest_row['rvol']
                    }]

            except Exception as e:
                logger.error(f"Error scanning {name}: {e}")
            
            # Small throttle to avoid tight loop burst
            time.sleep(0.1)
            return []

    def run_full_scan(self, underlying_names: list):
        """Scans all symbols in parallel and identifies signals generated today."""
        logger.info(f"Starting Parallel EOD Full Scan for {len(underlying_names)} symbols...")
        start_time = time.time()
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        signals_found = []

        # Use ThreadPoolExecutor for parallel network I/O
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(self._scan_single_symbol, name, today_str): name for name in underlying_names}
            
            for future in as_completed(future_to_symbol):
                res = future.result()
                if res:
                    signals_found.extend(res)

        end_time = time.time()
        logger.info(f"Parallel scan completed in {end_time - start_time:.1f} seconds.")
        
        self._send_summary_report(signals_found)

    def _send_summary_report(self, signals: list):
        """Sends a single consolidated Telegram message."""
        if not signals:
            self.notifier.send_message("<b>📊 EOD Market Scan</b>\n\nNo new RVOL-Trend signals detected today.")
            return

        buy_signals = [s for s in signals if s['signal'] == 1]
        sell_signals = [s for s in signals if s['signal'] == -1]

        report = [f"<b>📊 EOD Market Scan - {datetime.now().strftime('%d %b %Y')}</b>\n"]
        
        if buy_signals:
            report.append("<b>🚀 BULLISH BREAKOUTS (High Vol):</b>")
            for s in buy_signals:
                report.append(f"• {s['symbol']}: ₹{s['price']:.2f} (Vol: {s['rvol']:.1f}x)")
        
        if sell_signals:
            report.append("\n<b>🛑 BEARISH REVERSALS:</b>")
            for s in sell_signals:
                report.append(f"• {s['symbol']}: ₹{s['price']:.2f}")

        report.append("\n<i>Strategy: Supertrend + Relative Volume > 2.0x</i>")
        
        full_message = "\n".join(report)
        self.notifier.send_message(full_message)
        logger.info("EOD Summary report sent to Telegram.")
