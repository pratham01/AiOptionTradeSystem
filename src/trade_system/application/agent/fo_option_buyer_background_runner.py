import os
import sys
import time
import logging
import argparse
from datetime import datetime, time as dt_time
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from trade_system.application.agent.fo_option_buyer_agent import build_recommendations, Recommendation
from trade_system.application.agent.fo_option_buyer_daily_agent import setup_broker, fetch_market_context
from trade_system.config import Settings
from trade_system.infrastructure.data.fo_universe import get_fo_universe
from trade_system.application.analysis.breakout_screener import BreakoutScreener
from trade_system.infrastructure.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)

class FOOptionBuyerBackgroundRunner:
    def __init__(self, interval_minutes: int = 15, min_score: float = 75.0):
        self.interval_minutes = interval_minutes
        self.min_score = min_score
        self.settings = Settings.load()
        self.broker = None
        self.breakout_screener = BreakoutScreener()
        self.last_alerts: dict[str, datetime] = {}
        
        # Initialize notifier
        telegram_cfg = self.settings.st_confirmed_telegram if self.settings.st_confirmed_telegram.enabled else self.settings.top_gainer_telegram
        if telegram_cfg.enabled:
            self.notifier = TelegramNotifier(telegram_cfg.bot_token, telegram_cfg.chat_id)
        else:
            self.notifier = None

    def is_market_open(self) -> bool:
        now = datetime.now().time()
        start = dt_time(9, 15)
        end = dt_time(15, 30)
        return start <= now <= end

    def run(self):
        LOGGER.info(f"Starting FO Option Buyer Background Runner (Interval: {self.interval_minutes}m, Min Score: {self.min_score})")
        
        while True:
            try:
                if not self.is_market_open():
                    LOGGER.info("Market is closed. Waiting...")
                    time.sleep(60)
                    continue

                if self.broker is None:
                    self.broker = setup_broker()

                LOGGER.info("Scanning for high probability setups...")
                context = fetch_market_context(self.broker)
                quotes = self.broker.get_quotes(get_fo_universe())
                
                recs = build_recommendations(
                    quotes=quotes, 
                    top_n=10, 
                    vix=context.get("vix"), 
                    pcr=context.get("pcr")
                )
                
                high_conviction = [r for r in recs if r.score >= self.min_score]
                
                
                if high_conviction:
                    self.send_alerts(high_conviction, context)
                else:
                    LOGGER.info("No high conviction setups found in this scan.")
                    
                LOGGER.info("Running Mathematical Sector Breakout Scan...")
                breakouts = self.breakout_screener.scan_for_breakouts(use_sector_filter=False)
                if breakouts:
                    self.send_breakout_alerts(breakouts)

                LOGGER.info(f"Scan complete. Sleeping for {self.interval_minutes} minutes.")
                time.sleep(self.interval_minutes * 60)

            except Exception as e:
                LOGGER.error(f"Error in background runner: {e}")
                self.broker = None # Re-authenticate on next loop
                time.sleep(60)

    def send_alerts(self, recs: list[Recommendation], context: dict):
        now = datetime.now()
        alerts_to_send = []
        
        for r in recs:
            # Avoid duplicate alerts for the same symbol within the same hour
            if r.symbol in self.last_alerts:
                if (now - self.last_alerts[r.symbol]).total_seconds() < 3600:
                    continue
            
            alerts_to_send.append(r)
            self.last_alerts[r.symbol] = now

        if not alerts_to_send:
            return

        if self.notifier:
            lines = [
                f"🔥 <b>High Probability F&O Setups</b>",
                f"Time: {now.strftime('%H:%M')}",
                f"PCR: {context.get('pcr')} | VIX: {context.get('vix')}",
                ""
            ]
            for r in alerts_to_send:
                sym = r.symbol.replace("NSE:", "").replace("-EQ", "")
                lines.append(f"🎯 <b>{sym}</b> ({r.sector})")
                lines.append(f"Score: <b>{r.score:.2f}</b> | Spot: ₹{r.spot:.2f}")
                lines.append(f"Rationale: {r.rationale}")
                lines.append("")
                
            self.notifier.send("\n".join(lines))
            LOGGER.info(f"Sent alerts for {len(alerts_to_send)} symbols.")

    def send_breakout_alerts(self, breakouts: list[dict]):
        now = datetime.now()
        alerts_to_send = []
        for b in breakouts:
            if b['symbol'] in self.last_alerts:
                if (now - self.last_alerts[b['symbol']]).total_seconds() < 3600:
                    continue
            alerts_to_send.append(b)
            self.last_alerts[b['symbol']] = now
            
        if not alerts_to_send or not self.notifier:
            return
            
        lines = [
            f"🚀 <b>[SECTOR BREAKOUT] Chartink-Style Signal</b>",
            f"Time: {now.strftime('%H:%M')}",
            ""
        ]
        
        for b in alerts_to_send:
            sym = b['symbol'].replace("NSE:", "").replace("-EQ", "")
            lines.append(f"📈 <b>{sym}</b> ({b['sector']})")
            lines.append(f"Breakout Spot: ₹{b['close']:.2f} (Above ORB: ₹{b['orb_high']:.2f})")
            lines.append(f"Volume Surge: {b['volume']} (vs 20SMA {b['vol_sma']:.0f})")
            lines.append("")
            
        self.notifier.send("\n".join(lines))
        LOGGER.info(f"Sent Breakout alerts for {len(alerts_to_send)} symbols.")

def main():
    parser = argparse.ArgumentParser(description="Background F&O Option Buyer Agent")
    parser.add_argument("--interval", type=int, default=15, help="Scan interval in minutes")
    parser.add_argument("--min-score", type=float, default=75.0, help="Minimum score for alert")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    runner = FOOptionBuyerBackgroundRunner(interval_minutes=args.interval, min_score=args.min_score)
    runner.run()

if __name__ == "__main__":
    main()
