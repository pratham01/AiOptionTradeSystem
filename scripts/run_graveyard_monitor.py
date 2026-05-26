"""
GraveyardMonitorAgent — Watches the 'Sideways Graveyard' for explosive breakouts.
Notifies if a dormant stock suddenly gains > 1.5% momentum from its rejection point.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
from sqlalchemy import text
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.infrastructure.notifications.telegram import TelegramNotifier
from trade_system.config import Settings

LOGGER = logging.getLogger("graveyard_monitor")

class GraveyardMonitorAgent:
    def __init__(self, broker: FyersBroker):
        self.broker = broker
        self.settings = Settings.load()
        self.engine = get_engine()
        self.notifier = TelegramNotifier(self.settings.telegram.bot_token, self.settings.telegram.chat_id)
        self.alerted_today: set[str] = set()

    async def run(self):
        LOGGER.info("🚀 Graveyard Monitor Agent Active. Watching for explosive 'Waking' events...")
        
        while True:
            try:
                # 1. Fetch recently rejected stocks (Last 4 hours)
                rejected = self._get_graveyard_stocks()
                if not rejected:
                    await asyncio.sleep(300) # Sleep 5 mins if nothing to watch
                    continue

                # 2. Fetch live quotes
                symbols = list(rejected.keys())
                quotes = self.broker.get_quotes(symbols)

                # 3. Check for momentum surge
                for sym, q in quotes.items():
                    if sym in self.alerted_today: continue
                    
                    rej_info = rejected[sym]
                    curr_change = q.change_percent
                    prev_change = rej_info['change_pct']
                    
                    # Momentum surge logic: 1.5% move from rejection OR > 3.0% total intraday
                    momentum_gain = curr_change - prev_change
                    
                    if abs(momentum_gain) >= 1.5 or abs(curr_change) >= 3.0:
                        self._alert_breakout(sym, rej_info, curr_change, momentum_gain)
                        self.alerted_today.add(sym)

                await asyncio.sleep(120) # Check every 2 minutes

            except Exception as e:
                LOGGER.error(f"Graveyard monitor error: {e}")
                await asyncio.sleep(300)

    def _get_graveyard_stocks(self) -> Dict[str, Any]:
        """Queries DB for stocks in SIDEWAYS or DORMANT state."""
        try:
            query = text("""
                SELECT symbol, action, message, timestamp 
                FROM agent_thought_stream 
                WHERE action IN ('SIDEWAYS', 'DORMANT')
                AND timestamp > datetime('now', '-4 hours')
                ORDER BY timestamp DESC
            """)
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            if df.empty: return {}
            
            # Keep latest rejection per symbol
            df = df.drop_duplicates(subset=['symbol'], keep='first')
            
            results = {}
            for _, row in df.iterrows():
                # Parse change % from message "Change: +1.20%"
                match = re.search(r"Change: ([\+\-\d\.]+)", row['message'])
                prev_change = float(match.group(1)) if match else 0.0
                
                results[row['symbol']] = {
                    "action": row['action'],
                    "change_pct": prev_change,
                    "timestamp": row['timestamp']
                }
            return results
        except: return {}

    def _alert_breakout(self, symbol: str, rej_info: dict, curr_change: float, gain: float):
        short_sym = symbol.split(':')[-1].replace('-EQ', '')
        icon = "💥" if gain > 0 else "🩸"
        label = "BULLISH BREAKOUT" if gain > 0 else "BEARISH BREAKDOWN"
        
        msg = (
            f"{icon} <b>GRAVEYARD BREAKOUT: {short_sym}</b>\n"
            f"Status: <b>{label}</b>\n\n"
            f"• Was {rej_info['action']} at {rej_info['change_pct']:+.2f}%\n"
            f"• Now surging at <b>{curr_change:+.2f}%</b>\n"
            f"• Momentum Gain: <b>{gain:+.2f}%</b>\n\n"
            f"<i>💡 Note: This dormant stock has just 'Woken Up' with significant range expansion.</i>"
        )
        self.notifier.send(msg)
        LOGGER.info(f"Breakout alert sent for {symbol}: {gain:+.2f}% gain.")

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        settings = Settings.load()
        auth = FyersAuthService(settings)
        token = auth.get_valid_token()
        broker = FyersBroker(client_id=settings.fyers.client_id, access_token=token, user_id=settings.fyers.user_id, authenticator=auth.authenticator)
        
        monitor = GraveyardMonitorAgent(broker)
        await monitor.run()
        
    asyncio.run(main())
