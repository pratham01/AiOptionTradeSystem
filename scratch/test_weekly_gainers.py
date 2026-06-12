import sys, os
from pathlib import Path
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

import logging
logging.basicConfig(level=logging.INFO)

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.application.analysis.weekly_gainers import NSEWeeklyGainersFetcher

s = Settings.load()
broker = FyersBrokerClient(client_id=s.fyers.client_id, access_token=s.fyers.access_token, user_id=s.fyers.user_id)

fetcher = NSEWeeklyGainersFetcher(broker)
# Test with a small universe
universe = ["NSE:TCS-EQ", "NSE:INFY-EQ", "NSE:RELIANCE-EQ", "NSE:GUJGASLTD-EQ"]
gainers = fetcher.fetch_weekly_gainers(universe)
for g in gainers:
    print(f"{g.name}: {g.change_pct}%")
    
# Send to telegram
fetcher.send_telegram_report(gainers, top_n=3)
