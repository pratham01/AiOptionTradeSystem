import sys, os
from pathlib import Path
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.domains.analysis.application.analysis.top_gainers import NSETop100GainersFetcher
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

s = Settings.load()
broker = FyersBrokerClient(client_id=s.fyers.client_id, access_token=s.fyers.access_token, user_id=s.fyers.user_id)
fetcher = NSETop100GainersFetcher(broker=broker)

fo = get_fo_universe()
quotes = fetcher.fetch_all_quotes(symbols=fo[:50], batch_size=50) # Fetch 50 just to test if it works

if quotes:
    print(f"Successfully fetched {len(quotes)} FO quotes!")
    for q in quotes[:3]:
        print(f"{q.symbol}: {q.change_pct}% (close: {q.close}, open: {q.open})")
else:
    print("NO QUOTES RETURNED!")
