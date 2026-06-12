import sys, os
from datetime import datetime
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.application.analysis.top_gainers import NSETop100GainersFetcher, TopGainersResult
from trade_system.infrastructure.data.fo_universe import get_fo_universe
from trade_system.infrastructure.notifications.telegram import TelegramNotifier

s = Settings.load()
broker = FyersBrokerClient(client_id=s.fyers.client_id, access_token=s.fyers.access_token, user_id=s.fyers.user_id)
fetcher = NSETop100GainersFetcher(broker=broker)

fo = get_fo_universe()
quotes = fetcher.fetch_all_quotes(symbols=fo, batch_size=50)

if quotes:
    filtered = fetcher.apply_filters(quotes)
    result = fetcher.get_top_gainers(filtered, top_n=10)
    notifier = TelegramNotifier(s.top_gainer_telegram.bot_token or s.telegram.bot_token, s.top_gainer_telegram.chat_id or s.telegram.chat_id)
    
    print("Sending real data to Telegram...")
    fetcher.send_telegram_report(result, notifier, max_rows=5, all_quotes=quotes)
    print("Sent!")
else:
    print("NO QUOTES RETURNED!")
