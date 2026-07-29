import sys, os
from datetime import datetime
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
quotes = fetcher.fetch_all_quotes(symbols=fo, batch_size=50)

if quotes:
    fo_symbols = set(get_fo_universe())
    fo_quotes = [q for q in quotes if q.symbol in fo_symbols and q.close > 0 and q.prev_close > 0]
    fo_quotes_sorted = sorted(fo_quotes, key=lambda x: x.change_pct, reverse=True)
    
    print('\n📈 Top 5 F&O Stocks:')
    for rank, stock in enumerate(fo_quotes_sorted[:5], 1):
        print(f'• {stock.name}  ₹{stock.close:.2f}  ({stock.change_pct:+.2f}%)')
        
    print('\n📉 Worst 5 F&O Stocks:')
    for rank, stock in enumerate(reversed(fo_quotes_sorted[-5:]), 1):
        print(f'• {stock.name}  ₹{stock.close:.2f}  ({stock.change_pct:+.2f}%)')
