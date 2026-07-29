import sys, os
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBrokerV2
from trade_system.shared.config import Settings
import logging

logging.basicConfig(level=logging.ERROR)
s = Settings.load()
token = s.fyers.access_token
b = FyersBrokerV2(s.fyers.client_id, token)
b.authenticate()

symbols = get_fo_universe()
quotes = b.get_quotes(symbols)

g = []
for sym, q in quotes.items():
    if q.open > 0:
        intra_chp = (q.last_price - q.open) / q.open * 100
        g.append((sym, q.last_price, intra_chp, q.change_percent))

g.sort(key=lambda x: x[2], reverse=True)
print("TOP 5 BY INTRADAY CHANGE (LTP vs OPEN):")
for s in g[:5]: print(f"{s[0]:<20} LTP:{s[1]:<8} Intra%:{s[2]:.2f}% PrevClose%:{s[3]:.2f}%")
print("\nWORST 5 BY INTRADAY CHANGE (LTP vs OPEN):")
for s in g[-5:][::-1]: print(f"{s[0]:<20} LTP:{s[1]:<8} Intra%:{s[2]:.2f}% PrevClose%:{s[3]:.2f}%")

print("\n---")
g.sort(key=lambda x: x[3], reverse=True)
print("TOP 5 BY PREV CLOSE CHANGE (Fyers chp):")
for s in g[:5]: print(f"{s[0]:<20} LTP:{s[1]:<8} Intra%:{s[2]:.2f}% PrevClose%:{s[3]:.2f}%")
print("\nWORST 5 BY PREV CLOSE CHANGE (Fyers chp):")
for s in g[-5:][::-1]: print(f"{s[0]:<20} LTP:{s[1]:<8} Intra%:{s[2]:.2f}% PrevClose%:{s[3]:.2f}%")
