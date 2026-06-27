import sys, os
sys.path.insert(0, 'src')
import yfinance as yf
from trade_system.infrastructure.data.nse_universe import NSE_UNIVERSE

yf_symbols = [s.replace('NSE:', '').replace('-EQ', '') + '.NS' for s in NSE_UNIVERSE[:10]]
data = yf.download(yf_symbols, period="5d")
print(data['Close'])
