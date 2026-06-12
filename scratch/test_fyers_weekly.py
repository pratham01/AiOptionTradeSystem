import sys, os
from datetime import datetime, timedelta
sys.path.insert(0, 'src')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient

s = Settings.load()
broker = FyersBrokerClient(client_id=s.fyers.client_id, access_token=s.fyers.access_token, user_id=s.fyers.user_id)

now = datetime.now()
from_date = (now - timedelta(days=10)).strftime('%Y-%m-%d')
to_date = now.strftime('%Y-%m-%d')

df = broker.get_historical_data("NSE:TCS-EQ", "1W", from_date, to_date)
print(df)
