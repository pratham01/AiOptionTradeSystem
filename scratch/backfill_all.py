from datetime import date, timedelta
from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.domains.market_data.infrastructure.data.storage import CsvDataCatalog
from trade_system.domains.market_data.infrastructure.data.history import HistoricalDataService
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import upsert_market_data_fast
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
import logging
import time

logging.basicConfig(level=logging.INFO)
settings = Settings.load()
auth = FyersAuthService(settings)
token = auth.read_cached_token() or settings.fyers.access_token

broker = FyersBrokerClient(client_id=settings.fyers.client_id, access_token=token, user_id=settings.fyers.user_id, authenticator=auth.authenticator)
catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
service = HistoricalDataService(broker, catalog)
engine = get_engine()

symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"] + get_fo_universe()
to_date = date.today()
from_date = to_date - timedelta(days=60) # Fetch last 60 days to fill any gaps

for sym in symbols:
    try:
        # Daily
        df_d = service.collect(sym, resolution="D", from_date=from_date, to_date=to_date, chunk_days=60, sleep_seconds=0.1)
        if not df_d.empty:
            records = df_d.to_dict('records')
            upsert_market_data_fast(engine, sym, "D", records)
        
        # 15m
        df_15 = service.collect(sym, resolution="15", from_date=from_date, to_date=to_date, chunk_days=60, sleep_seconds=0.1)
        if not df_15.empty:
            records = df_15.to_dict('records')
            upsert_market_data_fast(engine, sym, "15", records)
            
        print(f"Backfilled {sym}")
    except Exception as e:
        print(f"Failed for {sym}: {e}")
