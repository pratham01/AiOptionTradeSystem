import logging
import time
from datetime import date, datetime, timedelta
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from trade_system.config import Settings
from trade_system.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
from trade_system.infrastructure.data.history import HistoricalDataService
from trade_system.infrastructure.data.storage import CsvDataCatalog
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.models import OhlcvDaily, Ohlcv15m
from trade_system.infrastructure.database.repository import save_market_data_batch
from trade_system.infrastructure.data.fo_universe import get_fo_universe

LOGGER = logging.getLogger(__name__)

class DataSanityAgent:
    """
    DataSanityAgent - Audits the SQLite market data tables, identifies missing/incomplete history,
    and automatically fetches & heals data gaps from the Fyers API.
    """
    def __init__(self, settings: Settings | None = None, broker: FyersBrokerClient | None = None):
        self.settings = settings or Settings.load()
        self.engine = get_engine()
        self.broker = broker
        
    def _get_broker(self) -> FyersBrokerClient:
        if self.broker is not None:
            return self.broker
        
        auth_service = FyersAuthService(self.settings)
        token = auth_service.get_valid_token()
        if not token:
            raise RuntimeError("Failed to retrieve a valid Fyers token for DataSanityAgent.")
            
        self.broker = FyersBrokerClient(
            client_id=self.settings.fyers.client_id,
            access_token=token,
            user_id=self.settings.fyers.user_id,
            authenticator=auth_service.authenticator
        )
        return self.broker

    def ensure_data_sanity(self, check_15m: bool = False, min_candles: int = 100, lookback_days: int = 365) -> dict[str, list[str]]:
        """
        Audits OhlcvDaily table for active F&O stocks. Any stock missing historical data
        is automatically backfilled to restore data completeness.
        """
        LOGGER.info("🔍 Running database data sanity check...")
        universe = get_fo_universe()
        
        missing_daily = []
        short_daily = []
        
        with Session(self.engine) as session:
            for symbol in universe:
                count = session.query(func.count(OhlcvDaily.id)).filter(OhlcvDaily.symbol == symbol).scalar()
                if count == 0:
                    missing_daily.append(symbol)
                elif count < min_candles:
                    short_daily.append(symbol)
                    
        to_heal = missing_daily + short_daily
        
        if not to_heal:
            LOGGER.info("✅ Daily market data sanity: OK. All F&O symbols have sufficient history.")
            return {"healed_daily": []}
            
        LOGGER.warning(f"⚠️ Data sanity failed: {len(missing_daily)} symbols missing daily history, {len(short_daily)} symbols with short history (< {min_candles} candles).")
        LOGGER.info(f"Symbols needing daily healing: {to_heal}")
        
        # Heal them
        healed = []
        try:
            broker = self._get_broker()
            catalog = CsvDataCatalog(self.settings.data_dir / "fo_historical")
            service = HistoricalDataService(broker, catalog)
            
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days)
            
            for symbol in to_heal:
                try:
                    LOGGER.info(f"🩹 Healing {symbol} (fetching daily data since {start_date.isoformat()})...")
                    df = service.collect(
                        symbol=symbol,
                        resolution="D",
                        from_date=start_date,
                        to_date=end_date,
                        chunk_days=365,
                        sleep_seconds=0.25 # sleep 250ms to respect broker rate limits
                    )
                    
                    if not df.empty:
                        with Session(self.engine) as session:
                            saved_count = save_market_data_batch(session, symbol, "D", df.to_dict('records'))
                            LOGGER.info(f"✅ Healed {symbol}: Saved {saved_count} daily rows to DB.")
                            healed.append(symbol)
                    else:
                        LOGGER.warning(f"❌ Fetch returned empty data for {symbol}.")
                except Exception as e:
                    LOGGER.error(f"❌ Failed to heal {symbol}: {e}")
                    
        except Exception as e:
            LOGGER.critical(f"💥 Failed to initialize broker for healing: {e}")
            
        return {"healed_daily": healed}
