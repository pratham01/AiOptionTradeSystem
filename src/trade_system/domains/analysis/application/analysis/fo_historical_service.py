import logging
import pandas as pd
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional

from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient
from trade_system.domains.market_data.infrastructure.data.storage import CsvDataCatalog
from trade_system.domains.market_data.infrastructure.data.history import HistoricalDataService
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

LOGGER = logging.getLogger(__name__)

class FOHistoricalService:
    """
    Manages incremental F&O historical data fetching and wide-master creation.
    """

    def __init__(self, broker: FyersBrokerClient, settings: any):
        self.broker = broker
        self.settings = settings
        self.catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
        self.history_service = HistoricalDataService(broker, self.catalog)

    async def sync_fo_data(self, default_start_date: str = "2026-03-26"):
        """
        Synchronizes historical data for all F&O symbols incrementally.
        Checks existing data and only fetches missing periods.
        """
        LOGGER.info("🚀 Starting Incremental F&O Data Sync...")
        symbols = get_fo_universe()
        # Add major indices as well
        symbols += ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
        
        today = date.today()
        
        for symbol in symbols:
            try:
                # 1. Determine last date from CSV
                path = self.catalog.historical_path(symbol, "D")
                last_date = None
                
                if path.exists():
                    try:
                        df_existing = pd.read_csv(path, parse_dates=['timestamp'])
                        if not df_existing.empty:
                            last_date = df_existing['timestamp'].max().date()
                    except Exception as e:
                        LOGGER.warning(f"Could not read existing data for {symbol}: {e}")

                # 2. Determine from_date
                if last_date:
                    # Fetch from next day
                    from_date = last_date + timedelta(days=1)
                else:
                    from_date = datetime.strptime(default_start_date, "%Y-%m-%d").date()

                # 3. Fetch if needed
                if from_date <= today:
                    LOGGER.info(f"Fetching {symbol} from {from_date} to {today}...")
                    # history_service.collect saves data internally to catalog
                    self.history_service.collect(
                        symbol=symbol,
                        resolution="D",
                        from_date=from_date,
                        to_date=today,
                        sleep_seconds=0.2
                    )
                else:
                    LOGGER.info(f"✓ {symbol} is already up to date (last date: {last_date}).")

            except Exception as e:
                LOGGER.error(f"Error syncing {symbol}: {e}")

        LOGGER.info("✅ Incremental F&O Data Sync Complete.")
        
        # 4. Build the Ultimate Wide Master
        self.build_ultimate_wide_master()

    def build_ultimate_wide_master(self, output_name: str = "fo_ultimate_ohlcv_master.csv"):
        """
        Consolidates all individual symbol CSVs into a single wide-format file.
        """
        output_file = self.settings.data_dir / output_name
        LOGGER.info(f"Building Ultimate Wide Master at {output_file}...")
        
        all_files = list(self.catalog.root.glob("*_d_historical.csv"))
        if not all_files:
            LOGGER.error("No historical CSV files found to consolidate.")
            return

        metrics = ['open', 'high', 'low', 'close', 'volume']
        symbol_frames = []
        
        for filename in all_files:
            try:
                # Extract symbol name: NSE_SBIN-EQ_d_historical.csv -> SBIN
                # We want a cleaner name for columns
                raw_name = filename.name.split('_')[1]
                symbol = raw_name.replace('-EQ', '').replace('-INDEX', '')
                
                df = pd.read_csv(filename, parse_dates=['timestamp'])
                if df.empty:
                    continue
                
                # Ensure naive timestamps
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                
                # Select OHLCV and timestamp
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                
                # Rename columns to SYMBOL_METRIC
                df.columns = ['timestamp'] + [f"{symbol}_{m.upper()}" for m in metrics]
                df = df.set_index('timestamp')
                symbol_frames.append(df)
                    
            except Exception as e:
                LOGGER.error(f"Error processing {filename}: {e}")

        if symbol_frames:
            # Join all symbol dataframes horizontally on timestamp
            ultimate_df = pd.concat(symbol_frames, axis=1).sort_index()
            
            # Save to CSV
            ultimate_df.to_csv(output_file)
            LOGGER.info(f"Ultimate Wide Master Created! Dates: {len(ultimate_df)}, Columns: {len(ultimate_df.columns)}")
        else:
            LOGGER.warning("No data found to process into wide master.")
