import pandas as pd
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def resample_data(self, df: pd.DataFrame, timeframe: str = '3min') -> pd.DataFrame:
        """Resamples OHLC data to the specified timeframe."""
        if df is None or df.empty:
            return df
        
        logger.info(f"Resampling data to {timeframe}...")
        
        # Ensure timestamp is the index for resampling
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # Define aggregation rules
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample
        resampled_df = df.resample(timeframe).agg(agg_dict).dropna()
        
        # Reset index to bring timestamp back as a column
        resampled_df = resampled_df.reset_index()
        
        return resampled_df

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Saves OHLC data to a CSV file."""
        if df is None or df.empty:
            return
        
        filename = f"{symbol.replace(':', '_')}_{timeframe}.csv"
        file_path = self.data_dir / filename
        
        logger.info(f"Saving data for {symbol} ({timeframe}) to {file_path}...")
        
        # Check if file exists to append or overwrite
        if file_path.exists():
            # In a real system, we'd append only new data
            # For simplicity, we'll overwrite or merge
            existing_df = pd.read_csv(file_path)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            combined_df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)

    def load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Loads OHLC data from a CSV file."""
        filename = f"{symbol.replace(':', '_')}_{timeframe}.csv"
        file_path = self.data_dir / filename
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return None
