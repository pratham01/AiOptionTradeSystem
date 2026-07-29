import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
from typing import Dict, List, Any
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

class SectorRotationAnalyzer:
    """
    Identifies sector leadership using Relative Strength (RS) vs. Nifty 50.
    """

    def __init__(self):
        self.engine = get_engine()

    def get_sector_leadership(self, lookback_days: int = 20) -> pd.DataFrame:
        """
        Calculates RS Slope for all sectors directly from OHLCV data.
        """
        LOGGER.info(f"SectorRotationAnalyzer: Calculating leadership over {lookback_days} days...")
        
        try:
            # 1. Fetch Nifty Daily Data
            nifty_df = self._fetch_daily_data("NSE:NIFTY50-INDEX", lookback_days + 30)
            if nifty_df.empty: return pd.DataFrame()
            
            # 2. Get Sector Mapping
            from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping
            mapping = get_sector_mapping()
            all_symbols = list(mapping.keys())
            
            # 3. Fetch Data for all F&O stocks
            placeholders = ', '.join([f':s{i}' for i in range(len(all_symbols))])
            query = text(f"""
                SELECT timestamp, symbol, close 
                FROM ohlcv_daily 
                WHERE symbol IN ({placeholders}) 
                AND timestamp >= :start_date
            """)
            start_date = (date.today() - timedelta(days=lookback_days + 40)).isoformat()
            
            params = {f's{i}': sym for i, sym in enumerate(all_symbols)}
            params['start_date'] = start_date
            
            with self.engine.connect() as conn:
                raw_df = pd.read_sql(query, conn, params=params)
            
            if raw_df.empty: return pd.DataFrame()

            # Normalize timestamps
            raw_df['date'] = pd.to_datetime(raw_df['timestamp'], format='mixed').dt.date.astype(str)
            raw_df['sector'] = raw_df['symbol'].map(mapping)
            
            # 4. Aggregate to Sector Closes (Mean of percentage changes)
            raw_df = raw_df.sort_values(['symbol', 'date'])
            raw_df['pct_change'] = raw_df.groupby('symbol')['close'].pct_change()
            
            sector_daily = raw_df.groupby(['date', 'sector'])['pct_change'].mean().reset_index()
            
            leadership = []
            for sector, group in sector_daily.groupby('sector'):
                group = group.sort_values('date')
                group['price'] = 100 * (1 + group['pct_change'].fillna(0)).cumprod()
                
                # Merge with Nifty
                merged = group.merge(nifty_df, left_on='date', right_on='timestamp', how='inner')
                if len(merged) < lookback_days: continue
                
                merged['rs'] = merged['price'] / merged['close']
                y = merged['rs'].tail(lookback_days).values
                x = np.arange(len(y))
                slope, _ = np.polyfit(x, y, 1)
                
                leadership.append({
                    "Sector": sector,
                    "RS_Slope": slope * 1000, # Scaled for readability
                    "Status": "🔥 LEADING" if slope > 0 else "❄️ LAGGING",
                    "Momentum": "UP" if slope > 0 else "DOWN"
                })
            
            return pd.DataFrame(leadership).sort_values("RS_Slope", ascending=False)

        except Exception as e:
            LOGGER.error(f"Sector rotation analysis failed: {e}")
            return pd.DataFrame()

    def _fetch_daily_data(self, symbol: str, days: int) -> pd.DataFrame:
        query = text(f"SELECT timestamp, close FROM ohlcv_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"symbol": symbol, "limit": days})
        # Convert timestamp to date string for join
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed').dt.date.astype(str)
        return df

if __name__ == "__main__":
    analyzer = SectorRotationAnalyzer()
    print(analyzer.get_sector_leadership())
