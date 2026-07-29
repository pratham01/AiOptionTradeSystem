"""
MWPL Analyzer — Identifies high-conviction F&O setups based on Market Wide Position Limits.
Detects potential short squeezes, exhaustion, and ban period filters.
"""
from __future__ import annotations

import logging
import pandas as pd
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

class MwplAnalyzer:
    """
    Analyzes NSE MWPL data to find institutional positioning extremes.
    """

    NSE_MWPL_URL = "https://archives.nseindia.com/content/fo/mwpl.csv"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.load()
        self.data_dir = Path(self.settings.data_dir) / "mwpl"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.engine = get_engine()

    def update_data(self) -> bool:
        """Downloads the latest MWPL file from NSE."""
        try:
            # NSE requires headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/"
            }
            
            response = requests.get(self.NSE_MWPL_URL, headers=headers, timeout=15)
            if response.status_code == 200:
                today_str = date.today().strftime("%Y%m%d")
                file_path = self.data_dir / f"mwpl_{today_str}.csv"
                file_path.write_text(response.text)
                LOGGER.info(f"MWPL data updated successfully: {file_path}")
                return True
            else:
                LOGGER.warning(f"Failed to fetch MWPL data. Status: {response.status_code}")
                return False
        except Exception as e:
            LOGGER.error(f"Error fetching MWPL data: {e}")
            return False

    def get_mwpl_data(self) -> pd.DataFrame:
        """Loads and parses the latest MWPL file."""
        try:
            # Find the most recent file
            files = sorted(self.data_dir.glob("mwpl_*.csv"), reverse=True)
            if not files:
                # If no local file, try to update
                if self.update_data():
                    files = sorted(self.data_dir.glob("mwpl_*.csv"), reverse=True)
                else:
                    return pd.DataFrame()
            
            # The CSV from NSE has some header junk
            # Typically looks like: DATE, SYMBOL, MWPL, OPEN INTEREST, PERCENTAGE
            df = pd.read_csv(files[0])
            
            # Standardize columns
            # NSE CSV columns are often "DATE", "SYMBOL", "MWPL", "OPEN_INTEREST", "PERCENT"
            df.columns = [c.strip().upper() for c in df.columns]
            
            if 'SYMBOL' not in df.columns or 'PERCENT' not in df.columns:
                # Some formats have different headers, try to guess
                # Sometimes it's a fixed format where index 1 is symbol, index 4 is %
                pass
            
            # Clean Symbol names (Remove -EQ if present)
            df['SYMBOL'] = df['SYMBOL'].str.strip()
            df['MWPL_PCT'] = pd.to_numeric(df['PERCENT'], errors='coerce')
            
            return df[['SYMBOL', 'MWPL_PCT']]
        except Exception as e:
            LOGGER.error(f"Error parsing MWPL data: {e}")
            return pd.DataFrame()

    def identify_setups(self, lookback_days: int = 5) -> Dict[str, List[str]]:
        """
        Classifies stocks into SQUEEZE, EXHAUSTION, or BAN categories.
        """
        df = self.get_mwpl_data()
        if df.empty: return {}

        results = {
            "BAN": [],      # > 95%
            "SQUEEZE": [],  # 85-94% + Downtrend
            "UNWINDING": [] # 90-95% + Uptrend
        }

        for _, row in df.iterrows():
            symbol = row['SYMBOL']
            pct = row['MWPL_PCT']
            
            if pct >= 95.0:
                results["BAN"].append(symbol)
                continue
            
            if pct >= 85.0:
                # Check price context (Trend over last 5 days)
                trend = self._get_recent_trend(symbol, lookback_days)
                
                if pct >= 85.0 and trend < -2.0: # Significant downtrend
                    results["SQUEEZE"].append(symbol)
                elif pct >= 90.0 and trend > 2.0: # Significant uptrend
                    results["UNWINDING"].append(symbol)
                    
        return results

    def _get_recent_trend(self, symbol: str, days: int) -> float:
        """Calculates price change % over last N days."""
        try:
            full_symbol = f"NSE:{symbol}-EQ"
            query = text("SELECT close FROM ohlcv_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
            with self.engine.connect() as conn:
                res = conn.execute(query, {"symbol": full_symbol, "limit": days + 1}).fetchall()
            
            if len(res) < 2: return 0.0
            
            closes = [r[0] for r in res]
            # (Latest - Oldest) / Oldest
            change = ((closes[0] - closes[-1]) / closes[-1]) * 100
            return change
        except:
            return 0.0

if __name__ == "__main__":
    analyzer = MwplAnalyzer()
    print("Identified MWPL Setups:")
    print(analyzer.identify_setups())
