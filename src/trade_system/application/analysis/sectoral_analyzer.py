import requests
import pandas as pd
import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)

class SectoralAnalyzer:
    """
    Fetches sectoral indices performance from NSE India.
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/"
        }
        self.session = requests.Session()

    def fetch_sector_performance(self) -> Optional[pd.DataFrame]:
        """
        Fetches sectoral indices data and returns as a DataFrame.
        """
        try:
            # Step 1: Initialize session
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
            
            # Step 2: Fetch sectoral indices data
            url = "https://www.nseindia.com/api/allIndices"
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            indices = data.get("data", [])
            sector_data = []
            
            for idx in indices:
                if idx.get("key") == "SECTORAL_INDICES":
                    sector_data.append({
                        "Sector": idx.get("index", "Unknown"),
                        "LTP": idx.get("last", 0),
                        "Change_Pct": idx.get("percentChange", 0)
                    })
            
            if sector_data:
                return pd.DataFrame(sector_data)
            return None
            
        except Exception as e:
            LOGGER.error(f"Error fetching sectoral performance: {e}")
            return None

    def get_top_and_worst_sectors(self, top_n: int = 3) -> dict[str, pd.DataFrame]:
        """
        Returns top and worst performing sectors.
        """
        df = self.fetch_sector_performance()
        if df is None or df.empty:
            return {"top": pd.DataFrame(), "worst": pd.DataFrame()}
            
        sorted_df = df.sort_values("Change_Pct", ascending=False)
        return {
            "top": sorted_df.head(top_n),
            "worst": sorted_df.tail(top_n).iloc[::-1] # Reverse to show worst first
        }
