"""
VcpScannerAgent — Detects Volatility Contraction Patterns (VCP).
Identifies spring-loaded stocks with shrinking ATR and dry volume for future momentum.
"""
from __future__ import annotations

import logging
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Any

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import get_market_data
from sqlalchemy.orm import Session
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

LOGGER = logging.getLogger(__name__)

class VcpScannerAgent:
    """
    Scans the F&O universe for stocks in a multi-day consolidation phase.
    These 'Coiling' stocks are prime candidates for explosive breakouts.
    """

    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def scan_for_coiling(self, lookback_days: int = 10) -> List[Dict[str, Any]]:
        """Identifies stocks with steadily shrinking daily ranges."""
        universe = get_fo_universe()
        coiling_results = []
        
        with Session(self.engine) as session:
            for symbol in universe:
                try:
                    # Fetch last 15 days for a safe baseline
                    data = get_market_data(session, symbol, "D", limit=15)
                    if not data or len(data) < 10: continue
                    
                    df = pd.DataFrame([{"timestamp": d.timestamp, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume} for d in data])
                    
                    # 1. Calculate Daily Range % (High-Low / Close)
                    df['range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
                    
                    # 2. ATR Shrinkage (The Squeeze)
                    ranges = df['range_pct'].tail(lookback_days).tolist()
                    is_shrinking = all(ranges[i] >= ranges[i+1] * 0.9 for i in range(len(ranges)-2)) # Allow minor noise
                    
                    # More precise: Slope of the range
                    range_slope = np.polyfit(range(len(ranges)), ranges, 1)[0]
                    
                    # 3. Volume Dry-up
                    avg_vol = df['volume'].head(10).mean()
                    latest_vol = df['volume'].iloc[-1]
                    vol_dry_pct = (latest_vol / avg_vol) if avg_vol > 0 else 1.0
                    
                    # SELECTION CRITERIA: Shrinking range + Volume < 0.8x of average
                    if range_slope < 0 and vol_dry_pct < 0.85:
                        coiling_results.append({
                            "symbol": symbol,
                            "coiling_score": round(abs(range_slope) * 10, 2),
                            "vol_dry_pct": round(vol_dry_pct, 2),
                            "current_range": round(df['range_pct'].iloc[-1], 2),
                            "avg_range": round(df['range_pct'].mean(), 2)
                        })
                except: continue
                
        # Sort by coiling score (Tightest first)
        return sorted(coiling_results, key=lambda x: x["coiling_score"], reverse=True)

import numpy as np

if __name__ == "__main__":
    agent = VcpScannerAgent()
    print("Searching for Spring-Loaded Stocks (VCP)...")
    results = agent.scan_for_coiling()
    for r in results[:5]:
        print(f"🔥 {r['symbol']}: Coiling Score {r['coiling_score']} | Vol dry {r['vol_dry_pct']}x")
