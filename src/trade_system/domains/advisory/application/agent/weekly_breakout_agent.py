from __future__ import annotations

import sqlite3
import pandas as pd
from typing import List, Dict

from trade_system.domains.strategy.application.indicators.bollinger_bands import BollingerBandsDetector

class WeeklyBreakoutAgent:
    def __init__(self, db_path: str = "data/trade_system.db"):
        self.db_path = db_path
        # Use a 20-week period and a 20-week squeeze lookback (to represent 5 months of tight price action)
        self.bb_detector = BollingerBandsDetector(period=20, std_dev=2.0, squeeze_lookback=20, squeeze_percentile=15.0)

    def scan_universe(self, min_consolidation_weeks: int = 4, direction: str = "both") -> List[Dict]:
        """Scans all symbols in the DB for a weekly breakout."""
        suggestions_found = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all symbols that have daily data
            symbols_df = pd.read_sql("SELECT DISTINCT symbol FROM ohlcv_daily", conn)
            symbols = symbols_df['symbol'].tolist()
            
            for symbol in symbols:
                query = f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_daily
                WHERE symbol = ? 
                ORDER BY timestamp ASC 
                """
                df = pd.read_sql(query, conn, params=(symbol,))
                
                if df.empty or len(df) < 50:
                    continue
                    
                df['timestamp'] = pd.to_datetime(df['timestamp'], format="mixed")
                df.set_index('timestamp', inplace=True)
                
                # Resample to Weekly
                weekly = df.resample('W-FRI').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(weekly) < 40:
                    continue
                    
                weekly = weekly.reset_index()
                weekly = self.bb_detector.compute(weekly)
                
                # Check for breakout in the last 2 weeks (current incomplete week or last completed week)
                recent = weekly.tail(2 + min_consolidation_weeks).reset_index(drop=True)
                
                if len(recent) < 2 + min_consolidation_weeks:
                    continue
                    
                # We look at the last 2 bars for a breakout
                breakout_idx = None
                breakout_type = None
                
                # Check if [-1] or [-2] is a breakout
                for idx in [-1, -2]:
                    phase = recent.iloc[idx].get("bb_phase")
                    if direction in ["bullish", "both"] and phase == "EXPANSION_BULLISH":
                        breakout_idx = idx
                        breakout_type = "BULLISH"
                        break
                    if direction in ["bearish", "both"] and phase == "EXPANSION_BEARISH":
                        breakout_idx = idx
                        breakout_type = "BEARISH"
                        break
                        
                if breakout_idx is not None:
                    # Now check if the `min_consolidation_weeks` prior to the breakout were in a squeeze or flat
                    consolidation_valid = True
                    squeeze_count = 0
                    
                    # Convert negative index to positive relative to the recent chunk
                    abs_breakout_idx = len(recent) + breakout_idx 
                    
                    for i in range(abs_breakout_idx - min_consolidation_weeks, abs_breakout_idx):
                        if i < 0:
                            consolidation_valid = False
                            break
                        phase = recent.iloc[i].get("bb_phase")
                        if phase in ["EXPANSION_BULLISH", "EXPANSION_BEARISH"]:
                            consolidation_valid = False
                            break
                        if phase == "SQUEEZE":
                            squeeze_count += 1
                            
                    # Require at least one week of formal 'SQUEEZE' (or 50% of the period) during the consolidation period to ensure it was actually tight
                    # To be slightly lenient, we require squeeze_count >= 1 or at least half the min period
                    req_squeeze = max(1, min_consolidation_weeks // 2)
                    if consolidation_valid and squeeze_count >= req_squeeze:
                        bar = recent.iloc[breakout_idx]
                        suggestions_found.append({
                            "symbol": symbol,
                            "type": breakout_type,
                            "breakout_date": bar['timestamp'].strftime("%Y-%m-%d"),
                            "close": round(bar['close'], 2),
                            "volume_score": round(bar.get('volatility_explosion_score', 0), 2),
                            "squeeze_weeks": squeeze_count,
                            "bbw": round(bar['bbw'], 3)
                        })
                        
        return sorted(suggestions_found, key=lambda x: x['volume_score'], reverse=True)
