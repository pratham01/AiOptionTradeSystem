import logging
import json
from pathlib import Path
from typing import Any
import pandas as pd

from sqlalchemy.orm import Session
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.repository import get_market_data
from trade_system.infrastructure.data.fo_universe import get_fo_universe
from trade_system.config import Settings

LOGGER = logging.getLogger(__name__)

class ConsolidationScreenerAgent:
    """
    Scans the F&O universe daily to identify stocks in a tight consolidation phase
    (Volatility Contraction / Squeeze) before they break out.
    """
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.engine = get_engine()
        
    def scan(self, bbw_percentile: float = 0.05) -> list[dict[str, Any]]:
        # Run daily historical data sanity check and heal gaps before screening
        from trade_system.application.agent.data_sanity_agent import DataSanityAgent
        sanity_agent = DataSanityAgent(self.settings)
        try:
            sanity_agent.ensure_data_sanity(min_candles=100)
        except Exception as e:
            LOGGER.error(f"Data sanity check failed before scan: {e}")

        universe = get_fo_universe()
        LOGGER.info(f"Scanning {len(universe)} F&O stocks for volatility contraction (squeeze)...")
        
        squeezed_stocks = []
        
        with Session(self.engine) as session:
            for symbol in universe:
                try:
                    # Need at least 100 days for percentiles and moving averages
                    data = get_market_data(session, symbol, "D", limit=100)
                    if not data or len(data) < 60:
                        continue
                        
                    df = pd.DataFrame([
                        {"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume} 
                        for d in data
                    ])
                    df.set_index("timestamp", inplace=True)
                    
                    # Calculate indicators manually
                    # Bollinger Bands Width (BBW)
                    df["SMA_20"] = df["close"].rolling(window=20).mean()
                    df["STD_20"] = df["close"].rolling(window=20).std()
                    df["BBU_20"] = df["SMA_20"] + (df["STD_20"] * 2)
                    df["BBL_20"] = df["SMA_20"] - (df["STD_20"] * 2)
                    df["BBB_20_2.0"] = (df["BBU_20"] - df["BBL_20"]) / df["SMA_20"] * 100
                    
                    # Average True Range (ATR)
                    high_low = df["high"] - df["low"]
                    high_close = (df["high"] - df["close"].shift()).abs()
                    low_close = (df["low"] - df["close"].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    df["ATRr_14"] = tr.rolling(window=14).mean()
                    
                    # Ensure enough data for BBW and ATR
                    if "BBB_20_2.0" not in df.columns or "ATRr_14" not in df.columns:
                        continue
                        
                    df["ATR_SMA_50"] = df["ATRr_14"].rolling(window=50).mean()
                    
                    latest = df.iloc[-1]
                    bbw_current = latest["BBB_20_2.0"]
                    atr_current = latest["ATRr_14"]
                    atr_sma50 = latest.get("ATR_SMA_50")
                    
                    if pd.isna(bbw_current) or pd.isna(atr_current) or pd.isna(atr_sma50):
                        continue
                        
                    # Calculate BBW threshold (e.g. 5th percentile over the last 100 days)
                    bbw_threshold = df["BBB_20_2.0"].quantile(bbw_percentile)
                    
                    # Strategy logic: 
                    # 1. Bandwidth is extremely tight (bottom X%)
                    # 2. Current ATR is smaller than its 50-day average (volatility is dropping)
                    if bbw_current <= bbw_threshold and atr_current < atr_sma50:
                        # Find resistance/support over the last 20 days
                        recent_20 = df.tail(20)
                        resistance = float(recent_20["high"].max())
                        support = float(recent_20["low"].min())
                        
                        # Only add if the current price is reasonably close to resistance (e.g., within 5%)
                        # so that we catch it *right before* or *as* it breaks out
                        dist_to_res = (resistance - latest["close"]) / latest["close"]
                        
                        if dist_to_res <= 0.05:
                            squeezed_stocks.append({
                                "symbol": symbol,
                                "resistance": resistance,
                                "support": support,
                                "bbw": float(bbw_current),
                                "atr": float(atr_current),
                                "close": float(latest["close"]),
                                "date": str(latest.name)
                            })
                            LOGGER.info(f"🔥 Squeeze found: {symbol} (Res: {resistance:.2f}, Sup: {support:.2f})")
                except Exception as e:
                    LOGGER.debug(f"Error scanning {symbol}: {e}")
                    
        # Save to database
        from trade_system.infrastructure.database.repository import save_consolidation_watchlist
        from datetime import date
        today_str = date.today().strftime('%Y-%m-%d')
        
        with Session(self.engine) as session:
            try:
                save_consolidation_watchlist(session, today_str, squeezed_stocks)
                LOGGER.info(f"Found {len(squeezed_stocks)} consolidated stocks. Saved to database for date {today_str}.")
            except Exception as e:
                LOGGER.error(f"Failed to save consolidation watchlist to database: {e}")
                
        return squeezed_stocks
