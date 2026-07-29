import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import get_market_data
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.analysis.application.analysis.candlestick_patterns import CandlestickPatternAnalyzer
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fo_candlestick")

class FoCandlestickAnalyzer:
    def __init__(self):
        self.engine = get_engine()
        self.symbols = get_fo_universe()

    def analyze(self):
        logger.info(f"Analyzing {len(self.symbols)} F&O stocks for Doji and Hammer patterns...")
        
        results = []
        
        with Session(self.engine) as session:
            for symbol in self.symbols:
                try:
                    # Fetch last 5 daily candles to calculate trend_3
                    data = get_market_data(session, symbol, "D", limit=5)
                    if not data or len(data) < 4:
                        continue
                    
                    df = pd.DataFrame([
                        {
                            "timestamp": d.timestamp,
                            "open": d.open,
                            "high": d.high,
                            "low": d.low,
                            "close": d.close,
                            "volume": d.volume
                        } for d in data
                    ])
                    
                    # Calculate required columns for pattern detection
                    df["body"] = (df["close"] - df["open"]).abs()
                    df["range"] = df["high"] - df["low"]
                    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
                    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
                    df["direction"] = 0
                    df.loc[df["close"] > df["open"], "direction"] = 1
                    df.loc[df["close"] < df["open"], "direction"] = -1
                    df["trend_3"] = df["close"].diff(3)
                    
                    # Run pattern detection
                    labeled = CandlestickPatternAnalyzer._label_patterns(df)
                    latest = labeled.iloc[-1]
                    
                    patterns = []
                    if latest.get("doji"): patterns.append("Doji")
                    if latest.get("hammer"): patterns.append("Hammer")
                    if latest.get("inverted_hammer"): patterns.append("Inverted Hammer")
                    if latest.get("bullish_engulfing"): patterns.append("Bullish Engulfing")
                    if latest.get("bearish_engulfing"): patterns.append("Bearish Engulfing")
                    
                    if patterns:
                        results.append({
                            "Symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                            "Patterns": ", ".join(patterns),
                            "Close": latest["close"],
                            "Change %": round(((latest["close"] - latest["open"]) / latest["open"]) * 100, 2) if latest["open"] > 0 else 0
                        })
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")

        if results:
            df_results = pd.DataFrame(results)
            print("\n" + "="*60)
            print("🕯️ F&O CANDLESTICK PATTERN REPORT (DAILY)")
            print("="*60)
            print(df_results.to_string(index=False))
            print("="*60)
        else:
            print("\nNo significant candlestick patterns found today.")

if __name__ == "__main__":
    analyzer = FoCandlestickAnalyzer()
    analyzer.analyze()
