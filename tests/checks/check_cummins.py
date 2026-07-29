import sys
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session

# Setup python path to import local modules
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.repository import get_market_data
from trade_system.shared.config import Settings

def check_cummins():
    symbol = "NSE:CUMMINSIND-EQ"
    print(f"Checking data for {symbol}...")
    
    settings = Settings.load()
    engine = get_engine()
    
    with Session(engine) as session:
        data = get_market_data(session, symbol, "D", limit=100)
        print(f"Number of historical daily candles retrieved: {len(data)}")
        if not data:
            print("No daily data found for Cummins!")
            return
            
        df = pd.DataFrame([
            {"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume} 
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        
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
        
        df["ATR_SMA_50"] = df["ATRr_14"].rolling(window=50).mean()
        
        # BBW threshold (e.g. 5th percentile over the last 100 days)
        df["bbw_threshold_5"] = df["BBB_20_2.0"].rolling(100, min_periods=60).apply(lambda x: x.quantile(0.05))
        df["bbw_threshold_10"] = df["BBB_20_2.0"].rolling(100, min_periods=60).apply(lambda x: x.quantile(0.10))
        
        # Print latest candles
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print("\nRecent daily candles and calculations:")
        cols_to_show = ["open", "high", "low", "close", "volume", "BBB_20_2.0", "bbw_threshold_5", "ATRr_14", "ATR_SMA_50"]
        print(df[cols_to_show].tail(15))
        
        latest = df.iloc[-1]
        bbw_current = latest["BBB_20_2.0"]
        atr_current = latest["ATRr_14"]
        atr_sma50 = latest["ATR_SMA_50"]
        bbw_thresh = df["BBB_20_2.0"].quantile(0.05)
        
        print(f"\nLatest Data Point ({latest.name}):")
        print(f"Close Price: {latest['close']:.2f}")
        print(f"BBW Current: {bbw_current:.4f}% (Threshold at 5% percentile: {bbw_thresh:.4f}%)")
        print(f"ATR 14-day: {atr_current:.2f} (50-day average of ATR: {atr_sma50:.2f})")
        
        # Resistance calculation
        recent_20 = df.tail(20)
        resistance = float(recent_20["high"].max())
        support = float(recent_20["low"].min())
        dist_to_res = (resistance - latest["close"]) / latest["close"]
        
        print(f"20-day Resistance: {resistance:.2f} (Support: {support:.2f})")
        print(f"Distance to resistance: {dist_to_res * 100:.2f}%")
        
        # Check conditions
        cond_bbw = bbw_current <= bbw_thresh
        cond_atr = atr_current < atr_sma50
        cond_res = dist_to_res <= 0.05
        
        print("\nSqueeze Screener Rules:")
        print(f"1. BBW <= 5th percentile ({bbw_current:.3f} <= {bbw_thresh:.3f}): {cond_bbw}")
        print(f"2. ATR < ATR SMA 50 ({atr_current:.3f} < {atr_sma50:.3f}): {cond_atr}")
        print(f"3. Close within 5% of resistance ({dist_to_res * 100:.1f}% <= 5%): {cond_res}")
        print(f"Verdict: {'SQUEEZED (PASS)' if (cond_bbw and cond_atr and cond_res) else 'NOT SQUEEZED (FAIL)'}")

if __name__ == "__main__":
    check_cummins()
