"""
Test script for SmartOIAnalyzer.
"""

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
from datetime import datetime
from trade_system.application.analysis.smart_oi_analyzer import SmartOIAnalyzer
from trade_system.infrastructure.database.connection import get_engine

def run_tests():
    print("🧪 Starting SmartOIAnalyzer Verification Tests...")
    
    # 1. Initialize Analyzer
    analyzer = SmartOIAnalyzer("NSE:NIFTY50-INDEX")
    print("✅ Initialized SmartOIAnalyzer for Nifty 50.")

    # 2. Get available dates
    engine = get_engine()
    with engine.connect() as conn:
        df_dates = pd.read_sql("SELECT DISTINCT date(timestamp) as date FROM option_chain_data ORDER BY date DESC LIMIT 3", conn)
        if df_dates.empty:
            print("❌ No option chain data available in database to test. Please run the collector first.")
            return
        test_date = df_dates.iloc[0]["date"]
        print(f"✅ Selected date for testing: {test_date}")

        # Fetch latest two snapshots
        snapshots = analyzer.fetch_latest_snapshots(test_date, limit=2)
        print(f"✅ Fetched {len(snapshots)} snapshots from database.")
        
        if len(snapshots) < 2:
            print("⚠️ Only found one snapshot. Running single-snapshot analysis (changes will be mock/zero).")
            ts, current_df = snapshots[0]
            prev_df = None
        else:
            ts, current_df = snapshots[0]
            _, prev_df = snapshots[1]
            print(f"✅ Loaded current snapshot at {ts} (row count: {len(current_df)}) and previous snapshot.")

        # Estimate spot price
        spot_price = current_df["strike"].mean()
        # Find close price from ohlcv_1m around the same timestamp
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        ohlcv_query = """
            SELECT * FROM ohlcv_1m 
            WHERE symbol = 'NSE:NIFTY50-INDEX' 
              AND timestamp <= :ts_str
            ORDER BY timestamp DESC 
            LIMIT 50
        """
        ohlcv_df = pd.read_sql(ohlcv_query, conn, params={"ts_str": ts_str})
        
        if not ohlcv_df.empty:
            spot_price = ohlcv_df.iloc[0]["close"]
            print(f"✅ Found matching Nifty spot price: {spot_price} at {ohlcv_df.iloc[0]['timestamp']}")
        else:
            print(f"⚠️ No matching ohlcv_1m data found. Using default spot price approximation: {spot_price}")

        # 3. Test Smart OI Analysis & Filtering
        print("\n--- Test 3: Option Chain Filtering & Buildup Classification ---")
        analysis = analyzer.analyze_smart_oi(current_df, prev_df, spot_price=spot_price)
        
        print("Summary of analysis:")
        for k, v in analysis["summary"].items():
            print(f" • {k}: {v}")
            
        print(f" • Net Smart OI Signal: {analysis['signal'].upper()}")
        print(f" • Signal Strikes detected: {len(analysis['signal_strikes'])}")
        
        if len(analysis['signal_strikes']) > 0:
            print("First 3 Signal Strikes:")
            for s in analysis['signal_strikes'][:3]:
                print(f"   Strike {s['strike']} {s['option_type']}: OI={s['oi']:,} | Chg={s['oi_change']:,} ({s['oi_change_pct']:.2f}%) | LTP={s['ltp']} | Buildup={s['buildup']} ({s['action']})")
        else:
            print("⚠️ No signal strikes passed the filter criteria.")

        # 4. Test Confluence & Divergence detection
        print("\n--- Test 4: Confluence & Divergence Detection ---")
        if not ohlcv_df.empty:
            confluence = analyzer.detect_confluence_divergence(analysis["signal"], ohlcv_df)
            print(f" • Status: {confluence['status']}")
            print(f" • Price Trend: {confluence['price_trend']}")
            print(f" • Volume Expansion: {confluence['volume_expansion']}")
            print(f" • VWAP Level: ₹{confluence['vwap']:.2f}")
            print(f" • Swarm Narrative: {confluence['narrative']}")
        else:
            print("⚠️ Skipped confluence detection due to missing OHLCV data.")

    print("\n🎉 Verification tests completed successfully!")

if __name__ == "__main__":
    run_tests()
