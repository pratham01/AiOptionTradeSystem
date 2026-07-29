import sys
from pathlib import Path
import logging

# Add src to python path
sys.path.append(str(Path("src").resolve()))
logging.basicConfig(level=logging.INFO)

from trade_system.domains.analysis.application.analysis.breakout_screener import BreakoutScreener

def run_test():
    screener = BreakoutScreener()
    breakouts = screener.scan_for_breakouts()
    
    print("\n--- TEST RESULTS ---")
    if not breakouts:
        print("No breakouts found today. (Normal if market is quiet or closed early)")
    else:
        for b in breakouts:
            print(f"🚀 {b['symbol']} ({b['sector']}) | Breakout: {b['close']} > {b['orb_high']} | Vol Surge: {b['volume']} vs SMA {b['vol_sma']:.0f}")

if __name__ == "__main__":
    run_test()
