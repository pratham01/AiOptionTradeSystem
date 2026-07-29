import pandas as pd
import logging
from pathlib import Path
from trade_system.domains.strategy.application.indicators.sr_yata import SandRYata

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_sr_yata")

def test_yata(csv_path: str):
    logger.info(f"Loading historical data from {csv_path}...")
    if not Path(csv_path).exists():
        logger.error(f"Data file not found at {csv_path}.")
        return

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} rows. Calculating S&R Yata...")
    
    indicator = SandRYata(length=21, max_pivots=5)
    df = indicator.calculate(df)
    
    # Check results
    valid_res = df['res_line'].dropna()
    valid_sup = df['sup_line'].dropna()
    
    print("\n" + "="*50)
    print("🎯 S&R • YATA CALCULATION SUMMARY")
    print("="*50)
    print(f"Asset:          {csv_path}")
    print(f"Total Bars:     {len(df)}")
    print(f"Resistance Bars: {len(valid_res)}")
    print(f"Support Bars:    {len(valid_sup)}")
    
    if not valid_res.empty:
        print(f"Current Res:    ₹{valid_res.iloc[-1]:.2f}")
        print(f"Current Sup:    ₹{valid_sup.iloc[-1]:.2f}")
        
        # Show Fibonacci levels for the last bar
        last_row = df.iloc[-1]
        print("\nCurrent Fibonacci Levels (Retracement):")
        for i in range(1, 6):
            print(f"  Level {i}: ₹{last_row[f'fib_r{i}']:.2f}")
    else:
        print("Warning: No channels formed. Try increasing data range or reducing length.")
    print("="*50)

if __name__ == "__main__":
    test_yata("data/fo_historical/NSE_NIFTY50-INDEX_d_historical.csv")
