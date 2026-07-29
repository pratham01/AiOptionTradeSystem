import pandas as pd

from trade_system.domains.strategy.application.strategies.ema_vwap import EmaVwapStrategy
from trade_system.domains.analysis.application.backtesting.engine import BacktestEngine

def run_backtest():
    # Load 15-minute historical data for Nifty from the CSV (or DB)
    data_path = "data/fo_historical/NSE_NIFTY50-INDEX_15min_historical.csv"
    try:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    except FileNotFoundError:
        print(f"Data not found at {data_path}")
        return
        
    # Filter for the year 2026 to keep the backtest quick but significant
    df = df[df["timestamp"].dt.year == 2026].copy()
    
    if df.empty:
        print("No data found for 2026.")
        return
        
    print(f"Loaded {len(df)} candles for NSE:NIFTY50-INDEX starting from {df['timestamp'].iloc[0]}")
    
    # Initialize strategy and engine
    strategy = EmaVwapStrategy(ema_period=9)
    engine = BacktestEngine(strategy=strategy, initial_cash=100000.0)
    
    # Run
    result = engine.run(df)
    
    print("\n--- BACKTEST SUMMARY ---")
    for k, v in result.summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
        
    print(f"\n--- TRADES DETAILED ({len(result.trades)} trades) ---")
    for t in result.trades[:10]: # Print first 10
        print(f"[{t.entry_time}] {t.side} at {t.entry_price:.2f} -> Closed at {t.exit_price:.2f} (PnL: {t.pnl:.2f})")
        print(f"   Reason: {t.reason}")
        
    if len(result.trades) > 10:
        print(f"... and {len(result.trades) - 10} more trades.")

if __name__ == "__main__":
    run_backtest()
