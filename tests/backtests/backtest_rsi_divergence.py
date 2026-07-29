import pandas as pd
import numpy as np
import logging
from pathlib import Path
from trade_system.domains.strategy.application.indicators.rsi_divergence import RsiDivergence

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_rsi_div")

def run_backtest(df: pd.DataFrame, label: str):
    """Simple backtest runner."""
    strategy = RsiDivergence()
    df = strategy.calculate(df)
    
    trades = []
    pos = 0
    entry_price = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        sig = row['signal']
        price = row['close']
        
        if pos == 1 and sig == -1:
            trades.append((price - entry_price) / entry_price)
            pos = 0
        elif pos == -1 and sig == 1:
            trades.append((entry_price - price) / entry_price)
            pos = 0
            
        if pos == 0:
            if sig == 1:
                pos = 1
                entry_price = price
            elif sig == -1:
                pos = -1
                entry_price = price

    if not trades:
        return None
    
    return {
        "Timeframe": label,
        "Total Trades": len(trades),
        "Win Rate": (np.array(trades) > 0).mean(),
        "Total Return": np.sum(trades),
        "Avg Return": np.mean(trades)
    }

def main():
    # 1. Daily Backtest
    nifty_d_path = "data/fo_historical/NSE_NIFTY50-INDEX_d_historical.csv"
    results = []
    
    if Path(nifty_d_path).exists():
        df_d = pd.read_csv(nifty_d_path, parse_dates=['timestamp'])
        res_d = run_backtest(df_d, "Daily")
        if res_d: results.append(res_d)
    
    # 2. Intraday Backtest (Mock or fetch if available)
    # We'll try to fetch 5min data for the last week to compare
    import asyncio
    from trade_system.shared.config import Settings
    from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient as FyersBroker
    from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator
    from datetime import date, timedelta

    async def get_intraday_data():
        settings = Settings.load()
        authenticator = FyersAuthenticator(settings)
        broker = FyersBroker(settings.fyers.client_id, settings.fyers.access_token, settings.fyers.user_id, authenticator=authenticator)
        if not broker.authenticate(): return None
        
        logger.info("Fetching 5min data for comparison...")
        df_5m = broker.get_historical_data(
            "NSE:NIFTY50-INDEX", "5", 
            (date.today() - timedelta(days=30)).strftime("%Y-%m-%d"),
            date.today().strftime("%Y-%m-%d")
        )
        return df_5m

    df_5m = asyncio.run(get_intraday_data())
    if df_5m is not None and not df_5m.empty:
        res_5m = run_backtest(df_5m, "5-Minute")
        if res_5m: results.append(res_5m)

    if results:
        final_df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("📊 RSI DIVERGENCE BACKTEST: TIMEFRAME COMPARISON")
        print("="*60)
        print(final_df.to_string(index=False))
        print("="*60)
        print("\nObservation: Lower timeframes (5m) generate more signals but often have lower win rates")
        print("due to noise. Daily timeframes are smoother but signals are infrequent.")
    else:
        print("No data available for backtest.")

if __name__ == "__main__":
    main()
