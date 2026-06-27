"""
Unit test for GEX + EMA Stack backtester.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from trade_system.infrastructure.database.connection import get_engine
from tests.backtests.backtest_gamma_ema_stack import run_gex_backtest, compute_daily_gex_series

def test_gex_backtest_execution():
    engine = get_engine()
    
    # Create mock daily/15m data for a short backtest
    target_date = date(2026, 6, 27)
    timestamps = pd.date_range(end=target_date, periods=60, freq="15min")
    
    # Create trending stack
    closes = [100.0 + i * 0.1 for i in range(60)]
    opens = [c - 0.05 for c in closes]
    highs = [c + 0.2 for c in closes]
    lows = [c - 0.2 for c in closes]
    volumes = [200000] * 60
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    
    trades = run_gex_backtest(df, "NSE:NIFTY50-INDEX", engine)
    # The backtest should run to completion and return a list of trades (or empty if no signal triggers, but no crash)
    assert isinstance(trades, list)


def test_gex_series_computation():
    engine = get_engine()
    gex_df = compute_daily_gex_series("NSE:NIFTY50-INDEX", engine)
    assert isinstance(gex_df, pd.DataFrame)
    if not gex_df.empty:
        assert "gex_label" in gex_df.columns
        assert "total_gex" in gex_df.columns
