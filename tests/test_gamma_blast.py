"""
Unit tests for GammaBlastDetector strategy logic.
"""

from __future__ import annotations

from datetime import date, datetime, time as dt_time
import pandas as pd
import numpy as np

from trade_system.domains.analysis.application.analysis.gamma_blast_strategy import GammaBlastDetector

def test_is_expiry_day():
    # Tuesday (1) is Nifty expiry day in the database
    tuesday = date(2026, 6, 30)  # Tuesday
    thursday = date(2026, 6, 25) # Thursday
    friday = date(2026, 6, 26)   # Friday
    
    assert GammaBlastDetector.is_expiry_day("NSE:NIFTY50-INDEX", tuesday) is True
    assert GammaBlastDetector.is_expiry_day("NSE:NIFTY50-INDEX", thursday) is False
    
    # Thursday (3) is Sensex expiry day in the database
    assert GammaBlastDetector.is_expiry_day("BSE:SENSEX-INDEX", thursday) is True
    assert GammaBlastDetector.is_expiry_day("BSE:SENSEX-INDEX", friday) is False

def test_check_volatility_squeeze():
    detector = GammaBlastDetector(squeeze_percentile_threshold=50.0)
    
    # Generate 50 candles with flat/compressed volatility at the end
    np.random.seed(42)
    closes = np.linspace(100, 101, 30)  # Moderate movement
    closes = np.append(closes, np.full(20, 101.0))  # Flat compression (squeeze)
    
    df = pd.DataFrame({
        "close": closes,
        "high": closes + 0.1,
        "low": closes - 0.1
    })
    
    is_sq, width = detector.check_volatility_squeeze(df)
    assert is_sq is True
    assert width >= 0.0

def test_evaluate_breakout():
    detector = GammaBlastDetector()
    
    obs_high = 24050.0
    obs_low = 23950.0
    
    # Outside trading window
    dir_none, price_none = detector.evaluate_breakout(
        current_time=datetime(2026, 6, 30, 14, 15),
        current_spot=24075.0,
        obs_high=obs_high,
        obs_low=obs_low,
        is_squeezed=True
    )
    assert dir_none is None
    
    # Inside window, not squeezed (no breakout triggered to avoid whipsaws)
    dir_none, price_none = detector.evaluate_breakout(
        current_time=datetime(2026, 6, 30, 14, 45),
        current_spot=24075.0,
        obs_high=obs_high,
        obs_low=obs_low,
        is_squeezed=False
    )
    assert dir_none is None
    
    # Inside window, squeezed, CALL breakout
    dir_call, price_call = detector.evaluate_breakout(
        current_time=datetime(2026, 6, 30, 14, 45),
        current_spot=24075.0,
        obs_high=obs_high,
        obs_low=obs_low,
        is_squeezed=True
    )
    assert dir_call == "CALL"
    assert price_call == obs_high
    
    # Inside window, squeezed, PUT breakout (breakdown)
    dir_put, price_put = detector.evaluate_breakout(
        current_time=datetime(2026, 6, 30, 14, 45),
        current_spot=23925.0,
        obs_high=obs_high,
        obs_low=obs_low,
        is_squeezed=True
    )
    assert dir_put == "PUT"
    assert price_put == obs_low

def test_select_0dte_option():
    detector = GammaBlastDetector(min_premium=5.0, max_premium=25.0)
    
    # Create fake option chain dataframe
    df_oc = pd.DataFrame([
        {"symbol": "NIFTY2663023900CE", "strike": 23900.0, "option_type": "CE", "ltp": 45.0},
        {"symbol": "NIFTY2663023950CE", "strike": 23950.0, "option_type": "CE", "ltp": 20.0}, # Matches CE criteria
        {"symbol": "NIFTY2663024000CE", "strike": 24000.0, "option_type": "CE", "ltp": 7.5},  # Matches CE criteria (closest to spot)
        {"symbol": "NIFTY2663024050CE", "strike": 24050.0, "option_type": "CE", "ltp": 2.0},  # Below min premium
        
        {"symbol": "NIFTY2663023900PE", "strike": 23900.0, "option_type": "PE", "ltp": 4.0},  # Below min
        {"symbol": "NIFTY2663023950PE", "strike": 23950.0, "option_type": "PE", "ltp": 12.0}, # Matches PE criteria
        {"symbol": "NIFTY2663024000PE", "strike": 24000.0, "option_type": "PE", "ltp": 38.0},
    ])
    
    # CALL side (spot is 23980, ATM is 24000)
    res_ce = detector.select_0dte_option("NSE:NIFTY50-INDEX", 23980.0, "CALL", df_oc)
    assert res_ce is not None
    assert res_ce[0] == "NIFTY2663024000CE"
    assert res_ce[1] == 7.5
    
    # PUT side
    res_pe = detector.select_0dte_option("NSE:NIFTY50-INDEX", 23980.0, "PUT", df_oc)
    assert res_pe is not None
    assert res_pe[0] == "NIFTY2663023950PE"
    assert res_pe[1] == 12.0
