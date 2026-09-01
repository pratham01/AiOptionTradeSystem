"""
Unit tests for DailyReversalScanner.
"""
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from trade_system.domains.analysis.application.analysis.reversal_scanner import DailyReversalScanner, DailyReversalSetup


def test_bullish_hammer_reversal_detection():
    scanner = DailyReversalScanner(min_probability=40)
    
    # Generate 30 days of downward trending data followed by a hammer candle
    dates = [datetime(2026, 8, 1) + timedelta(days=i) for i in range(30)]
    closes = np.linspace(2500, 2200, 30)
    
    rows = []
    for d, c in zip(dates[:-1], closes[:-1]):
        rows.append({
            "timestamp": d,
            "open": c + 5.0,
            "high": c + 10.0,
            "low": c - 10.0,
            "close": c,
            "volume": 100000
        })
        
    # Last candle: Hammer (long lower wick, small body, close near high with volume surge)
    rows.append({
        "timestamp": dates[-1],
        "open": 2190.0,
        "high": 2210.0,
        "low": 2150.0,  # 60 pt range, 40 pt lower wick
        "close": 2205.0, # Green body, closed near high
        "volume": 250000 # 2.5x volume surge
    })
    
    df = pd.DataFrame(rows)
    setup = scanner.evaluate_series("NSE:RELIANCE-EQ", df, target_date=date(2026, 8, 30))
    
    assert setup is not None
    assert setup.direction == "CALL"
    assert setup.probability >= 40
    assert "Hammer / Rejection Pin Bar" in setup.confluences
    assert setup.stop_loss < setup.ltp
    assert setup.target_1 > setup.ltp


def test_bearish_shooting_star_reversal_detection():
    scanner = DailyReversalScanner(min_probability=40)
    
    # Generate 30 days of upward trending data followed by a shooting star candle
    dates = [datetime(2026, 8, 1) + timedelta(days=i) for i in range(30)]
    closes = np.linspace(1500, 1800, 30)
    
    rows = []
    for d, c in zip(dates[:-1], closes[:-1]):
        rows.append({
            "timestamp": d,
            "open": c - 5.0,
            "high": c + 10.0,
            "low": c - 5.0,
            "close": c,
            "volume": 100000
        })
        
    # Last candle: Shooting Star (long upper wick, small body, close near low with volume surge)
    rows.append({
        "timestamp": dates[-1],
        "open": 1810.0,
        "high": 1870.0,  # 60 pt range, 60 pt upper wick
        "low": 1805.0,
        "close": 1808.0, # Red/small body, rejected from 1870
        "volume": 220000
    })
    
    df = pd.DataFrame(rows)
    setup = scanner.evaluate_series("NSE:TCS-EQ", df, target_date=date(2026, 8, 30))
    
    assert setup is not None
    assert setup.direction == "PUT"
    assert setup.probability >= 40
    assert "Shooting Star / Upper Rejection" in setup.confluences
    assert setup.stop_loss > setup.ltp
    assert setup.target_1 < setup.ltp


def test_reversal_scanner_handles_short_data():
    scanner = DailyReversalScanner()
    df = pd.DataFrame([
        {"timestamp": datetime(2026, 8, 1), "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
    ])
    setup = scanner.evaluate_series("NSE:INFY-EQ", df)
    assert setup is None
