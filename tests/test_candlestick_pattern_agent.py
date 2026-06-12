import pytest
import pandas as pd
import numpy as np
from trade_system.application.agent.candlestick_pattern_agent import CandlestickPatternAgent

def test_pattern_detection():
    agent = CandlestickPatternAgent()
    
    # 1. Create a dummy dataframe with a clear Bullish Engulfing pattern at index 3
    # Bullish engulfing requires:
    # prev (index 2): close < open (red)
    # curr (index 3): close > open (green), body is larger than prev body, covers prev body
    timestamps = pd.date_range("2026-05-01", periods=10, freq="D")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open":  [100, 102, 105, 101, 108, 110, 112, 115, 114, 118],
        "high":  [102, 106, 107, 112, 110, 113, 115, 118, 116, 120],
        "low":   [98,  101, 100, 99,  106, 109, 110, 113, 112, 116],
        "close": [101, 104, 101, 110, 109, 111, 113, 114, 115, 119],
        "volume": [1000] * 10
    })
    
    df_res = agent.detect_patterns(df)
    
    # Check index 3 (corresponds to row 4 of dummy data)
    # index 2: open=105, close=101 (red, body = 4)
    # index 3: open=101, close=110 (green, body = 9, covers 105 to 101)
    assert df_res.loc[3, "bullish_engulfing"] == 1
    assert df_res.loc[3, "bearish_engulfing"] == 0

def test_doji_detection():
    agent = CandlestickPatternAgent()
    timestamps = pd.date_range("2026-05-01", periods=6, freq="D")
    
    # Doji requires body / range <= 0.05
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open":  [100, 101, 100, 100.1, 100, 100],
        "high":  [102, 105, 105, 105, 102, 102],
        "low":   [98,  95,  95,  95,  98,  98],
        "close": [101, 100, 100, 100.2, 101, 101],
        "volume": [1000] * 6
    })
    
    df_res = agent.detect_patterns(df)
    # index 2: open=100, close=100 (body=0, range=10) -> Doji
    # index 3: open=100.1, close=100.2 (body=0.1, range=10) -> ratio=0.01 -> Doji
    assert df_res.loc[2, "doji"] == 1
    assert df_res.loc[3, "doji"] == 1

def test_probability_calculation_empty_boundary():
    agent = CandlestickPatternAgent()
    # Querying a non-existent symbol should return empty dictionary gracefully
    res = agent.calculate_probabilities("NSE:NONEXISTENT-EQ")
    assert isinstance(res, dict)
    assert len(res) == 0
