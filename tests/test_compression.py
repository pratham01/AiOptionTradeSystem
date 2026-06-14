import pytest
import pandas as pd
import numpy as np
from trade_system.application.indicators.compression import CompressionIndicator

def test_compression_indicator_calculation():
    # Construct 20 bars of data
    # We want a pattern that creates an inside bar and an NR7 candle at the end
    opens =   [100.0] * 20
    highs =   [102.0] * 20
    lows =    [98.0] * 20
    closes =  [100.0] * 20
    
    # Make some typical variation to establish ATR
    for i in range(15):
        highs[i] = 105.0 if i % 2 == 0 else 102.0
        lows[i] = 95.0 if i % 2 == 0 else 98.0
        closes[i] = 100.0
        
    # Inside bar at index 18
    # Previous bar (17): High=102, Low=98
    highs[17] = 102.0
    lows[17] = 98.0
    # Current bar (18): High=101, Low=99 (completely inside previous)
    highs[18] = 101.0
    lows[18] = 99.0
    
    # NR7 bar at index 19 (range of 0.2, should be smallest in last 7 bars)
    # Let's set current range to be very small
    highs[19] = 100.1
    lows[19] = 99.9
    
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000.0] * 20
    })
    
    indicator = CompressionIndicator(atr_period=10, lookback=3)
    result = indicator.calculate(df)
    
    # Verify columns exist
    assert "atr" in result.columns
    assert "nr7" in result.columns
    assert "inside_bar" in result.columns
    assert "is_compressed" in result.columns
    assert "range_compression" in result.columns
    
    # Verify inside bar detection at index 18
    assert result.loc[18, "inside_bar"] == True
    
    # Verify NR7 detection at index 19
    assert result.loc[19, "nr7"] == True
    
    # Verify combined compression signal
    assert result.loc[18, "is_compressed"] == True
    assert result.loc[19, "is_compressed"] == True
