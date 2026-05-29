import pandas as pd
import numpy as np
from trade_system.application.indicators.volumetric_order_flow import VolumetricOrderFlowDetector

def test_volumetric_order_flow_detector():
    # 1. Create a mock DataFrame with a clear breakout structure
    # Pivot length = 3
    n = 100
    timestamps = pd.date_range("2026-05-26 09:15:00", periods=n, freq="15min")
    
    # Initialize baseline prices around 100
    opens = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    closes = [100.0] * n
    volumes = [1000.0] * n
    
    # Setup a pivot high at index 10 (confirmed at index 13)
    # High at 10 should be 105.0. Highs around it should be lower.
    highs[10] = 105.0
    opens[10] = 102.0
    closes[10] = 103.0
    lows[10] = 101.0
    
    # Keep prices elevated from index 15 to 21 to prevent premature mitigation
    for idx in range(15, 22):
        opens[idx] = 103.0
        highs[idx] = 105.0
        lows[idx] = 102.0
        closes[idx] = 104.0
        
    # Setup a crossover (bullish breakout) at index 15: close rises to 106.0
    closes[15] = 106.0
    highs[15] = 107.0
    opens[15] = 103.0
    
    # Setup a manipulation sweep at index 18:
    # Low of the bullish OB is 101.0 (lows[10]).
    # Price dips to 100.0 (below 101.0) but closes at 102.0 (above 101.0).
    lows[18] = 100.0
    closes[18] = 102.0
    volumes[18] = 2500.0  # high volume sweep
    
    # Setup mitigation at index 22: close drops to 99.5 (below 101.0)
    opens[22] = 102.0
    highs[22] = 103.0
    lows[22] = 99.0
    closes[22] = 99.5
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    
    detector = VolumetricOrderFlowDetector(
        pivot_length=3,
        vol_lookback=10,
        max_obs=5,
        hide_overlapping=False,
        show_manipulation=True,
        manip_size=1.0
    )
    
    history = detector.calculate(df)
    
    assert len(history) == n
    
    # At index 15, we expect a bullish order block to be created and active
    # The order block should have start_idx = 10, high = 105, low = 101
    active_at_15 = history[15]
    assert len(active_at_15) == 1
    ob = active_at_15[0]
    assert ob.is_bullish is True
    assert ob.start_idx == 10
    assert ob.high == 105.0
    assert ob.low == 101.0
    assert ob.label_text in ["BOS", "CHoCH"]
    
    # At index 18, we expect a sweep to be recorded on the active order block
    active_at_18 = history[18]
    assert len(active_at_18) == 1
    ob_at_18 = active_at_18[0]
    assert len(ob_at_18.manipulations) == 1
    sweep = ob_at_18.manipulations[0]
    assert sweep["type"] == "Bullish Sweep"
    assert sweep["price"] == 100.0
    assert sweep["volume"] == 2500.0
    
    # At index 22, the price closes at 99.5. This:
    # 1. Mitigates and removes the bullish order block (ob.low = 101.0).
    # 2. Creates a new bearish order block because of the crossunder of the pivot low at index 18 (100.0).
    active_at_22 = history[22]
    assert len(active_at_22) == 1
    new_ob = active_at_22[0]
    assert new_ob.is_bullish is False
    assert new_ob.start_idx == 18  # Pivot low from index 18
    assert new_ob.high == 105.0   # highs[18] = 105.0
    assert new_ob.low == 100.0    # lows[18] = 100.0
    assert new_ob.label_text == "CHoCH"  # direction changed from bullish to bearish
