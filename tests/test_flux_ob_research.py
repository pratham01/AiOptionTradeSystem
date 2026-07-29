from pathlib import Path
import pandas as pd
import numpy as np
from trade_system.domains.strategy.application.indicators.flux_order_blocks import FluxOrderBlockDetector
from trade_system.domains.analysis.application.backtesting.flux_ob_research import FluxOrderBlockResearch

def test_flux_order_block_detector():
    # Construct a mock dataframe with a clear swing pattern
    # Let's create 50 rows of data
    timestamps = pd.date_range("2026-05-26 09:15:00", periods=50, freq="3min")
    
    # We will build a swing high: prices rise up to index 15, then fall down
    closes = [100.0] * 50
    highs = [101.0] * 50
    lows = [99.0] * 50
    volumes = [100.0] * 50
    
    for i in range(15):
        closes[i] = 100.0 + i
        highs[i] = 101.0 + i
        lows[i] = 99.0 + i
        
    # Index 15 is the peak (115 high)
    closes[15] = 114.0
    highs[15] = 115.0
    lows[15] = 113.0
    
    # Prices fall after index 15
    for i in range(16, 30):
        closes[i] = 114.0 - (i - 15)
        highs[i] = 115.0 - (i - 15)
        lows[i] = 113.0 - (i - 15)
        
    # Then price shoots up breaking the swing high at index 35
    for i in range(30, 45):
        closes[i] = closes[29] + (i - 29) * 2
        highs[i] = highs[29] + (i - 29) * 2
        lows[i] = lows[29] + (i - 29) * 2
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    
    detector = FluxOrderBlockDetector(swing_length=5, max_atr_mult=10.0, num_render_blocks=3)
    history = detector.calculate(df)
    
    assert len(history) == len(df)
    
    # There should be order blocks formed and active in the latter part of the history
    any_obs = any(len(obs) > 0 for obs in history)
    assert any_obs, "No order blocks were detected in mock history"

def test_flux_ob_research_runs(tmp_path: Path):
    # Create a mock CSV for a Nifty data year
    file_path = tmp_path / "NSE_NIFTY50-INDEX_3min_2026.csv"
    
    timestamps = pd.date_range("2026-05-26 09:15:00", periods=100, freq="3min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0 + i * 0.1 for i in range(100)],
        "high": [101.0 + i * 0.1 for i in range(100)],
        "low": [99.0 + i * 0.1 for i in range(100)],
        "close": [100.5 + i * 0.1 for i in range(100)],
        "volume": [1000] * 100
    })
    df.to_csv(file_path, index=False)
    
    output_dir = tmp_path / "reports"
    research = FluxOrderBlockResearch(swing_length=5, max_atr_mult=5.0)
    artifacts = research.run(tmp_path, output_dir)
    
    assert artifacts.summary_path.exists()
    assert artifacts.report_path.exists()
    assert not artifacts.summary.empty
