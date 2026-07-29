import pandas as pd
import numpy as np
import pytest
from trade_system.domains.strategy.application.indicators.bollinger_bands import BollingerBandsDetector, BBPhase

@pytest.fixture
def sample_price_data():
    """Generates synthetic price data for testing."""
    # 200 bars of normal price action
    close = np.linspace(100, 150, 200)
    # Add some random noise
    np.random.seed(42)
    noise = np.random.normal(0, 2, 200)
    close = close + noise
    
    # Create a squeeze at the end (flat price, low volatility)
    squeeze_close = np.linspace(150, 150, 20)
    squeeze_noise = np.random.normal(0, 0.1, 20)
    squeeze_close = squeeze_close + squeeze_noise
    
    # Create a breakout
    breakout_close = np.array([152, 155, 160, 165])
    
    full_close = np.concatenate([close, squeeze_close, breakout_close])
    
    df = pd.DataFrame({
        'close': full_close,
        'high': full_close + 1,
        'low': full_close - 1,
        'volume': np.random.randint(1000, 5000, len(full_close))
    })
    return df

def test_bollinger_bands_calculation(sample_price_data):
    detector = BollingerBandsDetector(period=20, std_dev=2.0)
    df = detector.compute(sample_price_data)
    
    # Check that columns were added
    assert 'bb_basis' in df.columns
    assert 'bb_upper' in df.columns
    assert 'bb_lower' in df.columns
    assert 'bbw' in df.columns
    assert 'bb_percent_b' in df.columns
    assert 'bb_phase' in df.columns
    assert 'trend_state' in df.columns
    assert 'is_walking_upper_band' in df.columns
    assert 'is_walking_lower_band' in df.columns
    assert 'volatility_explosion_score' in df.columns
    
    # Spot check math on a specific row
    row = df.iloc[50]
    expected_basis = df['close'].iloc[31:51].mean()
    expected_std = df['close'].iloc[31:51].std(ddof=0)
    
    np.testing.assert_almost_equal(row['bb_basis'], expected_basis)
    np.testing.assert_almost_equal(row['bb_upper'], expected_basis + (2.0 * expected_std))
    np.testing.assert_almost_equal(row['bb_lower'], expected_basis - (2.0 * expected_std))
    
    expected_bbw = (row['bb_upper'] - row['bb_lower']) / row['bb_basis']
    np.testing.assert_almost_equal(row['bbw'], expected_bbw)
    
    expected_pb = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
    np.testing.assert_almost_equal(row['bb_percent_b'], expected_pb)

def test_bb_phases(sample_price_data):
    detector = BollingerBandsDetector(period=20, squeeze_lookback=100, squeeze_percentile=10.0)
    df = detector.compute(sample_price_data)
    
    # The squeeze occurs from index 200 to 219 (extremely low volatility)
    # The breakout occurs at index 220 to 223
    
    # Check Squeeze
    squeeze_phase = df['bb_phase'].iloc[210:219]
    assert all(p == BBPhase.SQUEEZE.value for p in squeeze_phase), "Failed to detect squeeze phase"
    
    # Check Breakout (Expansion Bullish)
    breakout_phase = df['bb_phase'].iloc[-1]
    assert breakout_phase == BBPhase.EXPANSION_BULLISH.value, "Failed to detect bullish expansion breakout"
