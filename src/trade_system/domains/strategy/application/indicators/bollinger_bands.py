import pandas as pd
import numpy as np
from enum import Enum

class BBPhase(Enum):
    SQUEEZE = "SQUEEZE"
    EXPANSION_BULLISH = "EXPANSION_BULLISH"
    EXPANSION_BEARISH = "EXPANSION_BEARISH"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    NEUTRAL = "NEUTRAL"

class BollingerBandsDetector:
    """
    Computes Bollinger Bands and identifies market phases like Squeeze, Accumulation, and Distribution.
    """
    def __init__(self, period: int = 20, std_dev: float = 2.0, squeeze_lookback: int = 100, squeeze_percentile: float = 10.0):
        self.period = period
        self.std_dev = std_dev
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Bollinger Bands, BBW, %b, and identifies the current phase.
        Input dataframe must have 'close', 'high', 'low', 'volume'.
        """
        if len(df) < self.period:
            return df
            
        result = df.copy()
        
        # Standard Bollinger Bands
        result['bb_basis'] = result['close'].rolling(window=self.period).mean()
        std = result['close'].rolling(window=self.period).std(ddof=0)
        
        result['bb_upper'] = result['bb_basis'] + (self.std_dev * std)
        result['bb_lower'] = result['bb_basis'] - (self.std_dev * std)
        
        # Bollinger Band Width (BBW)
        result['bbw'] = (result['bb_upper'] - result['bb_lower']) / result['bb_basis']
        
        # %b (Percent B) - location of price relative to bands
        # Handle division by zero if upper == lower
        band_range = result['bb_upper'] - result['bb_lower']
        band_range = band_range.replace(0, np.nan)
        result['bb_percent_b'] = (result['close'] - result['bb_lower']) / band_range
        
        # 1. Trend State (Using slope of the basis SMA)
        # Calculate slope over 3 periods as a % change
        basis_roc = result['bb_basis'].pct_change(periods=3) * 100
        result['trend_state'] = np.where(basis_roc > 0.1, 'UP', 
                                np.where(basis_roc < -0.1, 'DOWN', 'FLAT'))
        
        # 2. "Walking the Band" Detection (Momentum for Option Buyers)
        # True if price has been riding the upper/lower bands (>0.9 or <0.1) for 3+ consecutive bars
        is_high_b = result['bb_percent_b'] > 0.9
        result['is_walking_upper_band'] = is_high_b.rolling(window=3).sum() >= 3
        
        is_low_b = result['bb_percent_b'] < 0.1
        result['is_walking_lower_band'] = is_low_b.rolling(window=3).sum() >= 3
        
        # 3. Volatility Explosion Score (Gamma/Vega Spikes)
        # Rate of change of BBW multiplied by Volume Surge (relative to 20-period average)
        bbw_roc = result['bbw'].pct_change(periods=1).clip(lower=0) # Only care about expansion
        avg_vol = result['volume'].rolling(window=self.period).mean().replace(0, np.nan)
        vol_surge = (result['volume'] / avg_vol).fillna(1.0)
        result['volatility_explosion_score'] = bbw_roc * vol_surge
        
        # Phase Identification
        result['bb_phase'] = self._identify_phases(result)
        
        return result

    def _identify_phases(self, df: pd.DataFrame) -> pd.Series:
        phases = pd.Series(BBPhase.NEUTRAL.value, index=df.index, dtype='object')
        
        if len(df) < self.squeeze_lookback:
            # We need enough historical BBW data to determine the squeeze percentile
            return phases

        # 1. Identify Squeeze (BBW in the lowest percentile of historical lookback)
        rolling_bbw_min = df['bbw'].rolling(window=self.squeeze_lookback, min_periods=self.squeeze_lookback).quantile(self.squeeze_percentile / 100.0)
        is_squeeze = df['bbw'] <= rolling_bbw_min
        
        # 2. Identify Expansion (BBW expanding, breaking out of bands)
        # Expansion Bullish: price > upper band
        # Expansion Bearish: price < lower band
        is_expansion_bullish = (~is_squeeze) & (df['bb_percent_b'] > 1.0) & (df['bbw'] > df['bbw'].shift(1))
        is_expansion_bearish = (~is_squeeze) & (df['bb_percent_b'] < 0.0) & (df['bbw'] > df['bbw'].shift(1))
        
        # 3. Identify Accumulation / Distribution
        # Accumulation: Price is oscillating near lower band (0 to 0.3) without breaking down, volume often dropping
        # Distribution: Price is oscillating near upper band (0.7 to 1.0) without breaking up
        # PRO FILTER: Only flag mean-reversion zones if the trend is FLAT. If UP/DOWN, it's a trend pullback, not accumulation.
        is_flat = df['trend_state'] == 'FLAT'
        is_accumulation = (~is_squeeze) & (~is_expansion_bearish) & (df['bb_percent_b'] >= 0.0) & (df['bb_percent_b'] <= 0.3) & is_flat
        is_distribution = (~is_squeeze) & (~is_expansion_bullish) & (df['bb_percent_b'] >= 0.7) & (df['bb_percent_b'] <= 1.0) & is_flat

        # Apply labels in order of priority (Squeeze/Expansion overrides Accumulation/Distribution)
        phases[is_accumulation] = BBPhase.ACCUMULATION.value
        phases[is_distribution] = BBPhase.DISTRIBUTION.value
        phases[is_squeeze] = BBPhase.SQUEEZE.value
        phases[is_expansion_bullish] = BBPhase.EXPANSION_BULLISH.value
        phases[is_expansion_bearish] = BBPhase.EXPANSION_BEARISH.value
        
        return phases
