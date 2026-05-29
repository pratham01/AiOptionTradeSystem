import pandas as pd
import numpy as np

class PreBreakoutPredictor:
    """
    Identifies momentum stocks before they break out on the chart.
    Uses TTM Squeeze logic, Volume Accumulation, and flat price action detection.
    """

    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyzes a DataFrame (usually 15m) for pre-breakout accumulation.
        Returns a dictionary with scores and flags.
        """
        if len(df) < max(self.bb_length, self.kc_length, 20):
            return {"is_squeezed": False, "accumulation_score": 0.0, "predictive_direction": None}

        # 1. Volatility Squeeze (TTM Squeeze proxy)
        df_calc = df.copy()
        
        # Bollinger Bands
        df_calc['sma'] = df_calc['close'].rolling(window=self.bb_length).mean()
        df_calc['std'] = df_calc['close'].rolling(window=self.bb_length).std()
        df_calc['bb_upper'] = df_calc['sma'] + (self.bb_mult * df_calc['std'])
        df_calc['bb_lower'] = df_calc['sma'] - (self.bb_mult * df_calc['std'])

        # Keltner Channels
        df_calc['tr'] = np.maximum(
            df_calc['high'] - df_calc['low'],
            np.maximum(
                abs(df_calc['high'] - df_calc['close'].shift()),
                abs(df_calc['low'] - df_calc['close'].shift())
            )
        )
        df_calc['atr'] = df_calc['tr'].rolling(window=self.kc_length).mean()
        df_calc['kc_upper'] = df_calc['sma'] + (self.kc_mult * df_calc['atr'])
        df_calc['kc_lower'] = df_calc['sma'] - (self.kc_mult * df_calc['atr'])

        # Squeeze On = BB inside KC
        df_calc['squeeze_on'] = (df_calc['bb_lower'] > df_calc['kc_lower']) & (df_calc['bb_upper'] < df_calc['kc_upper'])
        
        # 2. Volume Accumulation (Proxy for CVD)
        # Up volume if close > open, down if close < open
        df_calc['vol_signed'] = np.where(df_calc['close'] > df_calc['open'], df_calc['volume'], 
                                np.where(df_calc['close'] < df_calc['open'], -df_calc['volume'], 0))
        
        # Cumulative Volume Delta over last 20 periods
        df_calc['cvd_20'] = df_calc['vol_signed'].rolling(20).sum()
        
        # Price change over last 20 periods
        df_calc['price_change_20'] = df_calc['close'] - df_calc['close'].shift(20)
        
        latest = df_calc.iloc[-1]
        
        is_squeezed = bool(latest['squeeze_on'])
        
        # Accumulation Score:
        # If price is relatively flat (change is small relative to ATR) but CVD is highly positive/negative
        accumulation_score = 0.0
        predictive_direction = None
        
        if pd.notna(latest['cvd_20']) and pd.notna(latest['price_change_20']) and latest['atr'] > 0:
            price_flatness = abs(latest['price_change_20']) / (latest['atr'] * 20)
            
            # Normalize CVD by average volume
            avg_vol_20 = df_calc['volume'].rolling(20).mean().iloc[-1]
            if avg_vol_20 > 0:
                cvd_ratio = latest['cvd_20'] / (avg_vol_20 * 20) # Ratio of net volume
                
                if price_flatness < 0.2: # Price is very flat
                    if cvd_ratio > 0.15: # Significant positive accumulation
                        accumulation_score = min(1.0, cvd_ratio * 3)
                        predictive_direction = "CALL"
                    elif cvd_ratio < -0.15: # Significant distribution
                        accumulation_score = min(1.0, abs(cvd_ratio) * 3)
                        predictive_direction = "PUT"
                        
        # Squeeze boosts the score
        if is_squeezed and accumulation_score > 0:
            accumulation_score = min(1.0, accumulation_score + 0.3)
            
        return {
            "is_squeezed": is_squeezed,
            "accumulation_score": round(accumulation_score, 3),
            "predictive_direction": predictive_direction,
            "cvd_ratio": round(cvd_ratio, 3) if 'cvd_ratio' in locals() else 0.0
        }
