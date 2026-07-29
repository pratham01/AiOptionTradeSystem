"""
Sniper Reversal Strategy Scanner

Detects a setup where price goes into consolidation (sideways), volume steadily dries up (supply exhaustion), 
and then an explosive volume-backed reversal or breakout occurs.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

class EquitySniperScanner:
    """
    Scans for the Sniper Reversal in standard OHLCV equity/index data.
    """
    
    def __init__(self, lookback_window: int = 5, volume_dryup_threshold: float = -0.05, breakout_vol_multiplier: float = 1.5):
        """
        :param lookback_window: Number of bars for the consolidation phase (before the breakout).
        :param volume_dryup_threshold: The maximum allowed slope (should be negative) for volume during consolidation.
        :param breakout_vol_multiplier: How much the breakout bar's volume must exceed the consolidation average.
        """
        self.lookback_window = lookback_window
        self.volume_dryup_threshold = volume_dryup_threshold
        self.breakout_vol_multiplier = breakout_vol_multiplier

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Expects a DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        Assumes data is sorted chronologically (latest bar at the end).
        """
        if len(df) < self.lookback_window + 1:
            return {"status": "NEUTRAL", "reason": "Not enough data"}

        # Current bar is the breakout trigger candidate
        current_bar = df.iloc[-1]
        
        # Consolidation phase (exclude the current bar)
        consolidation = df.iloc[-(self.lookback_window + 1):-1].copy()

        # 1. Price Consolidation Check
        # Calculate the True Range or simple High-Low range to ensure it's relatively flat
        consolidation["range"] = consolidation["high"] - consolidation["low"]
        avg_range = consolidation["range"].mean()
        
        # 2. Volume Dry-Up Check
        # We want the slope of volume to be negative (decreasing)
        y = consolidation["volume"].values
        x = np.arange(len(y))
        
        if len(y) > 1 and y.mean() > 0:
            # Normalize volume to 0-1 range to get a comparable slope
            y_max = y.max()
            y_norm = y / y_max if y_max > 0 else y
            slope, _ = np.polyfit(x, y_norm, 1)
        else:
            slope = 0.0

        if slope > self.volume_dryup_threshold:
            return {"status": "NEUTRAL", "reason": f"Volume not drying up enough (slope: {slope:.3f})"}

        # 3. Breakout Trigger Check
        avg_vol = consolidation["volume"].mean()
        current_vol = current_bar["volume"]
        
        if current_vol < avg_vol * self.breakout_vol_multiplier:
            return {"status": "NEUTRAL", "reason": f"Volume spike insufficient ({current_vol} vs {avg_vol * self.breakout_vol_multiplier})"}

        # 4. Determine Direction
        # A strong bullish bar (close near high) or bearish bar (close near low)
        body = current_bar["close"] - current_bar["open"]
        candle_range = current_bar["high"] - current_bar["low"]
        
        if candle_range == 0:
            return {"status": "NEUTRAL", "reason": "No range on trigger bar"}

        # Check if the breakout range is explosive (larger than avg consolidation range)
        if candle_range < avg_range * 1.2:
            return {"status": "NEUTRAL", "reason": "Price range didn't expand on breakout"}

        direction = "BULLISH" if body > 0 else "BEARISH"
        
        return {
            "status": f"{direction} SNIPER REVERSAL",
            "trigger_price": current_bar["close"],
            "volume_slope": round(slope, 3),
            "vol_expansion_ratio": round(current_vol / avg_vol, 2) if avg_vol > 0 else 0.0,
            "avg_consolidation_range": round(avg_range, 2),
            "breakout_range": round(candle_range, 2)
        }


class OptionOISniperScanner:
    """
    Scans for the Sniper Reversal in Options Option Chain (OI) data.
    """
    
    def __init__(self, lookback_window: int = 5, breakout_vol_multiplier: float = 1.5):
        self.lookback_window = lookback_window
        self.breakout_vol_multiplier = breakout_vol_multiplier

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Expects a DataFrame representing a single Option Strike over time.
        Required columns: ['timestamp', 'ltp', 'volume', 'oi', 'option_type']
        """
        if len(df) < self.lookback_window + 1:
            return {"status": "NEUTRAL", "reason": "Not enough data"}

        current_bar = df.iloc[-1]
        consolidation = df.iloc[-(self.lookback_window + 1):-1].copy()

        # 1. Volume Dry-Up Check
        y_vol = consolidation["volume"].values
        x = np.arange(len(y_vol))
        
        if len(y_vol) > 1 and y_vol.mean() > 0:
            y_vol_max = y_vol.max()
            y_vol_norm = y_vol / y_vol_max if y_vol_max > 0 else y_vol
            vol_slope, _ = np.polyfit(x, y_vol_norm, 1)
        else:
            vol_slope = 0.0

        if vol_slope > -0.05:
            return {"status": "NEUTRAL", "reason": f"Option volume not drying up (slope: {vol_slope:.3f})"}

        # 2. Breakout Trigger Check (Volume and Premium)
        avg_vol = consolidation["volume"].mean()
        current_vol = current_bar["volume"]
        
        if current_vol < avg_vol * self.breakout_vol_multiplier:
            return {"status": "NEUTRAL", "reason": f"Option volume spike insufficient ({current_vol} vs {avg_vol * self.breakout_vol_multiplier})"}

        # 3. Premium (LTP) Expansion
        avg_ltp = consolidation["ltp"].mean()
        current_ltp = current_bar["ltp"]
        ltp_change = (current_ltp - avg_ltp) / avg_ltp if avg_ltp > 0 else 0
        
        # Options premium must jump significantly (e.g., > 10% jump)
        if abs(ltp_change) < 0.10:
            return {"status": "NEUTRAL", "reason": "Premium didn't spike significantly"}

        # 4. OI Analysis (The key to institutional moves)
        avg_oi = consolidation["oi"].mean()
        current_oi = current_bar["oi"]
        oi_change = (current_oi - avg_oi) / avg_oi if avg_oi > 0 else 0

        # Determine the narrative based on Option Type + OI + Premium
        option_type = current_bar.get("option_type", "CE")
        
        if oi_change > 0.05:
            action = "Aggressive Long Buildup" if ltp_change > 0 else "Aggressive Short Buildup"
            direction = "BULLISH" if option_type == "CE" and ltp_change > 0 else ("BEARISH" if option_type == "PE" and ltp_change > 0 else "NEUTRAL")
        elif oi_change < -0.05:
            action = "Massive Short Covering" if ltp_change > 0 else "Long Unwinding"
            direction = "BULLISH" if option_type == "CE" and ltp_change > 0 else ("BEARISH" if option_type == "PE" and ltp_change > 0 else "NEUTRAL")
        else:
            action = "Volume Speculation (Flat OI)"
            direction = "BULLISH" if option_type == "CE" and ltp_change > 0 else ("BEARISH" if option_type == "PE" and ltp_change > 0 else "NEUTRAL")

        if direction == "NEUTRAL":
            return {"status": "NEUTRAL", "reason": "Conflicting Premium/OI logic for sniper reversal"}

        return {
            "status": f"{direction} SNIPER REVERSAL ({action})",
            "trigger_ltp": current_ltp,
            "ltp_surge_pct": round(ltp_change * 100, 2),
            "vol_expansion_ratio": round(current_vol / avg_vol, 2) if avg_vol > 0 else 0.0,
            "oi_change_pct": round(oi_change * 100, 2),
            "vol_slope": round(vol_slope, 3)
        }
