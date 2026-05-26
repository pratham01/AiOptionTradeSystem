from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from trade_system.application.strategies.base import BaseStrategy
from trade_system.application.indicators.volume_profile import VolumeProfileIndicator
from trade_system.core import Signal, VolumeProfileSummary
from trade_system.application.indicators.volume_delta import VolumeDeltaIndicator

LOGGER = logging.getLogger(__name__)

class VolumeProfileStrategy(BaseStrategy):
    name = "volume_profile"

    def __init__(
        self,
        price_step: float = 10.0,
        volume_multiplier: float = 1.2,
        value_area_pct: float = 0.70,
        confirm_with_delta: bool = True
    ) -> None:
        self.price_step = price_step
        self.volume_multiplier = volume_multiplier
        self.value_area_pct = value_area_pct
        self.confirm_with_delta = confirm_with_delta
        self.vp_indicator = VolumeProfileIndicator(price_step=price_step, value_area_pct=value_area_pct)
        self.vd_indicator = VolumeDeltaIndicator()
        self._profiles: dict[object, VolumeProfileSummary] = {}

    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        df = candles.copy().sort_values("timestamp").reset_index(drop=True)
        df.columns = [col.lower() for col in df.columns]
        df['date'] = df['timestamp'].dt.date
        
        # 1. Pre-calculate daily profiles
        dates = sorted(df['date'].unique())
        for d in dates:
            day_data = df[df['date'] == d]
            self._profiles[d] = self.vp_indicator.calculate(day_data)
        
        # 2. Calculate Volume Delta
        df = self.vd_indicator.calculate(df)
        
        # 3. Calculate moving average volume
        df['avg_vol'] = df['volume'].rolling(20).mean()
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compatibility with BaseStrategy."""
        prepared = self.prepare(data)
        prepared['final_signal'] = 0
        
        # For each bar, check if a signal would be generated
        for i in range(2, len(prepared)):
            row = prepared.iloc[i]
            prev_row = prepared.iloc[i-1]
            
            # Get previous trading day's profile
            curr_date = row['date']
            sorted_dates = sorted(self._profiles.keys())
            try:
                curr_idx = sorted_dates.index(curr_date)
                if curr_idx == 0:
                    continue
                prev_date = sorted_dates[curr_idx - 1]
                profile = self._profiles[prev_date]
            except (ValueError, IndexError):
                continue

            if not profile:
                continue

            # VAH Breakout (Long)
            is_breakout = row['close'] > profile.value_area_high and prev_row['close'] <= profile.value_area_high
            vol_confirmed = row['volume'] > row['avg_vol'] * self.volume_multiplier
            delta_confirmed = not self.confirm_with_delta or row['delta'] > 0

            if is_breakout and vol_confirmed and delta_confirmed:
                prepared.loc[prepared.index[i], 'final_signal'] = 1
                prepared.loc[prepared.index[i], 'reason'] = f"VAH Breakout ({profile.value_area_high:.1f})"

            # VAL Breakdown (Short)
            is_breakdown = row['close'] < profile.value_area_low and prev_row['close'] >= profile.value_area_low
            delta_short_confirmed = not self.confirm_with_delta or row['delta'] < 0

            if is_breakdown and vol_confirmed and delta_short_confirmed:
                prepared.loc[prepared.index[i], 'final_signal'] = -1
                prepared.loc[prepared.index[i], 'reason'] = f"VAL Breakdown ({profile.value_area_low:.1f})"
                
        return prepared

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        reason = row.get('reason', 'Profile setup')
        return f"VP Strategy | {symbol} | {reason} | Price: {row['close']}"
