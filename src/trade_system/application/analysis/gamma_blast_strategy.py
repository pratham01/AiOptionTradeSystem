"""
GammaBlastDetector — Strategy logic for high-leverage 0DTE late-day option buying.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time as dt_time
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

class GammaBlastDetector:
    """
    Monitors indices (Nifty, BankNifty, Sensex) on expiry days for volatility
    compression and explosive late-day breakouts.
    """

    def __init__(
        self,
        squeeze_percentile_threshold: float = 25.0,  # Bottom 25% of BB Width = Squeeze
        min_premium: float = 5.0,
        max_premium: float = 25.0,
    ) -> None:
        self.squeeze_percentile_threshold = squeeze_percentile_threshold
        self.min_premium = min_premium
        self.max_premium = max_premium

    @staticmethod
    def is_expiry_day(symbol: str, current_date: date) -> bool:
        """
        Check if today is the expiry day for the given index symbol.
        Standard rules:
          - NSE:NIFTY50-INDEX   -> Thursday (weekday 3)
          - NSE:NIFTYBANK-INDEX -> Wednesday (weekday 2)
          - BSE:SENSEX-INDEX    -> Friday (weekday 4)
        """
        # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4
        weekday = current_date.weekday()
        
        if "NIFTY50" in symbol:
            return weekday == 1  # Tuesday in database weekly contracts
        elif "NIFTYBANK" in symbol:
            return weekday == 2  # Wednesday
        elif "SENSEX" in symbol:
            return weekday == 3  # Thursday in database weekly contracts
            
        return False

    def check_volatility_squeeze(self, df: pd.DataFrame, length: int = 20, lookback: int = 100) -> tuple[bool, float]:
        """
        Calculate Bollinger Band Width and determine if it is in a squeeze.
        """
        if len(df) < length:
            return False, 0.0

        close = df["close"]
        sma = close.rolling(length).mean()
        std = close.rolling(length).std()
        
        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        
        # Avoid division by zero
        bb_width = (upper - lower) / sma.replace(0, np.nan)
        bb_width = bb_width.fillna(0.0)
        
        if len(bb_width) < 2:
            return False, 0.0
            
        current_width = float(bb_width.iloc[-1])
        
        # Get historical window for percentile ranking
        hist_widths = bb_width.tail(lookback)
        if len(hist_widths) < 5:
            return False, 0.0
            
        pct_rank = float(np.percentile(hist_widths, self.squeeze_percentile_threshold))
        is_squeezed = current_width <= pct_rank
        
        return is_squeezed, current_width

    def get_consolidation_range(
        self, df_1m: pd.DataFrame, target_date: date, start_time: dt_time = dt_time(14, 0), end_time: dt_time = dt_time(14, 30)
    ) -> tuple[float, float]:
        """
        Determine the consolidation high and low during the pre-blast observation window (2:00 PM - 2:30 PM).
        """
        # Filter to target day & time window
        day_df = df_1m[df_1m["timestamp"].dt.date == target_date]
        obs_df = day_df[(day_df["timestamp"].dt.time >= start_time) & (day_df["timestamp"].dt.time <= end_time)]
        
        if obs_df.empty:
            return 0.0, 0.0
            
        obs_high = float(obs_df["high"].max())
        obs_low = float(obs_df["low"].min())
        
        return obs_high, obs_low

    def evaluate_breakout(
        self,
        current_time: datetime,
        current_spot: float,
        obs_high: float,
        obs_low: float,
        is_squeezed: bool,
    ) -> tuple[str | None, float]:
        """
        Identify breakout direction. Returns (direction, trigger_price).
        Only active between 2:30 PM and 3:05 PM.
        """
        # Time window guard
        trade_start = dt_time(14, 30)
        trade_end = dt_time(15, 5)
        
        if not (trade_start <= current_time.time() <= trade_end):
            return None, 0.0
            
        if obs_high <= 0 or obs_low <= 0:
            return None, 0.0
            
        # Volatility squeeze check (must be coiled or recently coiled)
        # Note: If not squeezed, breakouts are prone to whipsaws.
        if not is_squeezed:
            return None, 0.0

        # Breakout checks
        if current_spot > obs_high:
            return "CALL", obs_high
        elif current_spot < obs_low:
            return "PUT", obs_low
            
        return None, 0.0

    def select_0dte_option(
        self,
        symbol: str,
        spot_price: float,
        direction: str,
        option_chain_df: pd.DataFrame,
    ) -> tuple[str, float] | None:
        """
        Identify the correct expiring 0DTE option contract.
        Looks for the strike closest to OTM that matches our target premium range (₹5 - ₹25).
        """
        if option_chain_df.empty:
            return None

        # Filter by direction
        opt_type = "CE" if direction == "CALL" else "PE"
        opts = option_chain_df[option_chain_df["option_type"] == opt_type].copy()
        
        if opts.empty:
            return None
            
        # Filter to options within the cheap 0DTE premium band (e.g. ₹5 - ₹25)
        qualifying = opts[(opts["ltp"] >= self.min_premium) & (opts["ltp"] <= self.max_premium)].copy()
        
        if qualifying.empty:
            # If nothing matches the exact band, fallback to the closest premium to self.min_premium
            opts["diff"] = (opts["ltp"] - self.min_premium).abs()
            cheapest = opts.sort_values("diff").iloc[0]
            return str(cheapest["symbol"]), float(cheapest["ltp"])

        # Pick the one closest to ATM (which minimizes strike distance to spot)
        qualifying["strike_dist"] = (qualifying["strike"] - spot_price).abs()
        best_opt = qualifying.sort_values("strike_dist").iloc[0]
        
        return str(best_opt["symbol"]), float(best_opt["ltp"])
