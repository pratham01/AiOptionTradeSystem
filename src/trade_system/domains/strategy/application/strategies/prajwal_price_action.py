from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass

from trade_system.domains.trading.domain.models.legacy import Signal
from trade_system.domains.strategy.application.strategies.base import BaseStrategy
from trade_system.domains.strategy.application.indicators.support_resistance_channels import SupportResistanceChannelDetector, SRChannel

@dataclass
class TradeSuggestion:
    action: str
    entry: float
    stop_loss: float
    target: float
    reason: str
    channels: list[SRChannel]

class PrajwalPriceActionStrategy(BaseStrategy):
    name = "prajwal_price_action"

    def __init__(self, pivot_period: int = 10, proximity_pct: float = 0.2):
        self.pivot_period = pivot_period
        self.proximity_pct = proximity_pct
        self.detector = SupportResistanceChannelDetector(
            pivot_period=self.pivot_period,
            channel_width_pct=2,  # Tighter channels for cleaner levels
            min_strength=1,
            max_num_sr=6,
            loopback=290
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.prepare(data)

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        return f"Prajwal Setup | {symbol} | Price: {row['close']:.2f}"

    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        df = candles.copy().sort_values("timestamp").reset_index(drop=True)
        return df

    def analyze_setup(self, window: pd.DataFrame) -> TradeSuggestion | None:
        if len(window) < 2 * self.pivot_period + 2:
            return None
            
        snapshots = self.detector.calculate(window)
        if not snapshots:
            return None
            
        last_snap = snapshots[-1]
        last_bar = window.iloc[-1]
        
        close = float(last_bar['close'])
        open_ = float(last_bar['open'])
        high = float(last_bar['high'])
        low = float(last_bar['low'])
        
        body_size = abs(close - open_)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        total_size = high - low
        
        if total_size == 0:
            return None

        # Determine candlestick shape
        is_hammer = lower_wick > 2 * body_size and upper_wick < body_size
        is_shooting_star = upper_wick > 2 * body_size and lower_wick < body_size
        is_strong_bullish = close > open_ and body_size > 0.6 * total_size
        is_strong_bearish = close < open_ and body_size > 0.6 * total_size

        proximity = close * (self.proximity_pct / 100.0)
        
        for ch in last_snap.channels:
            # Reversal logic
            if ch.channel_type == "support":
                # If price is near support and prints a hammer or strong bullish candle
                if abs(low - ch.high) <= proximity or (low <= ch.high and close > ch.high):
                    if is_hammer or is_strong_bullish:
                        # Target is the next resistance above
                        target = close * 1.01  # default 1% target
                        resistances = [c.low for c in last_snap.channels if c.channel_type == "resistance" and c.low > close]
                        if resistances:
                            target = min(resistances)
                            
                        # Stop loss just below support
                        sl = ch.low * 0.999
                        
                        # Only take if RR >= 1.5
                        if (target - close) > 1.5 * (close - sl):
                            return TradeSuggestion(
                                action="BUY", 
                                entry=close,
                                stop_loss=sl,
                                target=target,
                                reason=f"Bullish rejection off Support ({ch.low:.2f}-{ch.high:.2f}).",
                                channels=last_snap.channels
                            )
            
            elif ch.channel_type == "resistance":
                # If price is near resistance and prints a shooting star or strong bearish
                if abs(high - ch.low) <= proximity or (high >= ch.low and close < ch.low):
                    if is_shooting_star or is_strong_bearish:
                        # Target is next support below
                        target = close * 0.99
                        supports = [c.high for c in last_snap.channels if c.channel_type == "support" and c.high < close]
                        if supports:
                            target = max(supports)
                            
                        # Stop loss just above resistance
                        sl = ch.high * 1.001
                        
                        if (close - target) > 1.5 * (sl - close):
                            return TradeSuggestion(
                                action="SELL", 
                                entry=close,
                                stop_loss=sl,
                                target=target,
                                reason=f"Bearish rejection off Resistance ({ch.low:.2f}-{ch.high:.2f}).",
                                channels=last_snap.channels
                            )
                        
        # Breakout logic
        if last_snap.break_event:
            ev = last_snap.break_event
            if ev.break_type == "resistance_broken" and is_strong_bullish:
                sl = ev.level_low * 0.999
                target = close + 2 * (close - sl)
                return TradeSuggestion(
                    action="BUY", 
                    entry=close,
                    stop_loss=sl,
                    target=target,
                    reason=f"Strong Breakout above Resistance ({ev.level_low:.2f}-{ev.level_high:.2f}).",
                    channels=last_snap.channels
                )
            if ev.break_type == "support_broken" and is_strong_bearish:
                sl = ev.level_high * 1.001
                target = close - 2 * (sl - close)
                return TradeSuggestion(
                    action="SELL", 
                    entry=close,
                    stop_loss=sl,
                    target=target,
                    reason=f"Strong Breakdown below Support ({ev.level_low:.2f}-{ev.level_high:.2f}).",
                    channels=last_snap.channels
                )

        return None

    def on_bar(self, window: pd.DataFrame):
        suggestion = self.analyze_setup(window)
        if suggestion:
            return Signal(action=suggestion.action, reason=suggestion.reason)
        return None
