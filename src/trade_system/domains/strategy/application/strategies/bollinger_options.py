from __future__ import annotations

import pandas as pd

from trade_system.domains.trading.domain.models.legacy import Signal
from trade_system.domains.strategy.application.strategies.base import BaseStrategy
from trade_system.domains.strategy.application.indicators.bollinger_bands import BollingerBandsDetector, BBPhase


class BollingerOptionStrategy(BaseStrategy):
    name = "bollinger_option_buyer"

    def __init__(
        self,
        mode: str = "intraday",  # 'intraday' or 'swing'
        period: int = 20,
        std_dev: float = 2.0,
        squeeze_lookback: int = 100,
        squeeze_percentile: float = 10.0,
        min_volatility_score: float = 1.2,
    ) -> None:
        self.mode = mode
        self.period = period
        self.std_dev = std_dev
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile
        self.min_volatility_score = min_volatility_score
        
        self.detector = BollingerBandsDetector(
            period=self.period,
            std_dev=self.std_dev,
            squeeze_lookback=self.squeeze_lookback,
            squeeze_percentile=self.squeeze_percentile,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(data)
        # Simplified for plotting/returning state if needed
        return df

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        return f"Bollinger Breakout | {symbol} | Price: {row['close']:.2f}"

    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        df = candles.copy().sort_values("timestamp").reset_index(drop=True)
        # Apply the Bollinger Bands indicator
        return self.detector.compute(df)

    def on_bar(self, window: pd.DataFrame):
        if len(window) < 2:
            return None
            
        last = window.iloc[-1]
        prev = window.iloc[-2]
        
        # If timestamp is missing or na, skip
        if pd.isna(last.get("timestamp")):
            return None
            
        timestamp = pd.to_datetime(last["timestamp"], format="mixed")
        
        # Determine EOD exit for intraday mode
        if self.mode == "intraday" and timestamp.hour == 15 and timestamp.minute >= 15:
            return Signal(action="EXIT", reason="Intraday EOD Exit (15:15)")
            
        # 1. Entry Logic: Transition from Squeeze to Expansion Bullish/Bearish
        # Option Buyer wants Momentum -> High Volatility Explosion Score
        
        # Was previously in a squeeze or accumulation?
        prev_tight = prev["bb_phase"] in [BBPhase.SQUEEZE.value, BBPhase.ACCUMULATION.value]
        
        # Bullish Breakout
        if prev_tight and last["bb_phase"] == BBPhase.EXPANSION_BULLISH.value:
            if last.get("volatility_explosion_score", 0.0) >= self.min_volatility_score:
                return Signal(action="BUY", reason="Bollinger Bullish Breakout with Vol Surge")
                
        # Bearish Breakout (Proxy by "SELL" which in options would mean buying a Put)
        if prev_tight and last["bb_phase"] == BBPhase.EXPANSION_BEARISH.value:
            if last.get("volatility_explosion_score", 0.0) >= self.min_volatility_score:
                return Signal(action="SELL", reason="Bollinger Bearish Breakout with Vol Surge")
                
        # 2. Exit Logic: Mean Reversion or Loss of Momentum
        
        # Exit Bullish (Long) trade if price closes back below the Upper Band after being outside it
        # OR if it breaks below the basis SMA
        if prev["close"] > prev["bb_upper"] and last["close"] < last["bb_upper"]:
            return Signal(action="EXIT", reason="Reversion below upper band")
            
        if prev["close"] > prev["bb_basis"] and last["close"] < last["bb_basis"]:
            return Signal(action="EXIT", reason="Trend breakdown below SMA20")
            
        # Exit Bearish (Short) trade if price closes back above the Lower Band
        # OR if it breaks above the basis SMA
        if prev["close"] < prev["bb_lower"] and last["close"] > last["bb_lower"]:
            return Signal(action="EXIT", reason="Reversion above lower band")
            
        if prev["close"] < prev["bb_basis"] and last["close"] > last["bb_basis"]:
            return Signal(action="EXIT", reason="Trend breakdown above SMA20")

        return None
