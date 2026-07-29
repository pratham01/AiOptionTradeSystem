from __future__ import annotations

import pandas as pd

from trade_system.domains.trading.domain.models.legacy import Signal
from trade_system.domains.strategy.application.strategies.base import BaseStrategy
from trade_system.domains.strategy.application.indicators.vwap import VWAPIndicator
from trade_system.domains.strategy.application.indicators.bollinger_bands import BollingerBandsDetector


class EmaVwapStrategy(BaseStrategy):
    name = "ema_vwap"

    def __init__(self, ema_period: int = 9, risk_reward_ratio: float | None = None) -> None:
        self.ema_period = ema_period
        self.risk_reward_ratio = risk_reward_ratio
        self.vwap_indicator = VWAPIndicator()
        self.bb_detector = BollingerBandsDetector()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(data)
        return df

    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        return f"EMA VWAP | {symbol} | Price: {row['close']:.2f}"

    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        df = candles.copy().sort_values("timestamp").reset_index(drop=True)
        # Calculate VWAP
        df = self.vwap_indicator.calculate(df)
        
        # Calculate EMA
        df["ema"] = df["close"].ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df = self.bb_detector.compute(df)
        
        return df

    def on_bar(self, window: pd.DataFrame):
        if len(window) < 2:
            return None
            
        last = window.iloc[-1]
        prev = window.iloc[-2]
        
        if pd.isna(last.get("timestamp")) or pd.isna(last.get("vwap")) or pd.isna(last.get("ema")):
            return None
            
        timestamp = pd.to_datetime(last["timestamp"], format="mixed")
        
        # Determine EOD exit for intraday mode
        if timestamp.hour == 15 and timestamp.minute >= 15:
            return Signal(action="EXIT", reason="Intraday EOD Exit (15:15)")
            
        # Entry Logic
        # LONG: EMA > VWAP, Price closes above both, AND Bollinger phase is NOT Squeeze or Distribution.
        if prev["close"] <= prev["vwap"] and last["close"] > last["vwap"] and last["close"] > last["ema"] and last["ema"] > last["vwap"]:
            if last.get("bb_phase") not in ["SQUEEZE", "DISTRIBUTION"]:
                return Signal(action="BUY", reason="Price crossed above VWAP during Volatility Expansion")
            
        # SHORT: EMA < VWAP, Price closes below both, AND Bollinger phase is NOT Squeeze or Accumulation.
        if prev["close"] >= prev["vwap"] and last["close"] < last["vwap"] and last["close"] < last["ema"] and last["ema"] < last["vwap"]:
            if last.get("bb_phase") not in ["SQUEEZE", "ACCUMULATION"]:
                return Signal(action="SELL", reason="Price crossed below VWAP during Volatility Expansion")
            
        # Exit Logic
        # Trailing Stop: Exit LONG if price closes below VWAP
        if prev["close"] > prev["vwap"] and last["close"] < last["vwap"]:
            return Signal(action="EXIT", reason="Trailing stop: Price broke below VWAP")
            
        # Trailing Stop: Exit SHORT if price closes above VWAP
        if prev["close"] < prev["vwap"] and last["close"] > last["vwap"]:
            return Signal(action="EXIT", reason="Trailing stop: Price broke above VWAP")
            
        return None
