from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates buy/sell signals on the provided OHLC data."""
        pass

    @abstractmethod
    def get_signal_message(self, symbol: str, row: pd.Series) -> str:
        """Returns a formatted message for the signal."""
        pass


class Strategy(ABC):
    @abstractmethod
    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Prepares candle data for the strategy."""
        pass

    @abstractmethod
    def on_bar(self, window: pd.DataFrame):
        """Decides what to do based on the latest candle window."""
        pass
