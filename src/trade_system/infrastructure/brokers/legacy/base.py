from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd

class BaseBroker(ABC):
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticates with the broker."""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, resolution: str, range_from: str, range_to: str) -> Optional[pd.DataFrame]:
        """Fetches historical OHLC data."""
        pass

    @abstractmethod
    def subscribe_to_realtime_data(self, symbols: List[str], callback: Any):
        """Subscribes to real-time data."""
        pass

    @abstractmethod
    def get_option_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches option chain data."""
        pass
