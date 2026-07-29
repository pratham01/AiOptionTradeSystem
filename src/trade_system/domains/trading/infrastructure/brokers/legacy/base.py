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

    @abstractmethod
    def place_order(self, symbol: str, side: int, quantity: int, order_type: int, product_type: str, price: float = 0.0, stoploss: float = 0.0) -> Dict[str, Any]:
        """Places a new order."""
        pass

    @abstractmethod
    def modify_order(self, order_id: str, quantity: int, order_type: int, price: float = 0.0) -> Dict[str, Any]:
        """Modifies an existing open order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancels an existing open order."""
        pass
        
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Gets all current positions."""
        pass
        
    @abstractmethod
    def get_funds(self) -> Dict[str, Any]:
        """Gets account funding and margin limits."""
        pass
