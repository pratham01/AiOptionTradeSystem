"""
Trade System - Autonomous trading platform.
"""

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.data.nse_universe import NSE_UNIVERSE
from trade_system import domains

__version__ = "1.0.0"

__all__ = [
    "Settings",
    "NSE_UNIVERSE",
    "domains",
]
