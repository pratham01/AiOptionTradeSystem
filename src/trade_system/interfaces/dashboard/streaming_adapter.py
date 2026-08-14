"""
LiveStateStreamingAdapter — Thread-safe, non-blocking state and streaming data provider for Streamlit dashboards.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.shared.live_state import LiveStateReader, SymbolState

LOGGER = logging.getLogger("StreamingAdapter")


class LiveStateStreamingAdapter:
    """
    Adapter between live background processing and Streamlit frontend.
    Guarantees thread-safe atomic snapshots without database locking contention.
    """
    _instance: Optional[LiveStateStreamingAdapter] = None
    _lock = threading.Lock()

    def __new__(cls) -> LiveStateStreamingAdapter:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.engine = get_engine()
        self.reader = LiveStateReader()
        self._cache_lock = threading.Lock()
        self._price_cache: Dict[str, float] = {}
        self._last_tick_time: Optional[datetime] = None
        self._initialized = True

    def record_tick(self, symbol: str, price: float) -> None:
        """Update in-memory price cache."""
        with self._cache_lock:
            self._price_cache[symbol] = price
            self._last_tick_time = datetime.now()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price from memory cache, falling back to LiveState snapshot."""
        with self._cache_lock:
            if symbol in self._price_cache:
                return self._price_cache[symbol]
        prices = self.reader.get_current_prices([symbol])
        state = prices.get(symbol)
        return float(state.ltp) if state and state.ltp > 0 else None

    def get_stream_health(self) -> Dict[str, Any]:
        """Return real-time stream status for the dashboard header."""
        is_bot_running = self.reader.is_bot_running()
        is_fresh = self.reader.is_fresh()
        last_updated = self._last_tick_time
        now = datetime.now()
        latency_sec = (now - last_updated).total_seconds() if last_updated else None

        is_streaming = (is_bot_running or is_fresh) and (latency_sec is None or latency_sec < 60)

        return {
            "is_bot_running": is_bot_running,
            "is_streaming": is_streaming,
            "last_updated": last_updated.strftime("%H:%M:%S") if last_updated else ("Fresh" if is_fresh else "None"),
            "latency_seconds": round(latency_sec, 1) if latency_sec is not None else None,
            "status_label": "🟢 LIVE STREAM" if is_streaming else ("🟡 CONNECTED" if is_bot_running else "🔴 OFFLINE"),
        }



    def execute_safe_query(self, query_str: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> pd.DataFrame:
        """
        Execute a database read query with retry-on-lock logic to prevent Streamlit UI crashes.
        """
        params = params or {}
        for attempt in range(1, max_retries + 1):
            try:
                with self.engine.connect() as conn:
                    return pd.read_sql(text(query_str), conn, params=params)
            except Exception as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    LOGGER.warning("Database locked during query attempt %d/%d. Retrying...", attempt, max_retries)
                    time.sleep(0.1 * attempt)
                    continue
                LOGGER.error("Safe query failed: %s", exc)
                return pd.DataFrame()
        return pd.DataFrame()
