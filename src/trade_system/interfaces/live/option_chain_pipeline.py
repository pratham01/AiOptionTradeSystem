"""
OptionChainPipeline — Focused module for option chain data fetching, snapshot saving, and ATM tracking.

Extracted from LiveMarketDataService (collector.py).
Single Responsibility: Periodically fetch the OC, persist it, and compute strike-level indicators.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trade_system.domains.strategy.application.indicators import calculate_supertrend
from trade_system.domains.market_data.domain.schemas.market_data import validate_option_chain

LOGGER = logging.getLogger("OptionChainPipeline")


class OptionChainPipeline:
    """
    Manages periodic option chain collection and storage.

    Responsibilities:
    - Poll the broker for option chain data every N seconds
    - Validate incoming data via OptionStrikeSchema
    - Append to today's CSV snapshot file
    - Compute strike-level Supertrend for the STFlip Telegram block
    - Track ATM strike over time

    Usage:
        pipeline = OptionChainPipeline(broker, settings, dispatcher)
        pipeline.start()   # starts background thread
        ...
        pipeline.stop()    # graceful shutdown
    """

    def __init__(
        self,
        broker,
        settings,
        dispatcher,
        supertrend_period: int = 7,
        supertrend_multiplier: int = 3,
        poll_interval_seconds: int = 180,
    ) -> None:
        self.broker = broker
        self.settings = settings
        self.dispatcher = dispatcher
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.poll_interval = poll_interval_seconds

        self._thread: threading.Thread | None = None
        self._running = False

        # State
        self.latest_oc: dict[str, pd.DataFrame | None] = {}
        self.latest_spot: dict[str, float | None] = {}
        self.latest_atm: dict[str, int | None] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            LOGGER.warning("OptionChainPipeline is already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="oc-pipeline")
        self._thread.start()
        LOGGER.info("✅ OptionChainPipeline started (interval=%ds).", self.poll_interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        LOGGER.info("OptionChainPipeline stopped.")

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN POLL LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            now = datetime.now()
            for symbol in self.settings.index_symbols:
                try:
                    self._fetch_and_process(symbol, now)
                except Exception as exc:
                    LOGGER.exception("OptionChainPipeline error for %s: %s", symbol, exc)
            time.sleep(self.poll_interval)

    def _fetch_and_process(self, symbol: str, now: datetime) -> None:
        """Fetch, validate, and persist the option chain for a single symbol."""
        clean_sym = symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "")

        try:
            oc_df, spot_price = self._fetch_option_chain(clean_sym)
        except Exception as exc:
            LOGGER.error("[%s] OC fetch failed: %s", clean_sym, exc)
            return

        if oc_df is None or oc_df.empty:
            LOGGER.warning("[%s] Empty OC data returned. Skipping.", clean_sym)
            return

        # — Validate schema —
        raw_rows = oc_df.to_dict("records")
        # Add required fields if missing
        for row in raw_rows:
            row.setdefault("timestamp", now)
            row.setdefault("spot_price", spot_price or 0)
            row.setdefault("symbol", row.get("symbol", ""))

        valid_rows, errors = validate_option_chain(raw_rows)
        if errors:
            LOGGER.warning("[%s] OC validation: %d rows discarded. First error: %s",
                           clean_sym, len(errors), errors[0])

        if not valid_rows:
            LOGGER.error("[%s] ALL option chain rows failed validation. Aborting to prevent GIGO.", clean_sym)
            return

        # Store in memory
        self.latest_oc[clean_sym] = oc_df
        self.latest_spot[clean_sym] = spot_price

        # Compute ATM
        if spot_price and spot_price > 0:
            strike_step = self._infer_strike_step(oc_df)
            self.latest_atm[clean_sym] = round(spot_price / strike_step) * strike_step

        # Persist to CSV
        self._save_snapshot(clean_sym, now, oc_df, spot_price)

    def _fetch_option_chain(self, symbol: str):
        """Delegate to the broker to fetch the option chain."""
        return self.broker.get_option_chain(symbol)

    def _infer_strike_step(self, oc_df: pd.DataFrame) -> int:
        """Infer strike interval from the data (e.g., 50 for NIFTY, 100 for BANKNIFTY)."""
        if "strike" not in oc_df.columns:
            return 50
        strikes = sorted(oc_df["strike"].dropna().unique())
        if len(strikes) < 2:
            return 50
        diffs = [abs(strikes[i+1] - strikes[i]) for i in range(min(5, len(strikes)-1))]
        return int(min(diffs)) or 50

    def _save_snapshot(self, symbol: str, now: datetime, oc_df: pd.DataFrame, spot_price: float | None) -> None:
        """Append option chain snapshot to today's CSV."""
        try:
            date_str = now.strftime("%Y%m%d")
            path = self.settings.option_chain_data_dir / f"{symbol}_strikes_{date_str}.csv"

            snap = oc_df.copy()
            if "oi" in snap.columns and "open_interest" not in snap.columns:
                snap.rename(columns={"oi": "open_interest"}, inplace=True)
            snap.insert(0, "timestamp", now.strftime("%Y-%m-%d %H:%M:%S"))
            snap.insert(1, "spot_price", spot_price if spot_price else 0)

            if not path.exists():
                snap.to_csv(path, index=False)
            else:
                snap.to_csv(path, mode="a", header=False, index=False)

            LOGGER.debug("[%s] OC snapshot saved → %s", symbol, path.name)
        except Exception as exc:
            LOGGER.error("[%s] Failed to save OC snapshot: %s", symbol, exc)

    # ─────────────────────────────────────────────────────────────────────────
    # STRIKE SUPERTREND (for the STFlip alert block)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_strike_supertrend(self, history_df: pd.DataFrame, strike: int, option_type: str) -> dict | None:
        """
        Compute the Supertrend for a single option strike's premium history.

        Safety: validates timestamp continuity before computing. Returns None
        if data gaps are detected (protects against GIGO corrupted ST signals).
        """
        subset = history_df[
            (history_df["option_type"].str.upper() == option_type.upper()) &
            (history_df["strike"].round().astype(int) == strike)
        ].copy()

        if subset.empty:
            return None

        subset = subset.sort_values("timestamp").reset_index(drop=True)
        prices = subset["ltp"].astype(float).values

        if len(prices) < self.supertrend_period + 2:
            return None

        opens = prices[:-1]
        closes = prices[1:]
        highs = np.maximum(opens, closes) * 1.001
        lows = np.minimum(opens, closes) * 0.999

        # ← Include actual timestamps so the gap-detector in supertrend.py can check
        timestamps = pd.to_datetime(subset["timestamp"].iloc[1:]).values
        ohlc = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "timestamp": timestamps,
        }, index=timestamps)

        st_df = calculate_supertrend(ohlc, period=self.supertrend_period, multiplier=self.supertrend_multiplier)

        if st_df is None or st_df.empty:
            LOGGER.error(
                "Strike %s%s: Supertrend aborted due to data gap. "
                "Suppressing strike info in the STFlip alert to prevent false signals.",
                strike, option_type
            )
            return None

        last = st_df.iloc[-1]
        return {
            "strike": strike,
            "option_type": option_type,
            "direction": int(last["supertrend_direction"]),
            "st_level": float(last["supertrend"]),
            "ltp": float(last["close"]),
        }
