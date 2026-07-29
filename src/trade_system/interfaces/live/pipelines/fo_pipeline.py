"""
FoPipeline — Lightweight analytics handler for F&O equity symbols.

Triggered
---------
``on_bar(symbol, bar, minute_data)``  — called by BarAggregator on every
completed 1-min bar for an F&O stock symbol (non-index).

Responsibilities
----------------
1. Resample 1-min data → strategy timeframe (e.g. 3 min).
2. Compute RSI (14) → detect overbought / oversold extremes.
3. Compute 20-bar volume MA → detect volume surge.
4. Track daily % change for sector scope analytics.
5. Persist 3-min bars to DB + CSV (via BarStore).
6. Write live snapshot to LiveStateWriter for dashboard.

What it intentionally does NOT do
-----------------------------------
- Send premarket Telegram reports (index-only, done at day init).
- Run SuperTrend flip alerts (too noisy for 180 stocks).
- Run ICT, Gamma Blast, Confirmed Strategy (index-only).
- Send any Telegram messages unless `enable_fo_telegram_alerts` is True.

Telegram alerts
---------------
The pipeline stores alert candidates.  The orchestrator checks
`get_pending_alerts()` and dispatches them to the appropriate notifier.
This keeps Telegram concerns out of the pipeline itself.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trade_system.domains.market_data.application.bar_store import BarStore
from trade_system.interfaces.live.helpers import (
    _completed_timeframe_bars,
    _latest_session_minute,
    _sanitize_intraday_minutes,
    resample_to_timeframe,
)
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# RSI thresholds
_RSI_OVERBOUGHT = 75.0
_RSI_OVERSOLD = 30.0
# Volume surge threshold (multiplier over 20-bar SMA)
_VOLUME_SURGE_MULTIPLIER = 2.0
# Debounce: don't re-alert same symbol within N seconds
_ALERT_DEBOUNCE_SECS = 3600


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Smoothing RSI (matches TradingView)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's RMA
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


class FoBarAlert:
    """A pending Telegram alert produced by FoPipeline."""

    __slots__ = ("symbol", "alert_type", "message", "created_at")

    def __init__(self, symbol: str, alert_type: str, message: str) -> None:
        self.symbol = symbol
        self.alert_type = alert_type
        self.message = message
        self.created_at = datetime.now(IST)


class FoPipeline:
    """
    Lightweight analytics pipeline for F&O equity symbols.

    Parameters
    ----------
    settings : Settings
    bar_store : BarStore
    strategy_timeframe_minutes : int
    rsi_period : int
        Default 14.
    volume_sma_period : int
        Period for volume surge baseline (default 20 bars on the strategy TF).
    """

    def __init__(
        self,
        settings: Settings,
        bar_store: BarStore,
        strategy_timeframe_minutes: int = 5,
        rsi_period: int = 14,
        volume_sma_period: int = 20,
    ) -> None:
        self.settings = settings
        self.bar_store = bar_store
        self.strategy_tf = strategy_timeframe_minutes
        self.rsi_period = rsi_period
        self.volume_sma_period = volume_sma_period

        # Per-symbol state
        self._strategy_data: dict[str, pd.DataFrame] = {}
        self._open_price: dict[str, float | None] = {}
        self._last_alert_time: dict[str, dict[str, datetime]] = {}
        self._pending_alerts: list[FoBarAlert] = []
        self._last_nmin_bar_time: dict[str, pd.Timestamp] = {}

        # Live snapshot for sector scope (symbol -> latest metrics)
        self._live_snapshot: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_symbol(self, symbol: str, prev_close: float | None = None) -> None:
        """Ensure state dicts are allocated for a new symbol."""
        if symbol not in self._strategy_data:
            self._strategy_data[symbol] = pd.DataFrame()
            self._open_price[symbol] = None
            self._last_alert_time[symbol] = {}
            self._live_snapshot[symbol] = {}

    def reset(self, symbols: list[str]) -> None:
        """Reset intraday state (called at day boundary)."""
        for sym in symbols:
            self._strategy_data[sym] = pd.DataFrame()
            self._open_price[sym] = None
            self._last_alert_time[sym] = {}
            self._live_snapshot[sym] = {}
            self._last_nmin_bar_time.pop(sym, None)
        self._pending_alerts.clear()

    def get_strategy_data(self, symbol: str) -> pd.DataFrame:
        return self._strategy_data.get(symbol, pd.DataFrame())

    def get_live_snapshot(self) -> dict[str, dict]:
        """Return the latest metrics for all tracked F&O symbols (for dashboards)."""
        return dict(self._live_snapshot)

    def get_pending_alerts(self) -> list[FoBarAlert]:
        """Pop and return all accumulated Telegram-ready alert objects."""
        alerts = list(self._pending_alerts)
        self._pending_alerts.clear()
        return alerts

    def on_bar(
        self,
        symbol: str,
        bar: pd.Series,
        minute_data: dict[str, pd.DataFrame],
    ) -> None:
        """
        Process a completed 1-min bar for an F&O equity symbol.

        Called by the orchestrator after BarAggregator emits a new bar.
        """
        self.register_symbol(symbol)
        current_date = self._today_ist()

        # Track opening price (first tick of the day)
        if self._open_price.get(symbol) is None:
            self._open_price[symbol] = float(bar.get("open", 0.0))

        # 1. Persist 1-min bar
        self.bar_store.save_1min_bar(symbol, bar, current_date)

        # 2. Resample + update strategy file
        updated_strategy = self.bar_store.update_strategy_file(
            symbol,
            minute_data.get(symbol, pd.DataFrame()),
            current_date,
            resample_fn=resample_to_timeframe,
            completed_bars_fn=_completed_timeframe_bars,
            sanitize_fn=_sanitize_intraday_minutes,
            latest_session_minute_fn=_latest_session_minute,
            timeframe_minutes=self.strategy_tf,
        )
        if not updated_strategy.empty:
            self._strategy_data[symbol] = updated_strategy
            last_bar = updated_strategy.iloc[-1]
            last_time = last_bar.name
            
            # 2b. Persist the N-min bar to DB if it just completed
            if self._last_nmin_bar_time.get(symbol) != last_time:
                self.bar_store.save_nmin_bar(symbol, last_bar, str(self.strategy_tf))
                self._last_nmin_bar_time[symbol] = last_time

        # 3. Compute analytics
        self._run_analytics(symbol, bar, current_date)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def _run_analytics(self, symbol: str, latest_bar: pd.Series, current_date: date) -> None:
        df = self._strategy_data.get(symbol, pd.DataFrame())
        if df.empty or len(df) < max(self.rsi_period + 1, self.volume_sma_period):
            return

        close = df["close"]
        volume = df["volume"]

        # --- RSI ---
        rsi_series = _compute_rsi(close, self.rsi_period)
        current_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else float("nan")

        # --- Volume surge ---
        vol_sma = float(volume.rolling(self.volume_sma_period).mean().iloc[-1] or 0.0)
        current_vol = float(latest_bar.get("volume", 0.0) or 0.0)
        surge_ratio = (current_vol / vol_sma) if vol_sma > 0 else 0.0

        # --- Daily % change ---
        open_price = self._open_price.get(symbol) or float(df["open"].iloc[0])
        current_price = float(latest_bar.get("close", 0.0))
        daily_pct = ((current_price - open_price) / open_price * 100.0) if open_price > 0 else 0.0

        short_sym = symbol.split(":")[-1].replace("-EQ", "").replace("-INDEX", "")

        # --- Update live snapshot (always) ---
        self._live_snapshot[symbol] = {
            "symbol": short_sym,
            "ltp": current_price,
            "open": open_price,
            "daily_pct": round(daily_pct, 2),
            "rsi": round(current_rsi, 1) if not pd.isna(current_rsi) else None,
            "volume": current_vol,
            "vol_surge": round(surge_ratio, 2),
            "updated_at": datetime.now(IST).isoformat(),
        }

        # --- Alerts (only if enabled) ---
        fo_alerts_on = getattr(self.settings, "enable_fo_telegram_alerts", False)
        if not fo_alerts_on:
            return

        now = datetime.now(IST)
        bar_time_str = str(df.index[-1].strftime("%H:%M")) if not df.empty else ""

        # RSI extreme
        if not pd.isna(current_rsi):
            if current_rsi >= _RSI_OVERBOUGHT:
                self._maybe_add_alert(
                    symbol, "RSI_OB", now,
                    f"🔥 <b>{short_sym} {bar_time_str}</b>\n"
                    f"RSI Overbought: <b>{current_rsi:.1f}</b>\n"
                    f"Price: ₹{current_price:.2f}\n"
                    f"Consider: SHORT / BUY PUT",
                )
            elif current_rsi <= _RSI_OVERSOLD:
                self._maybe_add_alert(
                    symbol, "RSI_OS", now,
                    f"💎 <b>{short_sym} {bar_time_str}</b>\n"
                    f"RSI Oversold: <b>{current_rsi:.1f}</b>\n"
                    f"Price: ₹{current_price:.2f}\n"
                    f"Consider: LONG / BUY CALL",
                )

        # Volume surge
        if surge_ratio >= _VOLUME_SURGE_MULTIPLIER:
            self._maybe_add_alert(
                symbol, "VOL_SURGE", now,
                f"📊 <b>{short_sym} {bar_time_str} — Volume Surge</b>\n"
                f"Volume: {current_vol:,.0f} ({surge_ratio:.1f}x 20-bar avg)\n"
                f"RSI: {current_rsi:.1f}\n"
                f"Daily: {daily_pct:+.2f}%",
            )

    def _maybe_add_alert(
        self,
        symbol: str,
        alert_type: str,
        now: datetime,
        message: str,
    ) -> None:
        """Add an alert if the debounce window has passed."""
        last = self._last_alert_time[symbol].get(alert_type)
        if last and (now - last).total_seconds() < _ALERT_DEBOUNCE_SECS:
            return
        self._last_alert_time[symbol][alert_type] = now
        self._pending_alerts.append(FoBarAlert(symbol, alert_type, message))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _today_ist() -> date:
        return datetime.now(IST).date()
