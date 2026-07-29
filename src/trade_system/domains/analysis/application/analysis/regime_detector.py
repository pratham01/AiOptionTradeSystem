"""
RegimeDetector — Classifies the current trading session as TRENDING, NEUTRAL, or CHOPPY.

Used to dynamically adjust the minimum EdgeScore threshold for option buying alerts:
  - TRENDING:  EdgeScore ≥ 60 (looser — market has momentum)
  - NEUTRAL:   EdgeScore ≥ 70 (standard)
  - CHOPPY:    EdgeScore ≥ 85 (strict — avoid whipsaws)

Signals analysed:
  1. Opening gap % (large gap → trending bias)
  2. VIX level (extreme levels → caution)
  3. First 30-min range vs Average Daily Range (room-to-run check)
  4. Nifty candle body-to-wick ratio (directional conviction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time as dt_time
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    NEUTRAL = "NEUTRAL"
    CHOPPY = "CHOPPY"


# Minimum EdgeScore thresholds keyed by regime.
# These are intentionally lower than a standalone edge-score gate because
# the pipeline's universe shortlist already pre-qualifies stocks through
# supertrend alignment, compression, volume surge, and sector leadership.
# The edge score here acts as a confirmation layer, not the primary filter.
REGIME_THRESHOLDS = {
    MarketRegime.TRENDING: 55,
    MarketRegime.NEUTRAL: 60,
    MarketRegime.CHOPPY: 75,
}


@dataclass
class RegimeResult:
    """Composite regime classification with supporting evidence."""
    regime: MarketRegime
    score_threshold: int
    gap_pct: float
    vix: float
    first_30m_range_pct: float   # as % of ADR
    body_wick_ratio: float       # avg body / avg range of first-hour candles
    signals: list[str]

    @property
    def label(self) -> str:
        icons = {
            MarketRegime.TRENDING: "🚀",
            MarketRegime.NEUTRAL: "⚖️",
            MarketRegime.CHOPPY: "🌀",
        }
        return f"{icons.get(self.regime, '')} {self.regime.value}"


class RegimeDetector:
    """
    Detects intraday market regime from Nifty 15-minute candles and VIX.
    """

    def __init__(self, nifty_symbol: str = "NSE:NIFTY50-INDEX") -> None:
        self.engine = get_engine()
        self.nifty_symbol = nifty_symbol

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, target_date: Optional[date] = None) -> RegimeResult:
        """
        Classify the current market regime based on Nifty behaviour.
        """
        df_15m = self._fetch_nifty_15m(target_date)
        df_daily = self._fetch_nifty_daily(target_date)

        if df_15m.empty or df_daily.empty:
            LOGGER.warning("Insufficient Nifty data for regime detection, defaulting to NEUTRAL.")
            return RegimeResult(
                regime=MarketRegime.NEUTRAL,
                score_threshold=REGIME_THRESHOLDS[MarketRegime.NEUTRAL],
                gap_pct=0.0, vix=0.0, first_30m_range_pct=0.0,
                body_wick_ratio=0.0, signals=["Insufficient data"],
            )

        if target_date is None:
            target_date = df_15m["timestamp"].max().date()

        today_candles = df_15m[df_15m["timestamp"].dt.date == target_date].sort_values("timestamp")
        if today_candles.empty:
            LOGGER.warning(f"No Nifty candles found for {target_date}.")
            return RegimeResult(
                regime=MarketRegime.NEUTRAL,
                score_threshold=REGIME_THRESHOLDS[MarketRegime.NEUTRAL],
                gap_pct=0.0, vix=0.0, first_30m_range_pct=0.0,
                body_wick_ratio=0.0, signals=["No intraday data"],
            )

        # Resolve previous close
        prev_candles = df_15m[df_15m["timestamp"].dt.date < target_date].sort_values("timestamp")
        prev_close = float(prev_candles.iloc[-1]["close"]) if not prev_candles.empty else float(today_candles.iloc[0]["open"])

        signals: list[str] = []
        regime_votes = {MarketRegime.TRENDING: 0, MarketRegime.NEUTRAL: 0, MarketRegime.CHOPPY: 0}

        # ── Signal 1: Opening Gap % ────────────────────────────────────────
        today_open = float(today_candles.iloc[0]["open"])
        gap_pct = ((today_open - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

        if abs(gap_pct) >= 0.5:
            regime_votes[MarketRegime.TRENDING] += 1
            signals.append(f"Gap {gap_pct:+.2f}% → Trending bias")
        elif abs(gap_pct) < 0.2:
            regime_votes[MarketRegime.CHOPPY] += 1
            signals.append(f"Flat open ({gap_pct:+.2f}%) → Choppy risk")
        else:
            regime_votes[MarketRegime.NEUTRAL] += 1
            signals.append(f"Moderate gap ({gap_pct:+.2f}%)")

        # ── Signal 2: VIX Level ────────────────────────────────────────────
        vix = self._get_vix(df_daily, target_date)
        if 12 <= vix <= 20:
            regime_votes[MarketRegime.TRENDING] += 1
            signals.append(f"VIX {vix:.1f} (optimal for directional)")
        elif vix > 25:
            regime_votes[MarketRegime.CHOPPY] += 1
            signals.append(f"VIX {vix:.1f} (high — premiums expensive)")
        elif vix < 10:
            regime_votes[MarketRegime.CHOPPY] += 1
            signals.append(f"VIX {vix:.1f} (very low — limited move expected)")
        else:
            regime_votes[MarketRegime.NEUTRAL] += 1
            signals.append(f"VIX {vix:.1f} (moderate)")

        # ── Signal 3: First 30-min Range vs ADR ───────────────────────────
        first_30m = today_candles.head(2)  # 2 × 15-min = 30 minutes
        first_30m_high = float(first_30m["high"].max())
        first_30m_low = float(first_30m["low"].min())
        first_30m_range = first_30m_high - first_30m_low

        # Average Daily Range (ADR) from last 20 sessions
        adr = self._compute_adr(df_daily, target_date, window=20)
        first_30m_range_pct = (first_30m_range / adr * 100) if adr > 0 else 50.0

        if first_30m_range_pct < 40:
            regime_votes[MarketRegime.TRENDING] += 1
            signals.append(f"30m range {first_30m_range_pct:.0f}% of ADR (room to run)")
        elif first_30m_range_pct > 70:
            regime_votes[MarketRegime.CHOPPY] += 1
            signals.append(f"30m range {first_30m_range_pct:.0f}% of ADR (extended)")
        else:
            regime_votes[MarketRegime.NEUTRAL] += 1
            signals.append(f"30m range {first_30m_range_pct:.0f}% of ADR (moderate)")

        # ── Signal 4: Candle Body-to-Wick Ratio ───────────────────────────
        # Use up to the first 4 candles (1 hour) to assess candle quality
        first_hour = today_candles.head(4)
        body_wick_ratio = self._compute_body_wick_ratio(first_hour)

        if body_wick_ratio >= 0.65:
            regime_votes[MarketRegime.TRENDING] += 1
            signals.append(f"Body/range {body_wick_ratio:.0%} (strong conviction candles)")
        elif body_wick_ratio < 0.35:
            regime_votes[MarketRegime.CHOPPY] += 1
            signals.append(f"Body/range {body_wick_ratio:.0%} (doji/spinning top candles)")
        else:
            regime_votes[MarketRegime.NEUTRAL] += 1
            signals.append(f"Body/range {body_wick_ratio:.0%} (mixed candles)")

        # ── Aggregate Votes ────────────────────────────────────────────────
        if regime_votes[MarketRegime.TRENDING] >= 3:
            regime = MarketRegime.TRENDING
        elif regime_votes[MarketRegime.CHOPPY] >= 3:
            regime = MarketRegime.CHOPPY
        else:
            regime = MarketRegime.NEUTRAL

        return RegimeResult(
            regime=regime,
            score_threshold=REGIME_THRESHOLDS[regime],
            gap_pct=round(gap_pct, 2),
            vix=round(vix, 1),
            first_30m_range_pct=round(first_30m_range_pct, 1),
            body_wick_ratio=round(body_wick_ratio, 2),
            signals=signals,
        )

    # ── Private: Data Fetching ─────────────────────────────────────────────────

    def _fetch_nifty_15m(self, target_date: Optional[date]) -> pd.DataFrame:
        """Fetch Nifty 15m candles with a 10-day lookback."""
        if target_date:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM ohlcv_15m
                WHERE symbol = :sym
                  AND timestamp >= date(:td, '-10 days')
                  AND timestamp <= date(:td, '+1 day')
                ORDER BY timestamp ASC
            """)
            params = {"sym": self.nifty_symbol, "td": target_date.isoformat()}
        else:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM ohlcv_15m
                WHERE symbol = :sym
                  AND timestamp >= date('now', '-10 days')
                ORDER BY timestamp ASC
            """)
            params = {"sym": self.nifty_symbol}

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch Nifty 15m data: {e}")
            return pd.DataFrame()

    def _fetch_nifty_daily(self, target_date: Optional[date]) -> pd.DataFrame:
        """Fetch Nifty daily candles (for ADR and VIX)."""
        nifty_daily_sym = "NSE:NIFTY50-INDEX"
        if target_date:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume, vix
                FROM ohlcv_daily
                WHERE symbol = :sym
                  AND timestamp >= date(:td, '-60 days')
                  AND timestamp <= :td
                ORDER BY timestamp ASC
            """)
            params = {"sym": nifty_daily_sym, "td": target_date.isoformat()}
        else:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume, vix
                FROM ohlcv_daily
                WHERE symbol = :sym
                  AND timestamp >= date('now', '-60 days')
                ORDER BY timestamp ASC
            """)
            params = {"sym": nifty_daily_sym}

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch Nifty daily data: {e}")
            return pd.DataFrame()

    # ── Private: Signal Helpers ────────────────────────────────────────────────

    def _get_vix(self, df_daily: pd.DataFrame, target_date: date) -> float:
        """Get VIX value from daily data (stored in the vix column)."""
        if df_daily.empty or "vix" not in df_daily.columns:
            return 15.0  # default moderate

        recent = df_daily[df_daily["timestamp"].dt.date <= target_date]
        if recent.empty:
            return 15.0

        # Use the latest non-null VIX value
        vix_vals = recent["vix"].dropna()
        if vix_vals.empty:
            return 15.0
        return float(vix_vals.iloc[-1])

    def _compute_adr(self, df_daily: pd.DataFrame, target_date: date, window: int = 20) -> float:
        """Compute Average Daily Range over the last N sessions."""
        recent = df_daily[df_daily["timestamp"].dt.date <= target_date].tail(window)
        if recent.empty:
            return 0.0
        daily_ranges = recent["high"] - recent["low"]
        return float(daily_ranges.mean())

    def _compute_body_wick_ratio(self, candles: pd.DataFrame) -> float:
        """
        Compute average body-to-range ratio for given candles.
        High ratio → directional conviction; Low ratio → indecisive (dojis).
        """
        if candles.empty:
            return 0.5

        bodies = (candles["close"] - candles["open"]).abs()
        ranges = candles["high"] - candles["low"]
        # Avoid division by zero
        valid = ranges > 0
        if not valid.any():
            return 0.5

        ratios = bodies[valid] / ranges[valid]
        return float(ratios.mean())
