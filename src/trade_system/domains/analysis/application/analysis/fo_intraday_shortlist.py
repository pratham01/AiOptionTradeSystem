"""
FOIntradayShortlist — Pre-market / rolling universe filter for option buying.

Narrows the 200+ F&O universe down to ~20 "ready" stocks using daily data:
  1. Daily Supertrend alignment (stock trend agrees with sector trend)
  2. Compression release candidates (NR7 / Inside Day in last 3 sessions)
  3. Volume breakout candidates (prev day vol ≥ 1.5x 10-day avg)
  4. Sector leaders + laggards (top 3 / bottom 3 sectors)
  5. 52-week high/low proximity (within 3% of 52W high or low)

All data comes from the existing ohlcv_daily SQLite table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe, get_sector_mapping
from trade_system.domains.strategy.application.indicators.supertrend import SupertrendIndicator
from trade_system.domains.strategy.application.indicators.compression import CompressionIndicator

LOGGER = logging.getLogger(__name__)


@dataclass
class ShortlistCandidate:
    """A stock that passes the universe filter."""
    symbol: str
    sector: str
    ltp: float
    prev_close: float
    change_pct: float
    reasons: List[str] = field(default_factory=list)
    daily_supertrend_dir: int = 0       # 1=bullish, -1=bearish
    is_compressed: bool = False
    volume_surge: float = 0.0
    is_near_52w_high: bool = False
    is_near_52w_low: bool = False
    is_near_ath: bool = False
    is_near_swing_high: bool = False
    is_near_swing_low: bool = False
    ath: float = 0.0
    swing_high: float = 0.0
    swing_low: float = 0.0
    sector_rank: int = 0                # 1=top sector, higher=weaker
    sector_type: str = "MID"            # "LEADER", "LAGGARD", "MID"

    @property
    def bias(self) -> str:
        """Suggested directional bias for option buying."""
        if self.daily_supertrend_dir == 1 and self.sector_type == "LEADER":
            return "STRONG_CALL"
        elif self.daily_supertrend_dir == -1 and self.sector_type == "LAGGARD":
            return "STRONG_PUT"
        elif self.daily_supertrend_dir == 1:
            return "CALL"
        elif self.daily_supertrend_dir == -1:
            return "PUT"
        return "NEUTRAL"


class FOIntradayShortlist:
    """
    Filters the F&O universe to identify the best candidates for intraday
    option buying on a given trading day.
    """

    def __init__(
        self,
        top_sector_count: int = 3,
        volume_surge_threshold: float = 1.5,
        proximity_52w_pct: float = 3.0,
        max_candidates: int = 25,
    ) -> None:
        self.engine = get_engine()
        self.sector_map = get_sector_mapping()
        self.fo_symbols = get_fo_universe()
        self._supertrend = SupertrendIndicator(period=7, multiplier=3)
        self._compression = CompressionIndicator(atr_period=14, lookback=4)

        self.top_sector_count = top_sector_count
        self.volume_surge_threshold = volume_surge_threshold
        self.proximity_52w_pct = proximity_52w_pct
        self.max_candidates = max_candidates

    # ── Public API ─────────────────────────────────────────────────────────────

    def shortlist(self, target_date: Optional[date] = None) -> List[ShortlistCandidate]:
        """
        Run the full shortlisting pipeline.

        Returns a list of ShortlistCandidate objects sorted by number of
        qualifying reasons (most reasons first).
        """
        df_daily = self._fetch_daily_data(target_date, lookback_days=270)
        if df_daily.empty:
            LOGGER.warning("No daily data available for shortlisting.")
            return []

        if target_date is None:
            target_date = df_daily["timestamp"].max().date()

        LOGGER.info(f"Running FOIntradayShortlist for {target_date}")

        # Group by symbol once
        symbol_groups: Dict[str, pd.DataFrame] = {
            sym: grp.sort_values("timestamp")
            for sym, grp in df_daily.groupby("symbol")
            if sym in self.sector_map and self.sector_map[sym] != "UNKNOWN"
        }

        # ── Stage 1: Sector Performance & Ranking ──────────────────────────
        sector_perf, stock_changes = self._compute_sector_performance(symbol_groups, target_date)
        sector_ranks = self._rank_sectors(sector_perf)
        total_sectors = len(sector_ranks)
        leading_sectors = {s for s, r in sector_ranks.items() if r <= self.top_sector_count}
        lagging_sectors = {s for s, r in sector_ranks.items() if r > total_sectors - self.top_sector_count}

        # ── Stage 1.5: Fetch ATH values ────────────────────────────────────
        ath_values = self._fetch_ath_values(target_date)

        # ── Stage 2: Per-Stock Analysis ────────────────────────────────────
        candidates: Dict[str, ShortlistCandidate] = {}

        for symbol, grp in symbol_groups.items():
            sector = self.sector_map.get(symbol, "UNKNOWN")
            if sector == "UNKNOWN":
                continue

            grp_up_to = grp[grp["timestamp"].dt.date <= target_date]
            if len(grp_up_to) < 30:
                continue

            latest = grp_up_to.iloc[-1]
            prev = grp_up_to.iloc[-2] if len(grp_up_to) >= 2 else latest
            ltp = float(latest["close"])
            prev_close = float(prev["close"])
            change_pct = ((ltp - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

            candidate = ShortlistCandidate(
                symbol=symbol,
                sector=sector,
                ltp=ltp,
                prev_close=prev_close,
                change_pct=change_pct,
                sector_rank=sector_ranks.get(sector, total_sectors),
            )

            # Classify sector type
            if sector in leading_sectors:
                candidate.sector_type = "LEADER"
                candidate.reasons.append(f"Sector Leader (# {sector_ranks[sector]})")
            elif sector in lagging_sectors:
                candidate.sector_type = "LAGGARD"
                candidate.reasons.append(f"Sector Laggard (# {sector_ranks[sector]})")

            # ── Filter 1: Daily Supertrend ─────────────────────────────────
            st_dir = self._check_supertrend(grp_up_to)
            candidate.daily_supertrend_dir = st_dir
            if st_dir != 0:
                if (st_dir == 1 and sector in leading_sectors) or (st_dir == -1 and sector in lagging_sectors):
                    candidate.reasons.append(f"ST aligned with sector ({'Bull' if st_dir == 1 else 'Bear'})")

            # ── Filter 2: Compression (NR7 / Inside Day) ──────────────────
            is_compressed = self._check_compression(grp_up_to, target_date)
            candidate.is_compressed = is_compressed
            if is_compressed:
                candidate.reasons.append("Compression (coiled for breakout)")

            # ── Filter 3: Volume Surge ─────────────────────────────────────
            vol_surge = self._check_volume_surge(grp_up_to)
            candidate.volume_surge = vol_surge
            if vol_surge >= self.volume_surge_threshold:
                candidate.reasons.append(f"Vol surge {vol_surge:.1f}x (prev day)")

            # ── Filter 4: 52-Week Proximity ────────────────────────────────
            near_high, near_low = self._check_52w_proximity(grp_up_to, ltp)
            candidate.is_near_52w_high = near_high
            candidate.is_near_52w_low = near_low
            if near_high:
                candidate.reasons.append("Near 52W High (breakout candidate)")
            if near_low:
                candidate.reasons.append("Near 52W Low (breakdown candidate)")

            # ── Filter 5: ATH Proximity ────────────────────────────────────
            ath_val = ath_values.get(symbol, 0.0)
            candidate.ath = ath_val
            if ath_val > 0:
                near_ath = ltp >= ath_val * 0.98
                candidate.is_near_ath = near_ath
                if near_ath:
                    candidate.reasons.append("Consolidating near ATH (strong breakout candidate)")

            # ── Filter 6: Previous Swing Proximity ─────────────────────────
            near_sw_high, near_sw_low, sw_high, sw_low = self._check_swing_proximity(grp_up_to, ltp)
            candidate.is_near_swing_high = near_sw_high
            candidate.is_near_swing_low = near_sw_low
            candidate.swing_high = sw_high
            candidate.swing_low = sw_low
            if near_sw_high:
                candidate.reasons.append("Holding near previous swing high")
            if near_sw_low:
                candidate.reasons.append("Holding near previous swing low")

            # Add candidate if it has at least one qualifying reason
            if candidate.reasons:
                candidates[symbol] = candidate

        # ── Stage 3: Sort and Limit ────────────────────────────────────────
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: len(c.reasons),
            reverse=True,
        )

        result = sorted_candidates[: self.max_candidates]
        LOGGER.info(
            f"Shortlisted {len(result)}/{len(symbol_groups)} stocks "
            f"(Leaders: {len(leading_sectors)}, Laggards: {len(lagging_sectors)})"
        )
        return result

    # ── Private: Data Fetching ─────────────────────────────────────────────────

    def _fetch_daily_data(self, target_date: Optional[date], lookback_days: int) -> pd.DataFrame:
        """Fetch daily OHLCV candles from the database."""
        if target_date:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM ohlcv_daily
                WHERE timestamp >= date(:td, :lookback)
                  AND timestamp <= :td
                  AND symbol LIKE '%-EQ%'
                ORDER BY symbol, timestamp ASC
            """)
            params = {"td": target_date.isoformat(), "lookback": f"-{lookback_days} days"}
        else:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM ohlcv_daily
                WHERE timestamp >= date('now', :lookback)
                  AND symbol LIKE '%-EQ%'
                ORDER BY symbol, timestamp ASC
            """)
            params = {"lookback": f"-{lookback_days} days"}

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch daily data: {e}")
            return pd.DataFrame()

    # ── Private: Sector Performance ────────────────────────────────────────────

    def _compute_sector_performance(
        self,
        symbol_groups: Dict[str, pd.DataFrame],
        target_date: date,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        """Compute sector-level performance based on stock changes."""
        stock_changes: Dict[str, float] = {}
        sector_changes: Dict[str, List[float]] = {}

        for symbol, grp in symbol_groups.items():
            sector = self.sector_map.get(symbol, "UNKNOWN")
            if sector == "UNKNOWN":
                continue

            grp_up_to = grp[grp["timestamp"].dt.date <= target_date]
            if len(grp_up_to) < 2:
                continue

            latest_close = float(grp_up_to.iloc[-1]["close"])
            prev_close = float(grp_up_to.iloc[-2]["close"])
            if prev_close <= 0:
                continue

            change_pct = ((latest_close - prev_close) / prev_close) * 100
            stock_changes[symbol] = change_pct
            sector_changes.setdefault(sector, []).append(change_pct)

        rows = [
            {"sector": sector, "pChange": np.mean(changes)}
            for sector, changes in sector_changes.items()
        ]
        sector_perf = pd.DataFrame(rows).sort_values("pChange", ascending=False)
        return sector_perf, stock_changes

    def _rank_sectors(self, sector_perf: pd.DataFrame) -> Dict[str, int]:
        """Rank sectors 1..N (1 = top performer)."""
        sorted_sectors = sector_perf.sort_values("pChange", ascending=False)
        return {
            row["sector"]: rank + 1
            for rank, (_, row) in enumerate(sorted_sectors.iterrows())
        }

    # ── Private: Individual Filters ────────────────────────────────────────────

    def _check_supertrend(self, grp: pd.DataFrame) -> int:
        """Return daily supertrend direction: 1=bullish, -1=bearish, 0=unknown."""
        if len(grp) < 15:
            return 0
        try:
            st_df = self._supertrend.calculate(grp)
            if "supertrend_direction" in st_df.columns:
                return int(st_df["supertrend_direction"].iloc[-1])
        except Exception:
            pass
        return 0

    def _check_compression(self, grp: pd.DataFrame, target_date: date) -> bool:
        """Check if the stock was in compression in the last 3 sessions."""
        if len(grp) < 20:
            return False
        try:
            comp_df = self._compression.calculate(grp)
            # Check last 3 candles (sessions) before target_date
            recent = comp_df[comp_df["timestamp"].dt.date <= target_date].tail(3)
            if recent.empty:
                return False
            return bool(
                recent["is_compressed"].any()
                if "is_compressed" in recent.columns
                else False
            )
        except Exception:
            return False

    def _check_volume_surge(self, grp: pd.DataFrame) -> float:
        """Check if the most recent session's volume is elevated vs 10-day avg."""
        if len(grp) < 11:
            return 0.0
        try:
            latest_vol = float(grp.iloc[-1]["volume"])
            avg_vol_10d = float(grp.iloc[-11:-1]["volume"].mean())
            if avg_vol_10d <= 0:
                return 0.0
            return round(latest_vol / avg_vol_10d, 2)
        except Exception:
            return 0.0

    def _check_52w_proximity(self, grp: pd.DataFrame, ltp: float) -> tuple[bool, bool]:
        """Check if LTP is within proximity_52w_pct of 52-week high or low."""
        if len(grp) < 50:
            return False, False
        try:
            last_250 = grp.tail(250)
            high_52w = float(last_250["high"].max())
            low_52w = float(last_250["low"].min())

            near_high = ltp >= high_52w * (1 - self.proximity_52w_pct / 100)
            near_low = ltp <= low_52w * (1 + self.proximity_52w_pct / 100)
            return near_high, near_low
        except Exception:
            return False, False

    def _fetch_ath_values(self, target_date: date) -> Dict[str, float]:
        """Fetch absolute All-Time High for all F&O stocks up to target_date."""
        query = text("""
            SELECT symbol, MAX(high) as ath
            FROM ohlcv_daily
            WHERE timestamp <= :td
            GROUP BY symbol
        """)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"td": target_date.isoformat()}).fetchall()
            return {row[0]: float(row[1]) for row in result if row[0] and row[1]}
        except Exception as e:
            LOGGER.error(f"Failed to fetch ATH values: {e}")
            return {}

    def _check_swing_proximity(
        self, grp_up_to: pd.DataFrame, ltp: float
    ) -> tuple[bool, bool, float, float]:
        """
        Check if the stock is currently consolidating within 1.5% of its previous swing high or swing low.
        A swing high/low is defined as the rolling 20-day high/low pivot.
        """
        if len(grp_up_to) < 21:
            return False, False, 0.0, 0.0

        # Exclude the current day's candle to prevent lookahead bias
        hist = grp_up_to.iloc[:-1].tail(20)
        if hist.empty:
            return False, False, 0.0, 0.0

        swing_high = float(hist["high"].max())
        swing_low = float(hist["low"].min())

        near_swing_high = abs(ltp - swing_high) / swing_high <= 0.015
        near_swing_low = abs(ltp - swing_low) / swing_low <= 0.015

        return near_swing_high, near_swing_low, swing_high, swing_low
