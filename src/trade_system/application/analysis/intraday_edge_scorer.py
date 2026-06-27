"""
IntradayEdgeScorer — Multi-layer confluence scoring engine for intraday FO trading.

Combines 7 independent signal layers into a single EdgeScore (0–100) per stock:
  1. Sector Momentum   — Is the stock's sector leading today?
  2. Relative Strength  — Is the stock outperforming its sector peers?
  3. VWAP Location      — Is price at a favorable VWAP zone?
  4. Volume Confirmation — Volume surge + positive CVD (buying pressure)?
  5. Momentum Timing    — Is the move early (catchable) or exhausted?
  6. Supertrend Align   — Do 15-minute and daily supertrend agree?
  7. Compression Release — Was the stock coiled and now releasing energy?

Each layer produces a sub-score (0.0–1.0) and a direction signal (CALL / PUT / NEUTRAL).
The composite EdgeScore applies a direction-alignment multiplier when 5+ layers agree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.data.fo_universe import get_sector_mapping
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.application.indicators.vwap import VWAPIndicator
from trade_system.application.indicators.volume_delta import VolumeDeltaIndicator
from trade_system.application.indicators.compression import CompressionIndicator

LOGGER = logging.getLogger(__name__)


# ── Layer Weights ──────────────────────────────────────────────────────────────

LAYER_WEIGHTS = {
    "sector_momentum": 0.10,
    "relative_strength": 0.15,
    "vwap_location": 0.15,
    "volume_confirmation": 0.15,
    "momentum_timing": 0.15,
    "supertrend_alignment": 0.15,
    "compression_release": 0.15,
}

# Direction agreement thresholds
STRONG_ALIGNMENT_MIN = 5   # 5+ layers agreeing = strong alignment
CONFLICT_MAX = 3           # 3+ layers disagreeing = conflicted


@dataclass
class LayerResult:
    """Result from a single confluence layer."""
    name: str
    score: float       # 0.0 – 1.0
    direction: str     # "CALL", "PUT", or "NEUTRAL"
    detail: str = ""   # Human-readable detail


@dataclass
class EdgeScore:
    """Composite edge score for a single stock."""
    symbol: str
    sector: str
    ltp: float
    change_pct: float
    raw_score: float                 # 0–100, before multiplier
    final_score: float               # 0–100, after multiplier
    direction: str                   # "CALL" or "PUT"
    direction_confidence: float      # 0.0–1.0
    layers: Dict[str, LayerResult] = field(default_factory=dict)

    # Smart entry fields (populated by SmartEntryTrigger)
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    entry_status: str = "WAITING"    # "TRIGGERED", "APPROACHING", "WAITING"
    atr: float = 0.0

    @property
    def risk_reward(self) -> float:
        if self.entry_price > 0 and self.stop_loss > 0 and self.target_1 > 0:
            risk = abs(self.entry_price - self.stop_loss)
            if risk > 0:
                reward = abs(self.target_1 - self.entry_price)
                return round(reward / risk, 2)
        return 0.0


class IntradayEdgeScorer:
    """
    Scans the entire FO universe and produces an EdgeScore for each stock.

    Usage:
        scorer = IntradayEdgeScorer()
        results = scorer.scan(target_date=date.today())
        for edge in sorted(results, key=lambda e: e.final_score, reverse=True)[:10]:
            print(f"{edge.symbol}: {edge.final_score:.0f} {edge.direction}")
    """

    def __init__(self) -> None:
        self.engine = get_engine()
        self.sector_map = get_sector_mapping()
        self._supertrend = SupertrendIndicator(period=7, multiplier=3)
        self._vwap = VWAPIndicator()
        self._volume_delta = VolumeDeltaIndicator()
        self._compression = CompressionIndicator(atr_period=14, lookback=4)

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(
        self,
        target_date: date | None = None,
        sector_perf: pd.DataFrame | None = None,
        stock_perf: pd.DataFrame | None = None,
    ) -> List[EdgeScore]:
        """
        Run the full edge scan on the FO universe.

        Args:
            target_date: The trading day to analyze. Defaults to latest in DB.
            sector_perf: Pre-computed sector performance DataFrame
                         with columns ['sector', 'pChange'].
            stock_perf:  Pre-computed stock performance DataFrame
                         with columns ['symbol', 'sector', 'close_last',
                         'close_prev', 'pChange', 'vol_surge'].

        Returns:
            List of EdgeScore objects, one per stock.
        """
        # 1. Fetch 15-minute candle data (10-day lookback)
        df_15m = self._fetch_15m_data(target_date)
        if df_15m.empty:
            LOGGER.warning("No 15m data available for edge scoring.")
            return []

        # Resolve target date from data
        if target_date is None:
            non_idx = df_15m[~df_15m["symbol"].str.contains("INDEX")]
            if non_idx.empty:
                return []
            target_date = non_idx["timestamp"].max().date()

        # 2. Fetch daily candle data (60-day lookback for daily supertrend)
        df_daily = self._fetch_daily_data(target_date, lookback_days=60)

        # 3. Compute sector performance if not provided
        if sector_perf is None or stock_perf is None:
            sector_perf, stock_perf = self._compute_performance(df_15m, target_date)

        if stock_perf.empty:
            LOGGER.warning("No stock performance data for edge scoring.")
            return []

        # 4. Pre-compute sector rankings
        sector_ranks = self._rank_sectors(sector_perf)

        # 5. Pre-compute per-symbol indicators (vectorized where possible)
        symbol_15m_groups = {
            sym: grp.sort_values("timestamp")
            for sym, grp in df_15m.groupby("symbol")
            if "INDEX" not in sym
        }
        symbol_daily_groups = {}
        if not df_daily.empty:
            symbol_daily_groups = {
                sym: grp.sort_values("timestamp")
                for sym, grp in df_daily.groupby("symbol")
            }

        # 6. Score each stock
        results: List[EdgeScore] = []
        for _, row in stock_perf.iterrows():
            symbol = row["symbol"]
            sector = row.get("sector", self.sector_map.get(symbol, "UNKNOWN"))
            if sector == "UNKNOWN":
                continue

            ltp = float(row.get("close_last", 0))
            change_pct = float(row.get("pChange", 0))
            vol_surge = float(row.get("vol_surge", 0))

            df_sym_15m = symbol_15m_groups.get(symbol, pd.DataFrame())
            df_sym_daily = symbol_daily_groups.get(symbol, pd.DataFrame())

            edge = self._score_stock(
                symbol=symbol,
                sector=sector,
                ltp=ltp,
                change_pct=change_pct,
                vol_surge=vol_surge,
                sector_ranks=sector_ranks,
                sector_perf=sector_perf,
                stock_perf=stock_perf,
                df_15m=df_sym_15m,
                df_daily=df_sym_daily,
                target_date=target_date,
            )
            results.append(edge)

        # Sort by final_score descending
        results.sort(key=lambda e: e.final_score, reverse=True)
        return results

    # ── Private: Data Fetching ─────────────────────────────────────────────────

    def _fetch_15m_data(self, target_date: date | None) -> pd.DataFrame:
        """Fetch 15-minute candles for the FO universe with 10-day lookback."""
        if target_date:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM ohlcv_15m
                WHERE timestamp >= date(:td, '-10 days') AND timestamp <= date(:td, '+1 day')
                ORDER BY symbol, timestamp ASC
            """)
            params = {"td": target_date.isoformat()}
        else:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM ohlcv_15m
                WHERE timestamp >= date('now', '-10 days')
                ORDER BY symbol, timestamp ASC
            """)
            params = {}

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch 15m data: {e}")
            return pd.DataFrame()

    def _fetch_daily_data(self, target_date: date, lookback_days: int = 60) -> pd.DataFrame:
        """Fetch daily candles for daily-timeframe supertrend."""
        query = text("""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE timestamp >= date(:td, :lookback) AND timestamp <= :td
            ORDER BY symbol, timestamp ASC
        """)
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={
                    "td": target_date.isoformat(),
                    "lookback": f"-{lookback_days} days",
                })
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch daily data: {e}")
            return pd.DataFrame()

    # ── Private: Performance Computation ───────────────────────────────────────

    def _compute_performance(
        self, df: pd.DataFrame, target_date: date
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute sector and stock performance from 15m candle data."""
        df["sector"] = df["symbol"].apply(lambda s: self.sector_map.get(s, "UNKNOWN"))
        non_idx = df[~df["symbol"].str.contains("INDEX")]

        today_df = non_idx[non_idx["timestamp"].dt.date == target_date]
        prev_df = non_idx[non_idx["timestamp"].dt.date < target_date]

        if today_df.empty or prev_df.empty:
            return pd.DataFrame(columns=["sector", "pChange"]), pd.DataFrame()

        latest = today_df.groupby("symbol").last().reset_index()
        prev = prev_df.groupby("symbol").last().reset_index()

        merged = pd.merge(latest, prev, on=["symbol", "sector"], suffixes=("_last", "_prev"))
        merged["pChange"] = (
            (merged["close_last"] - merged["close_prev"]) / merged["close_prev"]
        ) * 100

        # Volume surge computation
        prev_all = prev_df.copy()
        prev_all["trade_date"] = prev_all["timestamp"].dt.date
        daily_vols = prev_all.groupby(["symbol", "trade_date"])["volume"].sum().reset_index()
        avg_5d = daily_vols.groupby("symbol")["volume"].apply(
            lambda x: x.tail(5).mean()
        ).reset_index(name="avg_vol_5d")

        today_vol = today_df.groupby("symbol")["volume"].sum().reset_index(name="vol_today")
        vol_merged = pd.merge(today_vol, avg_5d, on="symbol", how="left")
        vol_merged["vol_surge"] = np.where(
            vol_merged["avg_vol_5d"] > 0,
            vol_merged["vol_today"] / vol_merged["avg_vol_5d"],
            0,
        )

        merged = pd.merge(
            merged, vol_merged[["symbol", "vol_surge"]], on="symbol", how="left"
        )
        merged["vol_surge"] = merged["vol_surge"].fillna(0)

        sector_perf = (
            merged.groupby("sector")["pChange"]
            .mean()
            .reset_index()
            .query("sector != 'UNKNOWN'")
            .sort_values("pChange", ascending=False)
        )

        stock_perf = merged[["symbol", "sector", "close_last", "close_prev", "pChange", "vol_surge"]]
        return sector_perf, stock_perf

    def _rank_sectors(self, sector_perf: pd.DataFrame) -> Dict[str, int]:
        """Rank sectors 1..N (1 = top performer)."""
        sorted_sectors = sector_perf.sort_values("pChange", ascending=False)
        return {
            row["sector"]: rank + 1
            for rank, (_, row) in enumerate(sorted_sectors.iterrows())
        }

    # ── Private: Stock Scoring ─────────────────────────────────────────────────

    def _score_stock(
        self,
        symbol: str,
        sector: str,
        ltp: float,
        change_pct: float,
        vol_surge: float,
        sector_ranks: Dict[str, int],
        sector_perf: pd.DataFrame,
        stock_perf: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_daily: pd.DataFrame,
        target_date: date,
    ) -> EdgeScore:
        """Compute the 7-layer EdgeScore for a single stock."""
        layers: Dict[str, LayerResult] = {}

        # Layer 1: Sector Momentum
        layers["sector_momentum"] = self._layer_sector_momentum(
            sector, sector_ranks, sector_perf
        )

        # Layer 2: Relative Strength
        layers["relative_strength"] = self._layer_relative_strength(
            symbol, sector, change_pct, stock_perf
        )

        # Layer 3: VWAP Location
        layers["vwap_location"] = self._layer_vwap_location(df_15m, ltp, target_date)

        # Layer 4: Volume Confirmation
        layers["volume_confirmation"] = self._layer_volume_confirmation(
            df_15m, vol_surge, target_date
        )

        # Layer 5: Momentum Timing
        layers["momentum_timing"] = self._layer_momentum_timing(
            df_15m, ltp, change_pct, target_date
        )

        # Layer 6: Supertrend Alignment
        layers["supertrend_alignment"] = self._layer_supertrend_alignment(
            df_15m, df_daily
        )

        # Layer 7: Compression Release
        layers["compression_release"] = self._layer_compression_release(
            df_15m, change_pct, target_date
        )

        # ── Composite Score ────────────────────────────────────────────────────
        raw_score = 0.0
        direction_votes: Dict[str, int] = {"CALL": 0, "PUT": 0, "NEUTRAL": 0}

        for layer_name, result in layers.items():
            weight = LAYER_WEIGHTS.get(layer_name, 0.0)
            raw_score += result.score * weight * 100
            direction_votes[result.direction] += 1

        # Determine dominant direction
        call_votes = direction_votes["CALL"]
        put_votes = direction_votes["PUT"]

        if call_votes > put_votes:
            direction = "CALL"
            agree_count = call_votes
        elif put_votes > call_votes:
            direction = "PUT"
            agree_count = put_votes
        else:
            direction = "CALL" if change_pct >= 0 else "PUT"
            agree_count = max(call_votes, put_votes)

        # Direction alignment multiplier
        total_directional = call_votes + put_votes
        if total_directional > 0:
            direction_confidence = agree_count / total_directional
        else:
            direction_confidence = 0.5

        if agree_count >= STRONG_ALIGNMENT_MIN:
            multiplier = 1.3
        elif agree_count <= CONFLICT_MAX and call_votes > 0 and put_votes > 0:
            multiplier = 0.7
        else:
            multiplier = 1.0

        # Time decay: reduce score after 14:00 IST
        now_time = datetime.now().time()
        if now_time > dt_time(14, 0):
            # Linear decay from 1.0 at 14:00 to 0.7 at 15:15
            minutes_past_2 = (now_time.hour - 14) * 60 + now_time.minute
            decay = max(0.7, 1.0 - (minutes_past_2 / 75) * 0.3)
            multiplier *= decay

        final_score = min(100, raw_score * multiplier)

        # Compute ATR from 15m data for entry trigger
        atr = 0.0
        if not df_15m.empty and len(df_15m) >= 14:
            tr = pd.concat([
                df_15m["high"] - df_15m["low"],
                (df_15m["high"] - df_15m["close"].shift(1)).abs(),
                (df_15m["low"] - df_15m["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 0.0

        return EdgeScore(
            symbol=symbol,
            sector=sector,
            ltp=ltp,
            change_pct=change_pct,
            raw_score=round(raw_score, 1),
            final_score=round(final_score, 1),
            direction=direction,
            direction_confidence=round(direction_confidence, 2),
            layers=layers,
            atr=round(atr, 2),
        )

    # ── Layer Implementations ──────────────────────────────────────────────────

    def _layer_sector_momentum(
        self, sector: str, sector_ranks: Dict[str, int], sector_perf: pd.DataFrame
    ) -> LayerResult:
        """Layer 1: Is the stock's sector among the top performers?"""
        rank = sector_ranks.get(sector, len(sector_ranks))
        total = len(sector_ranks) or 1

        # Top 3 sectors get high score, bottom 3 also score (for PUT direction)
        if rank <= 3:
            score = 1.0 - (rank - 1) * 0.15  # 1.0, 0.85, 0.70
            direction = "CALL"
            detail = f"Sector #{rank}/{total} (Leading)"
        elif rank >= total - 2:
            score = 1.0 - (total - rank) * 0.15
            direction = "PUT"
            detail = f"Sector #{rank}/{total} (Lagging)"
        else:
            score = max(0.0, 0.5 - abs(rank - total / 2) / total)
            direction = "NEUTRAL"
            detail = f"Sector #{rank}/{total} (Mid-pack)"

        return LayerResult(name="sector_momentum", score=score, direction=direction, detail=detail)

    def _layer_relative_strength(
        self, symbol: str, sector: str, change_pct: float, stock_perf: pd.DataFrame
    ) -> LayerResult:
        """Layer 2: Is the stock outperforming its sector peers?"""
        sector_stocks = stock_perf[stock_perf["sector"] == sector]
        if sector_stocks.empty:
            return LayerResult(name="relative_strength", score=0.0, direction="NEUTRAL", detail="No peers")

        sector_avg = sector_stocks["pChange"].mean()
        sector_std = sector_stocks["pChange"].std()
        if sector_std == 0 or pd.isna(sector_std):
            sector_std = 1.0

        # Z-score relative to sector
        z_score = (change_pct - sector_avg) / sector_std

        if z_score > 1.5:
            score = 1.0
            direction = "CALL"
            detail = f"Strong outperformer (z={z_score:.1f})"
        elif z_score > 0.5:
            score = 0.6 + (z_score - 0.5) * 0.4
            direction = "CALL"
            detail = f"Outperforming sector (z={z_score:.1f})"
        elif z_score < -1.5:
            score = 1.0
            direction = "PUT"
            detail = f"Strong underperformer (z={z_score:.1f})"
        elif z_score < -0.5:
            score = 0.6 + abs(z_score + 0.5) * 0.4
            direction = "PUT"
            detail = f"Underperforming sector (z={z_score:.1f})"
        else:
            score = 0.3
            direction = "NEUTRAL"
            detail = f"In-line with sector (z={z_score:.1f})"

        return LayerResult(name="relative_strength", score=min(1.0, score), direction=direction, detail=detail)

    def _layer_vwap_location(
        self, df_15m: pd.DataFrame, ltp: float, target_date: date
    ) -> LayerResult:
        """Layer 3: Price location relative to VWAP — is it at a favorable zone?"""
        if df_15m.empty or ltp <= 0:
            return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail="No data")

        try:
            vwap_df = self._vwap.calculate(df_15m)
            today_vwap = vwap_df[vwap_df["timestamp"].dt.date == target_date]
            if today_vwap.empty or "vwap" not in today_vwap.columns:
                return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail="No VWAP")

            current_vwap = float(today_vwap["vwap"].iloc[-1])
            if current_vwap <= 0:
                return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail="Invalid VWAP")

            vwap_dist_pct = ((ltp - current_vwap) / current_vwap) * 100

            # Ideal CALL: price is 0.1%–0.8% above VWAP (riding momentum, not stretched)
            # Ideal PUT: price is 0.1%–0.8% below VWAP
            if 0.1 <= vwap_dist_pct <= 0.8:
                score = 1.0
                direction = "CALL"
                detail = f"Above VWAP +{vwap_dist_pct:.2f}% (ideal pullback zone)"
            elif 0.0 <= vwap_dist_pct <= 0.1:
                score = 0.8
                direction = "CALL"
                detail = f"At VWAP (bounce candidate)"
            elif 0.8 < vwap_dist_pct <= 1.5:
                score = 0.5
                direction = "CALL"
                detail = f"Above VWAP +{vwap_dist_pct:.2f}% (stretched)"
            elif vwap_dist_pct > 1.5:
                score = 0.2
                direction = "CALL"
                detail = f"Far above VWAP +{vwap_dist_pct:.2f}% (overextended)"
            elif -0.8 <= vwap_dist_pct < -0.1:
                score = 1.0
                direction = "PUT"
                detail = f"Below VWAP {vwap_dist_pct:.2f}% (ideal rejection zone)"
            elif -0.1 <= vwap_dist_pct < 0.0:
                score = 0.8
                direction = "PUT"
                detail = f"At VWAP (rejection candidate)"
            elif -1.5 <= vwap_dist_pct < -0.8:
                score = 0.5
                direction = "PUT"
                detail = f"Below VWAP {vwap_dist_pct:.2f}% (stretched)"
            else:
                score = 0.2
                direction = "PUT"
                detail = f"Far below VWAP {vwap_dist_pct:.2f}% (overextended)"

            return LayerResult(name="vwap_location", score=score, direction=direction, detail=detail)
        except Exception as e:
            LOGGER.debug(f"VWAP layer error for {df_15m.get('symbol', 'N/A')}: {e}")
            return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail=f"Error: {e}")

    def _layer_volume_confirmation(
        self, df_15m: pd.DataFrame, vol_surge: float, target_date: date
    ) -> LayerResult:
        """Layer 4: Volume surge + CVD direction confirms the move."""
        if df_15m.empty:
            return LayerResult(name="volume_confirmation", score=0.0, direction="NEUTRAL", detail="No data")

        # Volume surge scoring
        if vol_surge >= 2.5:
            vol_score = 1.0
        elif vol_surge >= 1.5:
            vol_score = 0.6 + (vol_surge - 1.5) * 0.4
        elif vol_surge >= 1.0:
            vol_score = 0.3
        else:
            vol_score = 0.1

        # CVD direction
        cvd_direction = "NEUTRAL"
        try:
            vd_df = self._volume_delta.calculate(df_15m)
            today_vd = vd_df[vd_df["timestamp"].dt.date == target_date]
            if not today_vd.empty and "cvd" in today_vd.columns:
                cvd_latest = float(today_vd["cvd"].iloc[-1])
                cvd_mid = float(today_vd["cvd"].iloc[len(today_vd) // 2]) if len(today_vd) > 2 else 0
                if cvd_latest > 0 and cvd_latest > cvd_mid:
                    cvd_direction = "CALL"
                    vol_score = min(1.0, vol_score + 0.2)
                elif cvd_latest < 0 and cvd_latest < cvd_mid:
                    cvd_direction = "PUT"
                    vol_score = min(1.0, vol_score + 0.2)
        except Exception:
            pass

        direction = cvd_direction if cvd_direction != "NEUTRAL" else ("CALL" if vol_surge >= 1.5 else "NEUTRAL")
        detail = f"Vol surge {vol_surge:.1f}x, CVD={cvd_direction}"

        return LayerResult(name="volume_confirmation", score=vol_score, direction=direction, detail=detail)

    def _layer_momentum_timing(
        self, df_15m: pd.DataFrame, ltp: float, change_pct: float, target_date: date
    ) -> LayerResult:
        """Layer 5: Is the move still early (catchable) or already exhausted?"""
        if df_15m.empty or ltp <= 0:
            return LayerResult(name="momentum_timing", score=0.0, direction="NEUTRAL", detail="No data")

        today_df = df_15m[df_15m["timestamp"].dt.date == target_date]
        if today_df.empty:
            return LayerResult(name="momentum_timing", score=0.0, direction="NEUTRAL", detail="No today data")

        day_high = float(today_df["high"].max())
        day_low = float(today_df["low"].min())
        day_open = float(today_df["open"].iloc[0])
        day_range = day_high - day_low

        if day_range <= 0:
            return LayerResult(name="momentum_timing", score=0.3, direction="NEUTRAL", detail="Flat day")

        # Range position: where is current price in today's range?
        range_position = (ltp - day_low) / day_range  # 0.0 = at low, 1.0 = at high

        # ORB detection: did price break above/below the first 30-min range?
        orb_candles = today_df.head(2)  # First 2 x 15-min = 30 minutes
        orb_high = float(orb_candles["high"].max())
        orb_low = float(orb_candles["low"].min())

        is_orb_breakout_up = ltp > orb_high and change_pct > 0
        is_orb_breakout_down = ltp < orb_low and change_pct < 0

        if is_orb_breakout_up:
            # Early in move if price is in first 40% of range from open
            if range_position < 0.6:
                score = 1.0
                detail = f"ORB breakout UP, range pos {range_position:.0%} (early)"
            else:
                score = 0.5
                detail = f"ORB breakout UP, range pos {range_position:.0%} (extended)"
            direction = "CALL"
        elif is_orb_breakout_down:
            if range_position > 0.4:
                score = 1.0
                detail = f"ORB breakdown DN, range pos {range_position:.0%} (early)"
            else:
                score = 0.5
                detail = f"ORB breakdown DN, range pos {range_position:.0%} (extended)"
            direction = "PUT"
        elif change_pct > 0 and range_position < 0.5:
            # Positive change but price is still in lower half = pullback opportunity
            score = 0.7
            direction = "CALL"
            detail = f"Bullish pullback, range pos {range_position:.0%}"
        elif change_pct < 0 and range_position > 0.5:
            score = 0.7
            direction = "PUT"
            detail = f"Bearish bounce, range pos {range_position:.0%}"
        else:
            score = 0.3
            direction = "NEUTRAL"
            detail = f"Range pos {range_position:.0%}, no clear timing"

        return LayerResult(name="momentum_timing", score=score, direction=direction, detail=detail)

    def _layer_supertrend_alignment(
        self, df_15m: pd.DataFrame, df_daily: pd.DataFrame
    ) -> LayerResult:
        """Layer 6: Do 15-minute and daily supertrend agree on direction?"""
        st_15m_dir = 0
        st_daily_dir = 0

        # 15-minute supertrend
        if not df_15m.empty and len(df_15m) >= 10:
            try:
                st_df = self._supertrend.calculate(df_15m)
                if "supertrend_direction" in st_df.columns:
                    st_15m_dir = int(st_df["supertrend_direction"].iloc[-1])
            except Exception:
                pass

        # Daily supertrend
        if not df_daily.empty and len(df_daily) >= 10:
            try:
                st_daily = self._supertrend.calculate(df_daily)
                if "supertrend_direction" in st_daily.columns:
                    st_daily_dir = int(st_daily["supertrend_direction"].iloc[-1])
            except Exception:
                pass

        if st_15m_dir == 1 and st_daily_dir == 1:
            score = 1.0
            direction = "CALL"
            detail = "Both 15m & Daily bullish ✅"
        elif st_15m_dir == -1 and st_daily_dir == -1:
            score = 1.0
            direction = "PUT"
            detail = "Both 15m & Daily bearish ✅"
        elif st_15m_dir == 1 and st_daily_dir == -1:
            score = 0.3
            direction = "CALL"
            detail = "15m bullish but Daily bearish ⚠️"
        elif st_15m_dir == -1 and st_daily_dir == 1:
            score = 0.3
            direction = "PUT"
            detail = "15m bearish but Daily bullish ⚠️"
        else:
            score = 0.2
            direction = "NEUTRAL"
            detail = "Supertrend data insufficient"

        return LayerResult(name="supertrend_alignment", score=score, direction=direction, detail=detail)

    def _layer_compression_release(
        self, df_15m: pd.DataFrame, change_pct: float, target_date: date
    ) -> LayerResult:
        """Layer 7: Was the stock compressed recently and is now expanding?"""
        if df_15m.empty or len(df_15m) < 20:
            return LayerResult(name="compression_release", score=0.0, direction="NEUTRAL", detail="Insufficient data")

        try:
            comp_df = self._compression.calculate(df_15m)

            # Check if stock was compressed in prior sessions
            prev_data = comp_df[comp_df["timestamp"].dt.date < target_date]
            today_data = comp_df[comp_df["timestamp"].dt.date == target_date]

            if prev_data.empty:
                return LayerResult(name="compression_release", score=0.0, direction="NEUTRAL", detail="No prior data")

            # Was compressed in last 3 sessions?
            recent_prev = prev_data.tail(30)  # ~2 sessions of 15m data
            was_compressed = bool(recent_prev["is_compressed"].any()) if "is_compressed" in recent_prev.columns else False
            had_nr7 = bool(recent_prev["nr7"].any()) if "nr7" in recent_prev.columns else False

            # Is today expanding? (today's range > yesterday's range)
            if not today_data.empty and "current_range" in today_data.columns:
                today_range = float(today_data["current_range"].mean())
                prev_range = float(recent_prev["current_range"].mean()) if "current_range" in recent_prev.columns else 0
                is_expanding = today_range > prev_range * 1.2 if prev_range > 0 else False
            else:
                is_expanding = abs(change_pct) > 1.5  # Fallback: >1.5% move = expansion

            if was_compressed and is_expanding:
                score = 1.0
                signals = []
                if had_nr7:
                    signals.append("NR7")
                signals.append("Coiled → Releasing")
                detail = f"🔥 {' + '.join(signals)}"
            elif was_compressed and not is_expanding:
                score = 0.6
                detail = "Compressed but not yet releasing"
            elif is_expanding and not was_compressed:
                score = 0.4
                detail = "Expanding (no prior compression)"
            else:
                score = 0.1
                detail = "No compression signal"

            direction = "CALL" if change_pct > 0 else ("PUT" if change_pct < 0 else "NEUTRAL")
            return LayerResult(name="compression_release", score=score, direction=direction, detail=detail)

        except Exception as e:
            LOGGER.debug(f"Compression layer error: {e}")
            return LayerResult(name="compression_release", score=0.0, direction="NEUTRAL", detail=f"Error")
