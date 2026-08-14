"""
IntradayEdgeScorer — Multi-layer confluence scoring engine for FO trading.
Supports three horizons:
  - INTRADAY: 15-minute base chart, Daily trend alignment.
  - WEEKLY: Daily base chart (swing), Weekly trend alignment.
  - MONTHLY: Weekly base chart (positional), Monthly trend alignment.

Combines 11 independent signal layers into a single EdgeScore (0–100) per stock:
  1. Sector Momentum        — Is the stock's sector leading in the horizon?
  2. Relative Strength       — Is the stock outperforming its sector peers?
  3. VWAP Location           — Is price at a favorable VWAP zone?
  4. Volume Confirmation     — Volume surge + buying/selling pressure (CVD)?
  5. Momentum Timing         — Is the move early (catchable) or exhausted?
  6. Supertrend Align        — Do base and higher timeframe supertrends agree?
  7. Compression Release     — Was the stock coiled and now releasing energy?
  8. Daily Trend Alignment   — Market regime gate + RSI sweet spot.
  9. Key Level Proximity     — Near PDH/PDL, 52W High, or Weekly High?
 10. OBV Divergence          — Smart money flow confirmation.
 11. Candle Quality          — Wick rejection and body strength filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping
from trade_system.domains.strategy.application.indicators.supertrend import SupertrendIndicator
from trade_system.domains.strategy.application.indicators.vwap import VWAPIndicator
from trade_system.domains.strategy.application.indicators.volume_delta import VolumeDeltaIndicator
from trade_system.domains.strategy.application.indicators.compression import CompressionIndicator
from trade_system.domains.strategy.application.indicators.obv import OBVIndicator

LOGGER = logging.getLogger(__name__)

# ── Layer Weights ──────────────────────────────────────────────────────────────
# 11 layers, sum = 1.0

LAYER_WEIGHTS = {
    "sector_momentum": 0.08,
    "relative_strength": 0.10,
    "vwap_location": 0.10,
    "volume_confirmation": 0.10,
    "momentum_timing": 0.10,
    "supertrend_alignment": 0.10,
    "compression_release": 0.10,
    "daily_trend_alignment": 0.12,
    "key_level_proximity": 0.08,
    "obv_divergence": 0.06,
    "candle_quality": 0.06,
}

# Direction agreement thresholds (scaled for 11 layers)
STRONG_ALIGNMENT_MIN = 7   # 7+ layers agreeing = strong alignment
CONFLICT_MAX = 4           # 4+ layers disagreeing = conflicted

# Banking / finance sectors — use BankNifty as regime signal instead of Nifty50
_BANKING_SECTORS = frozenset({"BANKING_PVT", "BANKING_PSU", "FINANCE"})


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
    Supports INTRADAY, WEEKLY, and MONTHLY horizons.
    """

    def __init__(self, horizon: str = "INTRADAY") -> None:
        self.engine = get_engine()
        self.sector_map = get_sector_mapping()
        self.horizon = horizon.upper()  # "INTRADAY", "WEEKLY", "MONTHLY"
        self._supertrend = SupertrendIndicator(period=7, multiplier=3)
        self._supertrend_daily = SupertrendIndicator(period=10, multiplier=2)
        self._vwap = VWAPIndicator()
        self._volume_delta = VolumeDeltaIndicator()
        self._compression = CompressionIndicator(atr_period=14, lookback=4)
        self._obv = OBVIndicator()
        # Cached per-scan: market regime (Nifty/BankNifty daily supertrend direction)
        self._market_regime_cache: Dict[str, int] = {}

    # ── Resampling Helpers ─────────────────────────────────────────────────────

    def _resample_candles(self, df_daily: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample daily candles to weekly ('W') or monthly ('ME') candles."""
        if df_daily.empty:
            return df_daily
        
        df = df_daily.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df.set_index("timestamp", inplace=True)
        
        resampled_groups = []
        for symbol, grp in df.groupby("symbol"):
            res = grp.resample(rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "symbol": "first"
            }).dropna()
            res.reset_index(inplace=True)
            resampled_groups.append(res)
            
        if not resampled_groups:
            return pd.DataFrame()
        return pd.concat(resampled_groups, ignore_index=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(
        self,
        target_date: date | None = None,
        sector_perf: pd.DataFrame | None = None,
        stock_perf: pd.DataFrame | None = None,
    ) -> List[EdgeScore]:
        """Run the full edge scan based on the selected horizon."""
        
        # ── 1. INTRADAY HORIZON PATHWAY ──
        if self.horizon == "INTRADAY":
            df_15m = self._fetch_15m_data(target_date)
            if df_15m.empty:
                return []
            
            if target_date is None:
                non_idx = df_15m[~df_15m["symbol"].str.contains("INDEX")]
                if non_idx.empty:
                    return []
                target_date = non_idx["timestamp"].max().date()
                
            # Fetch daily data with 365-day lookback (needed for 52W high/low in Layer 9)
            df_daily = self._fetch_daily_data(target_date, lookback_days=365)
            
            if sector_perf is None or stock_perf is None:
                sector_perf, stock_perf = self._compute_performance(df_15m, target_date)
                
            if stock_perf.empty:
                return []

            # Cache market regime once per scan (Nifty50 + BankNifty daily supertrend)
            self._market_regime_cache = self._compute_market_regime(df_daily)

            sector_ranks = self._rank_sectors(sector_perf)
            symbol_15m_groups = {sym: grp.sort_values("timestamp") for sym, grp in df_15m.groupby("symbol") if "INDEX" not in sym}
            symbol_daily_groups = {sym: grp.sort_values("timestamp") for sym, grp in df_daily.groupby("symbol")} if not df_daily.empty else {}
            
            results: List[EdgeScore] = []
            for _, row in stock_perf.iterrows():
                symbol = row["symbol"]
                sector = row.get("sector", self.sector_map.get(symbol, "UNKNOWN"))
                if sector == "UNKNOWN": continue
                
                ltp = float(row.get("close_last", 0))
                change_pct = float(row.get("pChange", 0))
                vol_surge = float(row.get("vol_surge", 0))
                
                df_sym_15m = symbol_15m_groups.get(symbol, pd.DataFrame())
                df_sym_daily = symbol_daily_groups.get(symbol, pd.DataFrame())
                
                edge = self._score_stock(
                    symbol=symbol, sector=sector, ltp=ltp, change_pct=change_pct, vol_surge=vol_surge,
                    sector_ranks=sector_ranks, sector_perf=sector_perf, stock_perf=stock_perf,
                    df_base=df_sym_15m, df_htf=df_sym_daily, target_date=target_date
                )
                results.append(edge)
                
            results.sort(key=lambda e: e.final_score, reverse=True)
            return results

        # ── 2. WEEKLY & MONTHLY HORIZON PATHWAYS ──
        else:
            # Lookbacks: Weekly swing uses 365 days; Monthly positional uses 1000 days
            lookback_days = 365 if self.horizon == "WEEKLY" else 1000
            
            # If target_date is not specified, resolve from DB latest date
            if target_date is None:
                engine = get_engine()
                with engine.connect() as conn:
                    max_ts = conn.execute(text("SELECT MAX(timestamp) FROM ohlcv_daily")).scalar()
                    if max_ts:
                        target_date = pd.to_datetime(max_ts).date()
                    else:
                        target_date = date.today()
                        
            df_daily_all = self._fetch_daily_data(target_date, lookback_days=lookback_days)
            if df_daily_all.empty:
                LOGGER.warning(f"No daily data found for {self.horizon} scan.")
                return []
                
            # Compute base & HTF dataframes based on horizon
            if self.horizon == "WEEKLY":
                df_base = df_daily_all
                df_htf = self._resample_candles(df_daily_all, "W")
            else:  # MONTHLY
                df_base = self._resample_candles(df_daily_all, "W")
                df_htf = self._resample_candles(df_daily_all, "ME")
                
            # Sector and Stock Performance Calculations
            sector_perf, stock_perf = self._compute_horizon_performance(df_base, target_date)
            if stock_perf.empty:
                return []
                
            sector_ranks = self._rank_sectors(sector_perf)
            symbol_base_groups = {sym: grp.sort_values("timestamp") for sym, grp in df_base.groupby("symbol")}
            symbol_htf_groups = {sym: grp.sort_values("timestamp") for sym, grp in df_htf.groupby("symbol")}
            
            results: List[EdgeScore] = []
            for _, row in stock_perf.iterrows():
                symbol = row["symbol"]
                sector = row.get("sector", self.sector_map.get(symbol, "UNKNOWN"))
                if sector == "UNKNOWN": continue
                
                ltp = float(row.get("close_last", 0))
                change_pct = float(row.get("pChange", 0))
                vol_surge = float(row.get("vol_surge", 0))
                
                df_sym_base = symbol_base_groups.get(symbol, pd.DataFrame())
                df_sym_htf = symbol_htf_groups.get(symbol, pd.DataFrame())
                
                edge = self._score_stock(
                    symbol=symbol, sector=sector, ltp=ltp, change_pct=change_pct, vol_surge=vol_surge,
                    sector_ranks=sector_ranks, sector_perf=sector_perf, stock_perf=stock_perf,
                    df_base=df_sym_base, df_htf=df_sym_htf, target_date=target_date
                )
                results.append(edge)
                
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
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch 15m data: {e}")
            return pd.DataFrame()

    def _fetch_daily_data(self, target_date: date, lookback_days: int = 60) -> pd.DataFrame:
        """Fetch daily candles from the database."""
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

    def _compute_horizon_performance(
        self, df_base: pd.DataFrame, target_date: date
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute sector and stock performance for Weekly and Monthly horizons."""
        df_base = df_base.copy()
        df_base["sector"] = df_base["symbol"].apply(lambda s: self.sector_map.get(s, "UNKNOWN"))
        df_base["date"] = df_base["timestamp"].dt.date
        
        # Lookback: Weekly uses 5 days, Monthly uses 20 days
        lookback = 5 if self.horizon == "WEEKLY" else 20
        
        results = []
        for symbol, grp in df_base.groupby("symbol"):
            grp_sorted = grp.sort_values("timestamp")
            # Filter candles up to target date
            grp_filtered = grp_sorted[grp_sorted["date"] <= target_date]
            if len(grp_filtered) < lookback + 1:
                continue
                
            latest = grp_filtered.iloc[-1]
            prev = grp_filtered.iloc[-(lookback + 1)]
            
            pchange = ((latest["close"] - prev["close"]) / prev["close"]) * 100
            
            # Volume surge
            vol_today = float(latest["volume"])
            avg_vol_window = grp_filtered["volume"].tail(lookback * 4).mean()
            vol_surge = vol_today / avg_vol_window if avg_vol_window > 0 else 0.0
            
            results.append({
                "symbol": symbol,
                "sector": latest["sector"],
                "close_last": float(latest["close"]),
                "close_prev": float(prev["close"]),
                "pChange": pchange,
                "vol_surge": vol_surge
            })
            
        if not results:
            return pd.DataFrame(columns=["sector", "pChange"]), pd.DataFrame()
            
        stock_perf = pd.DataFrame(results)
        sector_perf = (
            stock_perf.groupby("sector")["pChange"]
            .mean()
            .reset_index()
            .query("sector != 'UNKNOWN'")
            .sort_values("pChange", ascending=False)
        )
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
        df_base: pd.DataFrame,
        df_htf: pd.DataFrame,
        target_date: date,
    ) -> EdgeScore:
        """Compute the 11-layer EdgeScore for a single stock."""
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
        layers["vwap_location"] = self._layer_vwap_location(df_base, ltp, target_date)

        # Layer 4: Volume Confirmation
        layers["volume_confirmation"] = self._layer_volume_confirmation(
            df_base, vol_surge, target_date
        )

        # Layer 5: Momentum Timing
        layers["momentum_timing"] = self._layer_momentum_timing(
            df_base, ltp, change_pct, target_date
        )

        # Layer 6: Supertrend Alignment
        layers["supertrend_alignment"] = self._layer_supertrend_alignment(
            df_base, df_htf
        )

        # Layer 7: Compression Release
        layers["compression_release"] = self._layer_compression_release(
            df_base, change_pct, target_date
        )

        # Layer 8: Daily Trend Alignment (regime gate + RSI sweet spot)
        layers["daily_trend_alignment"] = self._layer_daily_trend_alignment(
            df_htf, sector, target_date
        )

        # Layer 9: Key Level Proximity (PDH/PDL, 52W, Weekly High)
        layers["key_level_proximity"] = self._layer_key_level_proximity(
            df_htf, ltp, change_pct, vol_surge, target_date
        )

        # Layer 10: OBV Divergence (smart money confirmation)
        layers["obv_divergence"] = self._layer_obv_divergence(
            df_base, change_pct, target_date
        )

        # Layer 11: Candle Quality (wick rejection + body strength)
        layers["candle_quality"] = self._layer_candle_quality(
            df_base, change_pct, target_date
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

        # Time decay applies ONLY to Intraday trading
        if self.horizon == "INTRADAY":
            now_time = datetime.now().time()
            if now_time > dt_time(14, 0):
                minutes_past_2 = (now_time.hour - 14) * 60 + now_time.minute
                decay = max(0.7, 1.0 - (minutes_past_2 / 75) * 0.3)
                multiplier *= decay

        final_score = min(100, raw_score * multiplier)

        # Compute ATR from base data for entry trigger
        atr = 0.0
        if not df_base.empty and len(df_base) >= 14:
            tr = pd.concat([
                df_base["high"] - df_base["low"],
                (df_base["high"] - df_base["close"].shift(1)).abs(),
                (df_base["low"] - df_base["close"].shift(1)).abs(),
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

        if rank <= 3:
            score = 1.0 - (rank - 1) * 0.15
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
        self, df_base: pd.DataFrame, ltp: float, target_date: date
    ) -> LayerResult:
        """Layer 3: Price location relative to VWAP (Daily Session VWAP or Rolling Multi-day VWAP)."""
        if df_base.empty or ltp <= 0:
            return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail="No data")

        try:
            if self.horizon == "INTRADAY":
                # Session VWAP
                vwap_df = self._vwap.calculate(df_base)
                today_vwap = vwap_df[vwap_df["timestamp"].dt.date == target_date]
                if today_vwap.empty or "vwap" not in today_vwap.columns:
                    return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail="No VWAP")
                current_vwap = float(today_vwap["vwap"].iloc[-1])
            else:
                # Rolling VWAP: Weekly uses 20 days, Monthly uses 252 days
                lookback = 20 if self.horizon == "WEEKLY" else 252
                df = df_base.copy()
                df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
                df["tp_vol"] = df["tp"] * df["volume"]
                df["cum_tp_vol"] = df["tp_vol"].rolling(window=lookback).sum()
                df["cum_vol"] = df["volume"].rolling(window=lookback).sum()
                df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]
                
                # Filter to target_date
                target_df = df[df["timestamp"].dt.date <= target_date]
                if target_df.empty or pd.isna(target_df["vwap"].iloc[-1]):
                    return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail="No Rolling VWAP")
                current_vwap = float(target_df["vwap"].iloc[-1])

            if current_vwap <= 0:
                return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail="Invalid VWAP")

            vwap_dist_pct = ((ltp - current_vwap) / current_vwap) * 100

            # Slightly wider thresholds for Weekly/Monthly
            threshold = 0.8 if self.horizon == "INTRADAY" else 2.5
            extreme_threshold = 1.5 if self.horizon == "INTRADAY" else 5.0

            if 0.1 <= vwap_dist_pct <= threshold:
                score = 1.0
                direction = "CALL"
                detail = f"Above VWAP +{vwap_dist_pct:.2f}% (pullback zone)"
            elif 0.0 <= vwap_dist_pct <= 0.1:
                score = 0.8
                direction = "CALL"
                detail = f"At VWAP (bounce candidate)"
            elif threshold < vwap_dist_pct <= extreme_threshold:
                score = 0.5
                direction = "CALL"
                detail = f"Above VWAP +{vwap_dist_pct:.2f}% (stretched)"
            elif vwap_dist_pct > extreme_threshold:
                score = 0.2
                direction = "CALL"
                detail = f"Far above VWAP +{vwap_dist_pct:.2f}% (overextended)"
            elif -threshold <= vwap_dist_pct < -0.1:
                score = 1.0
                direction = "PUT"
                detail = f"Below VWAP {vwap_dist_pct:.2f}% (rejection zone)"
            elif -0.1 <= vwap_dist_pct < 0.0:
                score = 0.8
                direction = "PUT"
                detail = f"At VWAP (rejection candidate)"
            elif -extreme_threshold <= vwap_dist_pct < -threshold:
                score = 0.5
                direction = "PUT"
                detail = f"Below VWAP {vwap_dist_pct:.2f}% (stretched)"
            else:
                score = 0.2
                direction = "PUT"
                detail = f"Far below VWAP {vwap_dist_pct:.2f}% (overextended)"

            return LayerResult(name="vwap_location", score=score, direction=direction, detail=detail)
        except Exception as e:
            return LayerResult(name="vwap_location", score=0.0, direction="NEUTRAL", detail=f"Error: {e}")

    def _layer_volume_confirmation(
        self, df_base: pd.DataFrame, vol_surge: float, target_date: date
    ) -> LayerResult:
        """Layer 4: Volume surge + CVD trend confirmation."""
        if df_base.empty:
            return LayerResult(name="volume_confirmation", score=0.0, direction="NEUTRAL", detail="No data")

        if vol_surge >= 2.5:
            vol_score = 1.0
        elif vol_surge >= 1.5:
            vol_score = 0.6 + (vol_surge - 1.5) * 0.4
        elif vol_surge >= 1.0:
            vol_score = 0.3
        else:
            vol_score = 0.1

        cvd_direction = "NEUTRAL"
        try:
            vd_df = self._volume_delta.calculate(df_base)
            target_vd = vd_df[vd_df["timestamp"].dt.date <= target_date]
            if not target_vd.empty and "cvd" in target_vd.columns:
                cvd_latest = float(target_vd["cvd"].iloc[-1])
                window = min(5, len(target_vd))
                cvd_prev = float(target_vd["cvd"].iloc[-window])
                if cvd_latest > cvd_prev:
                    cvd_direction = "CALL"
                    vol_score = min(1.0, vol_score + 0.2)
                elif cvd_latest < cvd_prev:
                    cvd_direction = "PUT"
                    vol_score = min(1.0, vol_score + 0.2)
        except Exception:
            pass

        direction = cvd_direction if cvd_direction != "NEUTRAL" else ("CALL" if vol_surge >= 1.5 else "NEUTRAL")
        detail = f"Vol surge {vol_surge:.1f}x, CVD={cvd_direction}"

        return LayerResult(name="volume_confirmation", score=vol_score, direction=direction, detail=detail)

    def _layer_momentum_timing(
        self, df_base: pd.DataFrame, ltp: float, change_pct: float, target_date: date
    ) -> LayerResult:
        """Layer 5: Breakout of range boundaries (Intraday ORB, Previous Week's range, or Previous Month's range)."""
        if df_base.empty or ltp <= 0:
            return LayerResult(name="momentum_timing", score=0.0, direction="NEUTRAL", detail="No data")

        if self.horizon == "INTRADAY":
            today_df = df_base[df_base["timestamp"].dt.date == target_date]
            if today_df.empty:
                return LayerResult(name="momentum_timing", score=0.0, direction="NEUTRAL", detail="No today data")
            day_high = float(today_df["high"].max())
            day_low = float(today_df["low"].min())
            day_open = float(today_df["open"].iloc[0])
            day_range = day_high - day_low
            if day_range <= 0:
                return LayerResult(name="momentum_timing", score=0.3, direction="NEUTRAL", detail="Flat day")

            range_position = (ltp - day_low) / day_range
            orb_candles = today_df.head(2)
            orb_high = float(orb_candles["high"].max())
            orb_low = float(orb_candles["low"].min())

            if ltp > orb_high and change_pct > 0:
                score = 1.0 if range_position < 0.6 else 0.5
                direction = "CALL"
                detail = f"ORB High breakout, range pos {range_position:.0%}"
            elif ltp < orb_low and change_pct < 0:
                score = 1.0 if range_position > 0.4 else 0.5
                direction = "PUT"
                detail = f"ORB Low breakdown, range pos {range_position:.0%}"
            else:
                score = 0.3
                direction = "NEUTRAL"
                detail = f"Inside ORB range pos {range_position:.0%}"
            return LayerResult(name="momentum_timing", score=score, direction=direction, detail=detail)

        else:
            base_filtered = df_base[df_base["timestamp"].dt.date <= target_date].sort_values("timestamp")
            if len(base_filtered) < 2:
                return LayerResult(name="momentum_timing", score=0.0, direction="NEUTRAL", detail="Insufficient history")
                
            prev_candle = base_filtered.iloc[-2]
            prev_high = float(prev_candle["high"])
            prev_low = float(prev_candle["low"])
            
            ref_label = "Prev Week" if self.horizon == "WEEKLY" else "Prev Month"
            
            if ltp > prev_high:
                fresh_dist = (ltp - prev_high) / prev_high
                score = 1.0 if fresh_dist <= 0.015 else 0.5
                direction = "CALL"
                detail = f"Breakout above {ref_label} High ₹{prev_high:.2f} (+{fresh_dist:.1%})"
            elif ltp < prev_low:
                fresh_dist = (prev_low - ltp) / prev_low
                score = 1.0 if fresh_dist <= 0.015 else 0.5
                direction = "PUT"
                detail = f"Breakdown below {ref_label} Low ₹{prev_low:.2f} (-{fresh_dist:.1%})"
            else:
                range_size = prev_high - prev_low
                range_pos = (ltp - prev_low) / range_size if range_size > 0 else 0.5
                score = 0.5
                direction = "CALL" if range_pos > 0.5 else "PUT"
                detail = f"Pullback inside {ref_label} range (pos: {range_pos:.0%})"
                
            return LayerResult(name="momentum_timing", score=score, direction=direction, detail=detail)

    def _layer_supertrend_alignment(
        self, df_base: pd.DataFrame, df_htf: pd.DataFrame
    ) -> LayerResult:
        """Layer 6: Do base timeframe and higher timeframe supertrends agree?"""
        st_base_dir = 0
        st_htf_dir = 0

        if not df_base.empty and len(df_base) >= 10:
            try:
                st_df = self._supertrend.calculate(df_base)
                if "supertrend_direction" in st_df.columns:
                    st_base_dir = int(st_df["supertrend_direction"].iloc[-1])
            except Exception:
                pass

        if not df_htf.empty and len(df_htf) >= 10:
            try:
                st_htf = self._supertrend.calculate(df_htf)
                if "supertrend_direction" in st_htf.columns:
                    st_htf_dir = int(st_htf["supertrend_direction"].iloc[-1])
            except Exception:
                pass

        base_label = "15m" if self.horizon == "INTRADAY" else "Daily" if self.horizon == "WEEKLY" else "Weekly"
        htf_label = "Daily" if self.horizon == "INTRADAY" else "Weekly" if self.horizon == "WEEKLY" else "Monthly"

        if st_base_dir == 1 and st_htf_dir == 1:
            score = 1.0
            direction = "CALL"
            detail = f"Both {base_label} & {htf_label} bullish ✅"
        elif st_base_dir == -1 and st_htf_dir == -1:
            score = 1.0
            direction = "PUT"
            detail = f"Both {base_label} & {htf_label} bearish ✅"
        elif st_base_dir == 1 and st_htf_dir == -1:
            score = 0.3
            direction = "CALL"
            detail = f"{base_label} bullish but {htf_label} bearish ⚠️"
        elif st_base_dir == -1 and st_htf_dir == 1:
            score = 0.3
            direction = "PUT"
            detail = f"{base_label} bearish but {htf_label} bullish ⚠️"
        else:
            score = 0.2
            direction = "NEUTRAL"
            detail = "Supertrend data insufficient"

        return LayerResult(name="supertrend_alignment", score=score, direction=direction, detail=detail)

    def _layer_compression_release(
        self, df_base: pd.DataFrame, change_pct: float, target_date: date
    ) -> LayerResult:
        """Layer 7: Was the stock coiled recently (TTM Squeeze/Inside candle/NR7) and is now expanding?"""
        if df_base.empty or len(df_base) < 20:
            return LayerResult(name="compression_release", score=0.0, direction="NEUTRAL", detail="Insufficient data")

        try:
            comp_df = self._compression.calculate(df_base)

            prev_data = comp_df[comp_df["timestamp"].dt.date < target_date]
            today_data = comp_df[comp_df["timestamp"].dt.date == target_date]

            if prev_data.empty:
                return LayerResult(name="compression_release", score=0.0, direction="NEUTRAL", detail="No prior data")

            window = 30
            recent_prev = prev_data.tail(window)
            
            was_compressed = bool(recent_prev["is_compressed"].any()) if "is_compressed" in recent_prev.columns else False
            had_nr7 = bool(recent_prev["nr7"].any()) if "nr7" in recent_prev.columns else False

            if not today_data.empty and "current_range" in today_data.columns:
                today_range = float(today_data["current_range"].mean())
                prev_range = float(recent_prev["current_range"].mean()) if "current_range" in recent_prev.columns else 0
                is_expanding = today_range > prev_range * 1.2 if prev_range > 0 else False
            else:
                is_expanding = abs(change_pct) > (1.5 if self.horizon == "INTRADAY" else 3.0)

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

        except Exception:
            return LayerResult(name="compression_release", score=0.0, direction="NEUTRAL", detail=f"Error")

    # ── Helper: Market Regime Cache ────────────────────────────────────────────

    def _compute_market_regime(self, df_daily: pd.DataFrame) -> Dict[str, int]:
        """Compute and cache Nifty50 and BankNifty daily supertrend direction.

        Returns a dict like {"NIFTY": 1, "BANKNIFTY": -1} where 1=bull, -1=bear, 0=unknown.
        """
        regime: Dict[str, int] = {"NIFTY": 0, "BANKNIFTY": 0}
        for symbol, key in [("NSE:NIFTY50-INDEX", "NIFTY"), ("NSE:NIFTYBANK-INDEX", "BANKNIFTY")]:
            try:
                sym_df = df_daily[df_daily["symbol"] == symbol].sort_values("timestamp")
                if len(sym_df) >= 15:
                    st_df = self._supertrend_daily.calculate(sym_df.copy())
                    if "supertrend_direction" in st_df.columns:
                        regime[key] = int(st_df["supertrend_direction"].iloc[-1])
            except Exception as e:
                LOGGER.debug(f"Market regime computation failed for {symbol}: {e}")
        return regime

    @staticmethod
    def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
        """Compute Wilder's RSI from a close price series. Returns the latest value."""
        if len(closes) < period + 1:
            return 50.0  # neutral fallback
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if pd.notna(val) else 50.0

    # ── Layer 8: Daily Trend Alignment ─────────────────────────────────────────

    def _layer_daily_trend_alignment(
        self, df_htf: pd.DataFrame, sector: str, target_date: date
    ) -> LayerResult:
        """Layer 8: Market regime gate + daily RSI sweet-spot check.

        Checks:
        1. Is the broad market (Nifty/BankNifty) daily supertrend bullish or bearish?
        2. Is the stock's own daily supertrend aligned?
        3. Is the stock's daily RSI in the 'sweet spot' (not exhausted)?
        """
        if df_htf.empty or len(df_htf) < 15:
            return LayerResult(name="daily_trend_alignment", score=0.0, direction="NEUTRAL", detail="Insufficient daily data")

        # 1. Stock's daily supertrend
        stock_st_dir = 0
        try:
            st_df = self._supertrend_daily.calculate(df_htf.copy())
            if "supertrend_direction" in st_df.columns:
                stock_st_dir = int(st_df["supertrend_direction"].iloc[-1])
        except Exception:
            pass

        # 2. Market regime (cached per scan)
        regime_key = "BANKNIFTY" if sector in _BANKING_SECTORS else "NIFTY"
        market_regime = self._market_regime_cache.get(regime_key, 0)

        # 3. Daily RSI
        closes = df_htf["close"]
        daily_rsi = self._compute_rsi(closes)

        # Scoring logic
        if stock_st_dir == 0:
            return LayerResult(name="daily_trend_alignment", score=0.2, direction="NEUTRAL",
                               detail="Daily ST unknown")

        stock_direction = "CALL" if stock_st_dir == 1 else "PUT"

        # RSI sweet spot ranges
        if stock_direction == "CALL":
            rsi_in_sweet = 40 <= daily_rsi <= 65
            rsi_extreme = daily_rsi >= 75  # overbought exhaustion
        else:
            rsi_in_sweet = 35 <= daily_rsi <= 60
            rsi_extreme = daily_rsi <= 25  # oversold exhaustion

        regime_agrees = (market_regime == stock_st_dir)

        if regime_agrees and rsi_in_sweet:
            score = 1.0
            detail = f"Regime ✅ ST={stock_direction} RSI={daily_rsi:.0f} (sweet spot)"
        elif regime_agrees and not rsi_extreme:
            score = 0.7
            detail = f"Regime ✅ ST={stock_direction} RSI={daily_rsi:.0f}"
        elif regime_agrees and rsi_extreme:
            score = 0.4
            detail = f"Regime ✅ but RSI={daily_rsi:.0f} (exhaustion risk)"
        elif not regime_agrees and not rsi_extreme:
            score = 0.3
            detail = f"Regime ⚠️ ({regime_key} opposes) RSI={daily_rsi:.0f}"
        else:
            # Regime opposes AND RSI is extreme
            score = 0.0
            detail = f"Regime ❌ + RSI={daily_rsi:.0f} (exhausted, avoid)"

        return LayerResult(name="daily_trend_alignment", score=score, direction=stock_direction, detail=detail)

    # ── Layer 9: Key Level Proximity ───────────────────────────────────────────

    def _layer_key_level_proximity(
        self, df_htf: pd.DataFrame, ltp: float, change_pct: float,
        vol_surge: float, target_date: date
    ) -> LayerResult:
        """Layer 9: Proximity to previous day's high/low, weekly high, 52-week high.

        Stocks near key structural levels have higher probability of follow-through.
        """
        if df_htf.empty or ltp <= 0:
            return LayerResult(name="key_level_proximity", score=0.0, direction="NEUTRAL", detail="No data")

        df_sorted = df_htf.sort_values("timestamp")
        # Filter to data up to (not including) target_date for previous levels
        prev_data = df_sorted[df_sorted["timestamp"].dt.date < target_date]
        if prev_data.empty or len(prev_data) < 2:
            return LayerResult(name="key_level_proximity", score=0.0, direction="NEUTRAL", detail="Insufficient history")

        # Previous Day High / Low / Close
        prev_day_date = prev_data["timestamp"].dt.date.max()
        prev_day = prev_data[prev_data["timestamp"].dt.date == prev_day_date]
        pdh = float(prev_day["high"].max()) if not prev_day.empty else 0
        pdl = float(prev_day["low"].min()) if not prev_day.empty else 0

        # Weekly High (last 5 trading days)
        last_5d = prev_data.tail(5) if len(prev_data) >= 5 else prev_data
        weekly_high = float(last_5d["high"].max())
        weekly_low = float(last_5d["low"].min())

        # 52-Week High (up to ~250 trading days of available data)
        fifty_two_w_high = float(prev_data["high"].max())
        fifty_two_w_low = float(prev_data["low"].min())

        # Distance calculations (percentage)
        dist_pdh = ((pdh - ltp) / ltp * 100) if pdh > 0 else 999
        dist_pdl = ((ltp - pdl) / ltp * 100) if pdl > 0 else 999
        dist_weekly_high = ((weekly_high - ltp) / ltp * 100) if weekly_high > 0 else 999
        dist_52w_high = ((fifty_two_w_high - ltp) / ltp * 100) if fifty_two_w_high > 0 else 999

        # Scoring
        # 52W High breakout with volume confirmation = highest conviction
        if dist_52w_high <= 0.3 and vol_surge >= 1.5 and change_pct > 0:
            return LayerResult(name="key_level_proximity", score=1.0, direction="CALL",
                               detail=f"🏆 Near 52W High ₹{fifty_two_w_high:.0f} ({dist_52w_high:+.1f}%) + Vol {vol_surge:.1f}x")

        if change_pct > 0:
            # Bullish — near PDH or weekly high is positive
            if ltp > pdh and dist_pdh <= 0:
                score = 1.0
                detail = f"Above PDH ₹{pdh:.0f} (breakout zone)"
            elif abs(dist_pdh) <= 0.5:
                score = 0.9
                detail = f"Near PDH ₹{pdh:.0f} ({dist_pdh:+.1f}%)"
            elif abs(dist_weekly_high) <= 0.5:
                score = 0.8
                detail = f"Near Weekly High ₹{weekly_high:.0f} ({dist_weekly_high:+.1f}%)"
            elif abs(dist_52w_high) <= 1.0:
                score = 0.7
                detail = f"Near 52W High ₹{fifty_two_w_high:.0f} ({dist_52w_high:+.1f}%)"
            elif ltp > pdl and ltp < pdh:
                score = 0.4
                detail = f"Inside PDH-PDL range"
            else:
                score = 0.2
                detail = f"Below PDL — counter-trend risk"
            direction = "CALL"
        else:
            # Bearish — near PDL is positive for short
            if ltp < pdl and dist_pdl <= 0:
                score = 1.0
                detail = f"Below PDL ₹{pdl:.0f} (breakdown zone)"
            elif abs(dist_pdl) <= 0.5:
                score = 0.9
                detail = f"Near PDL ₹{pdl:.0f} ({dist_pdl:+.1f}%)"
            elif ltp > pdl and ltp < pdh:
                score = 0.4
                detail = f"Inside PDH-PDL range"
            else:
                score = 0.2
                detail = f"Above PDH — counter-trend risk for short"
            direction = "PUT"

        return LayerResult(name="key_level_proximity", score=score, direction=direction, detail=detail)

    # ── Layer 10: OBV Divergence ───────────────────────────────────────────────

    def _layer_obv_divergence(
        self, df_base: pd.DataFrame, change_pct: float, target_date: date
    ) -> LayerResult:
        """Layer 10: On-Balance Volume divergence detection.

        Compares OBV slope vs price slope over a lookback window to detect
        accumulation (bullish) or distribution (bearish) divergences.
        """
        if df_base.empty or len(df_base) < 10:
            return LayerResult(name="obv_divergence", score=0.0, direction="NEUTRAL", detail="Insufficient data")

        try:
            if self.horizon == "INTRADAY":
                session_df = df_base[df_base["timestamp"].dt.date == target_date]
                if len(session_df) < 5:
                    # Fallback to all available data
                    session_df = df_base.tail(20)
            else:
                session_df = df_base.tail(20)

            if len(session_df) < 5:
                return LayerResult(name="obv_divergence", score=0.0, direction="NEUTRAL", detail="Too few bars")

            obv_df = self._obv.calculate(session_df.copy())
            if "obv" not in obv_df.columns:
                return LayerResult(name="obv_divergence", score=0.0, direction="NEUTRAL", detail="OBV calc failed")

            # Use last 5 bars for slope comparison
            lookback = min(5, len(obv_df))
            recent = obv_df.tail(lookback)

            obv_values = recent["obv"].values
            close_values = recent["close"].values

            # Simple linear slope (normalized)
            x = np.arange(lookback, dtype=float)
            if len(x) < 2:
                return LayerResult(name="obv_divergence", score=0.0, direction="NEUTRAL", detail="Not enough points")

            obv_slope = np.polyfit(x, obv_values, 1)[0]
            price_slope = np.polyfit(x, close_values, 1)[0]

            # Normalize slopes by their mean to make them comparable
            obv_mean = np.abs(obv_values).mean()
            price_mean = np.abs(close_values).mean()
            norm_obv_slope = obv_slope / obv_mean if obv_mean > 0 else 0
            norm_price_slope = price_slope / price_mean if price_mean > 0 else 0

            # Determine agreement
            obv_rising = norm_obv_slope > 0.001
            obv_falling = norm_obv_slope < -0.001
            price_rising = norm_price_slope > 0.001
            price_falling = norm_price_slope < -0.001

            if obv_rising and price_rising:
                score = 1.0
                direction = "CALL"
                detail = "OBV confirming uptrend (accumulation ✅)"
            elif obv_falling and price_falling:
                score = 1.0
                direction = "PUT"
                detail = "OBV confirming downtrend (distribution ✅)"
            elif obv_rising and price_falling:
                # Bullish divergence — smart money accumulating despite price drop
                score = 0.6
                direction = "CALL"
                detail = "Bullish OBV divergence (hidden accumulation)"
            elif obv_falling and price_rising:
                # Bearish divergence — distribution despite price rise
                score = 0.2
                direction = "PUT"
                detail = "⚠️ Bearish OBV divergence (distribution risk)"
            elif obv_rising and not price_rising and not price_falling:
                score = 0.7
                direction = "CALL"
                detail = "OBV rising, price flat (building pressure)"
            elif obv_falling and not price_rising and not price_falling:
                score = 0.7
                direction = "PUT"
                detail = "OBV falling, price flat (weakening)"
            else:
                score = 0.3
                direction = "NEUTRAL"
                detail = "OBV inconclusive"

            return LayerResult(name="obv_divergence", score=score, direction=direction, detail=detail)

        except Exception as e:
            LOGGER.debug(f"OBV divergence calculation error: {e}")
            return LayerResult(name="obv_divergence", score=0.0, direction="NEUTRAL", detail="Error")

    # ── Layer 11: Candle Quality ───────────────────────────────────────────────

    def _layer_candle_quality(
        self, df_base: pd.DataFrame, change_pct: float, target_date: date
    ) -> LayerResult:
        """Layer 11: Candle body strength and wick rejection filter.

        Analyzes the latest candle's structure to determine if entry quality is good.
        """
        if df_base.empty:
            return LayerResult(name="candle_quality", score=0.0, direction="NEUTRAL", detail="No data")

        if self.horizon == "INTRADAY":
            session_df = df_base[df_base["timestamp"].dt.date == target_date]
            if len(session_df) < 2:
                session_df = df_base.tail(5)
        else:
            session_df = df_base.tail(5)

        if len(session_df) < 2:
            return LayerResult(name="candle_quality", score=0.3, direction="NEUTRAL", detail="Too few candles")

        latest = session_df.iloc[-1]
        prev = session_df.iloc[-2]

        o, h, l, c = float(latest["open"]), float(latest["high"]), float(latest["low"]), float(latest["close"])
        candle_range = h - l
        if candle_range <= 0:
            return LayerResult(name="candle_quality", score=0.3, direction="NEUTRAL", detail="Flat candle")

        body = abs(c - o)
        body_ratio = body / candle_range
        is_green = c > o
        is_red = c < o

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        upper_wick_pct = upper_wick / candle_range
        lower_wick_pct = lower_wick / candle_range

        # Check for engulfing against signal direction
        prev_o, prev_c = float(prev["open"]), float(prev["close"])
        prev_is_green = prev_c > prev_o
        prev_body = abs(prev_c - prev_o)

        signal_dir = "CALL" if change_pct > 0 else ("PUT" if change_pct < 0 else "NEUTRAL")

        if signal_dir == "CALL":
            # Bearish engulfing against CALL = danger
            if is_red and body > prev_body and not prev_is_green:
                return LayerResult(name="candle_quality", score=0.0, direction="PUT",
                                   detail="⚠️ Bearish engulfing — reversal signal")
            # Strong wick rejection against CALL
            if upper_wick_pct > 0.40:
                return LayerResult(name="candle_quality", score=0.2, direction="CALL",
                                   detail=f"Upper wick rejection ({upper_wick_pct:.0%})")
            # Strong green candle
            if is_green and body_ratio > 0.70:
                score = 1.0
                detail = f"Strong bullish candle (body {body_ratio:.0%})"
            elif is_green and body_ratio > 0.50:
                score = 0.8
                detail = f"Solid bullish candle (body {body_ratio:.0%})"
            elif body_ratio < 0.30:
                score = 0.5
                detail = f"Doji / small body ({body_ratio:.0%}) — indecisive"
            else:
                score = 0.6
                detail = f"Candle body {body_ratio:.0%}"
            direction = "CALL"

        elif signal_dir == "PUT":
            # Bullish engulfing against PUT = danger
            if is_green and body > prev_body and prev_is_green:
                return LayerResult(name="candle_quality", score=0.0, direction="CALL",
                                   detail="⚠️ Bullish engulfing — reversal signal")
            # Strong wick rejection against PUT
            if lower_wick_pct > 0.40:
                return LayerResult(name="candle_quality", score=0.2, direction="PUT",
                                   detail=f"Lower wick rejection ({lower_wick_pct:.0%})")
            # Strong red candle
            if is_red and body_ratio > 0.70:
                score = 1.0
                detail = f"Strong bearish candle (body {body_ratio:.0%})"
            elif is_red and body_ratio > 0.50:
                score = 0.8
                detail = f"Solid bearish candle (body {body_ratio:.0%})"
            elif body_ratio < 0.30:
                score = 0.5
                detail = f"Doji / small body ({body_ratio:.0%}) — indecisive"
            else:
                score = 0.6
                detail = f"Candle body {body_ratio:.0%}"
            direction = "PUT"

        else:
            score = 0.5
            direction = "NEUTRAL"
            detail = f"Neutral candle (body {body_ratio:.0%})"

        return LayerResult(name="candle_quality", score=score, direction=direction, detail=detail)
