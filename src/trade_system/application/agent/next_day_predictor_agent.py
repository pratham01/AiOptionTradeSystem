"""
NextDayPredictorAgent — Generates a PRIORITY WATCHLIST for the next trading day.

Backtest-validated approach:
  - Uses DAILY candle patterns (engulfing, hammer, compression breakout, vol-weighted close, ST flip)
  - Requires dual confirmation (2+ patterns aligning)
  - Market regime gate (Nifty daily supertrend)
  - Sector performance filter
  - ATR-based dynamic SL/TP zones
  - Outputs a WATCHLIST (not trade signals) for the intraday agent to confirm with real-time data

NOTE: This agent was redesigned based on 3 backtest iterations showing that single-day
prediction from 5m data has near-zero edge. Daily candle patterns with dual confirmation
and regime gating showed the best risk-adjusted results.
"""
from __future__ import annotations

import logging
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Any, List, Dict, Optional

from trade_system.core import MarketContext, TradeSuggestion, TradeDirection, TradeHorizon, OptionParams
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.repository import get_market_data, log_agent_thought
from sqlalchemy.orm import Session
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

# Supertrend for daily timeframe (10-period, 2x multiplier — less noisy than 7,3)
_ST_DAILY = SupertrendIndicator(period=10, multiplier=2)

# Pattern quality weights (from backtest: higher = more edge)
PATTERN_WEIGHT = {
    "Bearish Engulfing": 0.18,
    "Hammer": 0.15,
    "Daily ST Flip (Bull)": 0.20,
    "Daily ST Flip (Bear)": 0.20,
    "Bullish Engulfing": 0.10,
    "Shooting Star": 0.15,
    "Compression Breakout": 0.12,
    "Compression Breakdown": 0.12,
    "Vol-Weighted Close (Bull)": 0.10,
    "Vol-Weighted Close (Bear)": 0.10,
    "Momentum Surge (Bull)": 0.25,
    "Momentum Surge (Bear)": 0.25,
    "Monthly Breakout": 0.30,
    "Monthly Breakdown": 0.30,
    "Weekly Breakout": 0.20,
    "Weekly Breakdown": 0.20,
    "Daily ST Touch (Bull)": 0.15,
    "Daily ST Touch (Bear)": 0.15,
}


@dataclass
class WatchlistItem:
    """A stock flagged for priority monitoring during the next trading session."""
    symbol: str
    direction: str           # "CALL" or "PUT"
    confidence: float        # 0.0 – 1.0
    patterns: List[str]      # e.g. ["Bearish Engulfing", "Daily ST Flip (Bear)"]
    pattern_count: int
    close: float             # EOD close price
    atr: float               # Daily ATR value
    atr_pct: float           # ATR as % of close
    suggested_sl_pct: float  # 1.2 × ATR%
    suggested_tp_pct: float  # 2.5 × ATR%
    rsi: float
    daily_trend: int         # 1 (bull) or -1 (bear)
    sector: str
    sector_perf: float       # Sector avg % change
    narrative: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return asdict(self)


class NextDayPredictorAgent:
    """
    Scans the F&O universe using DAILY candle patterns to generate
    a priority watchlist for the next trading session.
    
    Output is a List[WatchlistItem], NOT TradeSuggestion.
    The intraday FoStockSuggesterAgent consumes this watchlist.
    """

    def __init__(self, broker: Any = None):
        self.broker = broker
        self.engine = get_engine()

    def _thought(self, message: str, symbol: str | None = None, action: str | None = None):
        LOGGER.info(f"[NextDayPredictorAgent] {message}")
        try:
            with Session(self.engine) as session:
                log_agent_thought(session, "NextDayPredictorAgent", message, symbol, action)
        except: pass

    # ─────────────── DATA LOADERS ───────────────

    def _load_daily(self, symbol: str, end_date: str = None, bars: int = 50) -> pd.DataFrame:
        """Load N daily bars for a symbol."""
        with self.engine.connect() as conn:
            params = {"sym": symbol, "n": bars}
            if end_date:
                q = text(
                    "SELECT timestamp, open, high, low, close, volume FROM ohlcv_daily "
                    "WHERE symbol = :sym AND date(timestamp) <= :dt "
                    "ORDER BY timestamp DESC LIMIT :n"
                )
                params["dt"] = end_date
            else:
                q = text(
                    "SELECT timestamp, open, high, low, close, volume FROM ohlcv_daily "
                    "WHERE symbol = :sym ORDER BY timestamp DESC LIMIT :n"
                )
            rows = conn.execute(q, params).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.replace(r"\.\d+", "", regex=True), errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)

    def _get_nifty_trend(self, end_date: str = None) -> int:
        """Nifty50 daily supertrend direction. 1=bull, -1=bear, 0=unknown."""
        df = self._load_daily("NSE:NIFTY50-INDEX", end_date, bars=30)
        if df.empty or len(df) < 15:
            return 0
        try:
            df = _ST_DAILY.calculate(df)
            return int(df.iloc[-1]["supertrend_direction"])
        except:
            return 0

    def _compute_sector_perf(self, symbols: List[str], sector_map: dict, end_date: str = None) -> dict:
        """Compute sector average daily change."""
        from collections import defaultdict
        sector_changes = defaultdict(list)
        for symbol in symbols:
            sector = sector_map.get(symbol, "UNKNOWN")
            if sector == "UNKNOWN":
                continue
            df = self._load_daily(symbol, end_date, bars=3)
            if len(df) < 2:
                continue
            prev_close = df.iloc[-2]["close"]
            curr_close = df.iloc[-1]["close"]
            pct = ((curr_close - prev_close) / prev_close) * 100
            sector_changes[sector].append(pct)
        return {s: np.mean(v) for s, v in sector_changes.items()}

    def _load_sector_map(self) -> dict:
        from pathlib import Path
        import json
        config_path = Path("config/fo_universe.json")
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    # ─────────────── TECHNICAL HELPERS ───────────────

    @staticmethod
    def _calc_rsi(prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        return 100 - (100 / (1 + avg_gain / avg_loss))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        tr0 = abs(df["high"] - df["low"])
        tr1 = abs(df["high"] - df["close"].shift(1))
        tr2 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return atr if not np.isnan(atr) else tr.mean()

    # ─────────────── PATTERN DETECTORS ───────────────

    @staticmethod
    def _detect_engulfing(df: pd.DataFrame) -> dict | None:
        if len(df) < 2:
            return None
        prev, curr = df.iloc[-2], df.iloc[-1]
        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])
        if curr_body < prev_body * 0.5:
            return None
        # Bullish engulfing
        if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
            if curr["close"] > prev["open"] and curr["open"] <= prev["close"]:
                return {"pattern": "Bullish Engulfing", "direction": "CALL", "strength": curr_body / prev_body}
        # Bearish engulfing
        if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
            if curr["close"] < prev["open"] and curr["open"] >= prev["close"]:
                return {"pattern": "Bearish Engulfing", "direction": "PUT", "strength": curr_body / prev_body}
        return None

    @staticmethod
    def _detect_pin_bar(df: pd.DataFrame) -> dict | None:
        if len(df) < 3:
            return None
        curr = df.iloc[-1]
        body = abs(curr["close"] - curr["open"])
        full_range = curr["high"] - curr["low"]
        if full_range == 0:
            return None
        upper_wick = curr["high"] - max(curr["close"], curr["open"])
        lower_wick = min(curr["close"], curr["open"]) - curr["low"]
        body_pct = body / full_range
        # Hammer
        if lower_wick > full_range * 0.60 and body_pct < 0.30 and upper_wick < full_range * 0.15:
            recent_low = df.tail(10)["low"].min()
            if curr["low"] <= recent_low * 1.005:
                return {"pattern": "Hammer", "direction": "CALL", "strength": lower_wick / full_range}
        # Shooting Star
        if upper_wick > full_range * 0.60 and body_pct < 0.30 and lower_wick < full_range * 0.15:
            recent_high = df.tail(10)["high"].max()
            if curr["high"] >= recent_high * 0.995:
                return {"pattern": "Shooting Star", "direction": "PUT", "strength": upper_wick / full_range}
        return None

    @staticmethod
    def _detect_compression_breakout(df: pd.DataFrame) -> dict | None:
        if len(df) < 5:
            return None
        curr = df.iloc[-1]
        prev_3 = df.iloc[-4:-1]
        range_high = prev_3["high"].max()
        range_low = prev_3["low"].min()
        range_pct = ((range_high - range_low) / range_low) * 100
        if range_pct > 4.0:
            return None
        vol_ratio = curr["volume"] / prev_3["volume"].mean() if prev_3["volume"].mean() > 0 else 0
        if vol_ratio < 1.5:
            return None
        if curr["close"] > range_high and curr["close"] > curr["open"]:
            return {"pattern": "Compression Breakout", "direction": "CALL", "strength": vol_ratio}
        if curr["close"] < range_low and curr["close"] < curr["open"]:
            return {"pattern": "Compression Breakdown", "direction": "PUT", "strength": vol_ratio}
        return None

    @staticmethod
    def _detect_vol_weighted_close(df: pd.DataFrame) -> dict | None:
        if len(df) < 20:
            return None
        curr = df.iloc[-1]
        full_range = curr["high"] - curr["low"]
        if full_range == 0:
            return None
        vol_sma = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = curr["volume"] / vol_sma if vol_sma > 0 else 0
        if vol_ratio < 2.0:
            return None
        close_position = (curr["close"] - curr["low"]) / full_range
        if close_position >= 0.75 and curr["close"] > curr["open"]:
            return {"pattern": "Vol-Weighted Close (Bull)", "direction": "CALL", "strength": vol_ratio}
        if close_position <= 0.25 and curr["close"] < curr["open"]:
            return {"pattern": "Vol-Weighted Close (Bear)", "direction": "PUT", "strength": vol_ratio}
        return None

    @staticmethod
    def _detect_momentum_surge(df: pd.DataFrame) -> dict | None:
        if len(df) < 20:
            return None
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        change_pct = ((curr["close"] - prev["close"]) / prev["close"]) * 100
        full_range = curr["high"] - curr["low"]
        if full_range == 0:
            return None
            
        vol_sma = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = curr["volume"] / vol_sma if vol_sma > 0 else 0
        
        if change_pct > 1.5 and vol_ratio > 1.2:
            close_pos = (curr["close"] - curr["low"]) / full_range
            if close_pos > 0.8:
                return {"pattern": "Momentum Surge (Bull)", "direction": "CALL", "strength": vol_ratio}
                
        if change_pct < -1.5 and vol_ratio > 1.2:
            close_pos = (curr["close"] - curr["low"]) / full_range
            if close_pos < 0.2:
                return {"pattern": "Momentum Surge (Bear)", "direction": "PUT", "strength": vol_ratio}
                
        return None

    @staticmethod
    def _detect_mtf_breakout(df: pd.DataFrame) -> dict | None:
        if len(df) < 25:
            return None
        curr = df.iloc[-1]
        prev_20 = df.iloc[-21:-1]
        prev_5 = df.iloc[-6:-1]
        
        month_high = prev_20["high"].max()
        month_low = prev_20["low"].min()
        week_high = prev_5["high"].max()
        week_low = prev_5["low"].min()
        
        vol_sma = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = curr["volume"] / vol_sma if vol_sma > 0 else 0
        
        if vol_ratio < 1.2:
            return None
            
        if curr["close"] > month_high:
            return {"pattern": "Monthly Breakout", "direction": "CALL", "strength": vol_ratio}
        if curr["close"] < month_low:
            return {"pattern": "Monthly Breakdown", "direction": "PUT", "strength": vol_ratio}
            
        if curr["close"] > week_high:
            return {"pattern": "Weekly Breakout", "direction": "CALL", "strength": vol_ratio}
        if curr["close"] < week_low:
            return {"pattern": "Weekly Breakdown", "direction": "PUT", "strength": vol_ratio}
            
        return None

    @staticmethod
    def _detect_supertrend_touch(df: pd.DataFrame) -> dict | None:
        if len(df) < 15:
            return None
        try:
            df_st = _ST_DAILY.calculate(df.copy())
        except:
            return None
        if "supertrend" not in df_st.columns:
            return None
            
        curr = df_st.iloc[-1]
        st_val = curr["supertrend"]
        st_dir = curr["supertrend_direction"]
        
        full_range = curr["high"] - curr["low"]
        if full_range == 0:
            return None
            
        if st_dir == 1:
            if curr["low"] <= st_val * 1.005 and curr["close"] > st_val:
                close_pos = (curr["close"] - curr["low"]) / full_range
                if close_pos > 0.5:
                    return {"pattern": "Daily ST Touch (Bull)", "direction": "CALL", "strength": 1.0}
                    
        if st_dir == -1:
            if curr["high"] >= st_val * 0.995 and curr["close"] < st_val:
                close_pos = (curr["close"] - curr["low"]) / full_range
                if close_pos < 0.5:
                    return {"pattern": "Daily ST Touch (Bear)", "direction": "PUT", "strength": 1.0}
                    
        return None


    @staticmethod
    def _detect_st_flip(df: pd.DataFrame) -> dict | None:
        if len(df) < 15:
            return None
        try:
            df_st = _ST_DAILY.calculate(df.copy())
        except:
            return None
        if "supertrend_signal" not in df_st.columns:
            return None
        latest = df_st.iloc[-1]
        if latest["supertrend_signal"] == 1:
            return {"pattern": "Daily ST Flip (Bull)", "direction": "CALL", "strength": 1.5}
        elif latest["supertrend_signal"] == -1:
            return {"pattern": "Daily ST Flip (Bear)", "direction": "PUT", "strength": 1.5}
        return None

    # ─────────────── MAIN API ───────────────

    async def predict_next_day_setups(
        self, symbols: List[str], market_context: MarketContext
    ) -> List[WatchlistItem]:
        """
        Analyze daily data for F&O symbols and return a WATCHLIST
        of stocks to monitor during the next trading session.
        """
        self._thought(f"Scanning {len(symbols)} symbols for next-day watchlist...", action="SCANNING")

        sector_map = self._load_sector_map()

        # Determine end date from market_context or today
        end_date = None
        if hasattr(market_context, "timestamp") and market_context.timestamp:
            ts = market_context.timestamp
            if isinstance(ts, str):
                end_date = ts[:10]
            elif hasattr(ts, "strftime"):
                end_date = ts.strftime("%Y-%m-%d")

        # Market regime gate
        nifty_trend = self._get_nifty_trend(end_date)
        regime = "BULL" if nifty_trend == 1 else "BEAR" if nifty_trend == -1 else "NEUTRAL"
        self._thought(f"Market regime: {regime} (Nifty daily ST direction={nifty_trend})", action="REGIME")

        # Sector performance
        sector_perf = self._compute_sector_perf(symbols, sector_map, end_date)

        watchlist: List[WatchlistItem] = []

        for symbol in symbols:
            try:
                item = self._analyze_symbol(symbol, end_date, nifty_trend, sector_perf, sector_map)
                if item:
                    watchlist.append(item)
            except Exception as e:
                LOGGER.debug(f"Error analyzing {symbol}: {e}")

        # Sort by confidence descending
        watchlist.sort(key=lambda x: x.confidence, reverse=True)
        top_picks = watchlist[:3]  # Max 3 picks

        if top_picks:
            syms = ", ".join(f"{w.symbol.split(':')[-1].replace('-EQ','')} ({w.direction})" for w in top_picks)
            self._thought(f"Watchlist generated: {syms}", action="APPROVED")
        else:
            self._thought("No high-conviction setups found for next-day watchlist.", action="IDLE")

        return top_picks

    def _analyze_symbol(
        self, symbol: str, end_date: str | None,
        nifty_trend: int, sector_perf: dict, sector_map: dict
    ) -> WatchlistItem | None:
        """Analyze a single symbol and return WatchlistItem or None."""

        df = self._load_daily(symbol, end_date, bars=50)
        if df.empty or len(df) < 20:
            return None

        # Ensure latest bar is on the target date
        latest = df.iloc[-1]
        if end_date:
            target = pd.to_datetime(end_date).date()
            if latest["timestamp"].date() != target:
                return None

        # ── Detect all patterns ──
        patterns = []
        for detector in [
            self._detect_engulfing, self._detect_pin_bar,
            self._detect_compression_breakout, self._detect_vol_weighted_close,
            self._detect_momentum_surge, self._detect_mtf_breakout,
            self._detect_supertrend_touch, self._detect_st_flip,
        ]:
            result = detector(df)
            if result:
                patterns.append(result)

        if not patterns:
            return None

        # ── Direction consensus ──
        call_votes = sum(1 for p in patterns if p["direction"] == "CALL")
        put_votes = sum(1 for p in patterns if p["direction"] == "PUT")
        if call_votes == put_votes:
            return None  # Conflicting signals

        direction = "CALL" if call_votes > put_votes else "PUT"
        aligned = [p for p in patterns if p["direction"] == direction]

        # ── EXTREME MOMENTUM BYPASS ──
        has_extreme_momentum = any(
            p["pattern"] in ["Monthly Breakout", "Monthly Breakdown", "Momentum Surge (Bull)", "Momentum Surge (Bear)"]
            for p in aligned
        )

        # ── DUAL CONFIRMATION: require 2+ aligning patterns OR extreme momentum ──
        if len(aligned) < 2 and not has_extreme_momentum:
            return None

        # ── Market regime gate (bypassed for extreme momentum) ──
        if not has_extreme_momentum:
            if direction == "CALL" and nifty_trend == -1:
                return None
            if direction == "PUT" and nifty_trend == 1:
                return None

        # ── Daily supertrend alignment ──
        try:
            df_st = _ST_DAILY.calculate(df.copy())
            daily_trend = int(df_st.iloc[-1]["supertrend_direction"])
        except:
            daily_trend = 0

        if daily_trend != 0:
            if direction == "CALL" and daily_trend == -1:
                return None
            if direction == "PUT" and daily_trend == 1:
                return None

        # ── RSI filter ──
        rsi = self._calc_rsi(df["close"])
        if direction == "CALL" and rsi > 75:
            return None
        if direction == "PUT" and rsi < 25:
            return None

        # ── Sector filter ──
        sector = sector_map.get(symbol, "UNKNOWN")
        sec_perf = sector_perf.get(sector, 0.0)
        if direction == "CALL" and sec_perf < -1.0:
            return None
        if direction == "PUT" and sec_perf > 1.0:
            return None

        # ── ATR-based SL/TP ──
        atr = self._calc_atr(df)
        atr_pct = (atr / latest["close"]) * 100 if latest["close"] > 0 else 2.0
        sl_pct = round(atr_pct * 1.2, 3)
        tp_pct = round(atr_pct * 2.5, 3)

        # ── Multi-factor confidence ──
        conf = 0.35
        for p in aligned:
            conf += PATTERN_WEIGHT.get(p["pattern"], 0.05)
        if len(aligned) >= 2:
            conf += 0.10
        if (direction == "CALL" and daily_trend == 1) or (direction == "PUT" and daily_trend == -1):
            conf += 0.10
        if direction == "CALL" and 45 <= rsi <= 65:
            conf += 0.05
        elif direction == "PUT" and 35 <= rsi <= 55:
            conf += 0.05
        if (direction == "CALL" and sec_perf > 0.5) or (direction == "PUT" and sec_perf < -0.5):
            conf += 0.05
        conf = min(0.95, conf)

        # ── Only keep medium-high confidence (backtest: [0.70, 0.80) was the profitable tier) ──
        if conf < 0.60:
            return None

        pattern_names = [p["pattern"] for p in aligned]
        narrative = (
            f"Next-day watchlist: {', '.join(pattern_names)} confluence. "
            f"Daily trend {'aligned' if daily_trend == (1 if direction=='CALL' else -1) else 'neutral'}. "
            f"RSI={rsi:.0f}, Sector {sector} ({sec_perf:+.1f}%). "
            f"ATR-based zones: SL {sl_pct:.1f}%, TP {tp_pct:.1f}%."
        )

        return WatchlistItem(
            symbol=symbol,
            direction=direction,
            confidence=round(conf, 3),
            patterns=pattern_names,
            pattern_count=len(aligned),
            close=latest["close"],
            atr=round(atr, 2),
            atr_pct=round(atr_pct, 3),
            suggested_sl_pct=sl_pct,
            suggested_tp_pct=tp_pct,
            rsi=round(rsi, 1),
            daily_trend=daily_trend,
            sector=sector,
            sector_perf=round(sec_perf, 2),
            narrative=narrative,
        )

    # ─────────── BACKWARD COMPATIBILITY ───────────
    # The PostMarketImproverAgent calls this and expects List[TradeSuggestion].
    # Convert watchlist items to TradeSuggestion for downstream compatibility.

    async def predict_as_suggestions(
        self, symbols: List[str], market_context: MarketContext
    ) -> List[TradeSuggestion]:
        """Convert watchlist to TradeSuggestion format for backward compat."""
        watchlist = await self.predict_next_day_setups(symbols, market_context)
        suggestions = []
        for item in watchlist:
            direction = TradeDirection.CALL if item.direction == "CALL" else TradeDirection.PUT
            suggestions.append(TradeSuggestion(
                id=str(uuid.uuid4()),
                symbol=item.symbol,
                timestamp=datetime.now(),
                direction=direction,
                horizon=TradeHorizon.SWING,
                entry_zone_low=item.close * (1 - item.atr_pct / 200),
                entry_zone_high=item.close * (1 + item.atr_pct / 200),
                target=item.close * (1 + item.suggested_tp_pct / 100) if direction == TradeDirection.CALL else item.close * (1 - item.suggested_tp_pct / 100),
                stop_loss=item.close * (1 - item.suggested_sl_pct / 100) if direction == TradeDirection.CALL else item.close * (1 + item.suggested_sl_pct / 100),
                option_params=OptionParams(direction=direction),
                confidence=item.confidence,
                narrative=item.narrative,
                setup_features=None,
                tags=["next_day_watchlist"] + item.patterns,
            ))
        return suggestions
