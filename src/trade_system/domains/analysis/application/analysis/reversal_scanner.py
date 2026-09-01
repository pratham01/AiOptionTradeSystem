"""
DailyReversalScanner — Quantitative Multi-Layer Trend Reversal Detector.

Identifies high-probability reversal setups across the F&O universe on the Daily timeframe
using institutional confluence techniques:
1. RSI Divergence (Regular Bullish & Bearish Divergence in extreme territory)
2. Candlestick Price Action & Forensic Absorption (Hammer, Shooting Star, Engulfing, Pin Bars)
3. Statistical Deviation & Mean Reversion Snapback (Bollinger Bands ±2.0σ & 20 EMA overstretch)
4. Volume Climax & Institutional Absorption (Volume expansion on rejection candle)
5. Liquidity Sweeps / Turtle Soup (Swept 20-day swing extremes and closed back inside)
6. Supertrend Rejection / Support Bounce
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping

LOGGER = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Smoothing RSI (matches TradingView)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


@dataclass
class DailyReversalSetup:
    """Dataclass holding a detected Daily Reversal setup."""
    symbol: str
    sector: str
    direction: str  # "CALL" (Bullish Reversal) or "PUT" (Bearish Reversal)
    probability: int  # 0 to 100%
    confidence: str  # "VERY HIGH" (>=80), "HIGH" (65-79), "MODERATE" (45-64)
    primary_pattern: str
    ltp: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    confluences: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    target_date: Optional[date] = None


class DailyReversalScanner:
    """
    Scans Daily OHLCV candles across all F&O universe stocks and indices
    to identify institutional mean-reversion and trend-reversal setups.
    """

    def __init__(self, min_probability: int = 45) -> None:
        self.min_probability = min_probability
        self.sector_map = get_sector_mapping()

    def scan(
        self,
        target_date: Optional[date] = None,
        min_probability: Optional[int] = None,
        symbols: Optional[List[str]] = None,
    ) -> List[DailyReversalSetup]:
        """
        Scan daily candles for high-probability reversal setups.
        """
        threshold = min_probability if min_probability is not None else self.min_probability
        target_d = target_date or date.today()

        engine = get_engine()
        with engine.connect() as conn:
            query = text("""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM ohlcv_daily 
                WHERE timestamp >= date(:t_date, '-250 days') AND timestamp <= :t_date
                ORDER BY symbol, timestamp ASC
            """)
            df_d = pd.read_sql(query, conn, params={"t_date": target_d.isoformat()})

        if df_d.empty:
            LOGGER.warning("No daily candle data found for reversal scan up to %s", target_d)
            return []

        df_d["timestamp"] = pd.to_datetime(df_d["timestamp"], format="mixed", errors="coerce")
        
        if symbols:
            df_d = df_d[df_d["symbol"].isin(symbols)]

        results: List[DailyReversalSetup] = []

        for sym, grp in df_d.groupby("symbol"):
            if len(grp) < 25:
                continue

            setup = self.evaluate_series(sym, grp.sort_values("timestamp").reset_index(drop=True), target_d)
            if setup and setup.probability >= threshold:
                results.append(setup)

        results.sort(key=lambda s: -s.probability)
        return results

    def evaluate_series(
        self,
        symbol: str,
        df: pd.DataFrame,
        target_date: Optional[date] = None,
    ) -> Optional[DailyReversalSetup]:
        """
        Evaluate a single symbol's daily OHLCV DataFrame for reversal signals.
        """
        if len(df) < 25:
            return None

        # Compute Technical Indicators (skip if precomputed)
        if "rsi" not in df.columns:
            df = df.copy()
            df["rsi"] = _compute_rsi(df["close"], 14)
            df["vol_sma20"] = df["volume"].rolling(20).mean()
            df["sma20"] = df["close"].rolling(20).mean()
            df["std20"] = df["close"].rolling(20).std()
            df["lower_bb"] = df["sma20"] - 2.0 * df["std20"]
            df["upper_bb"] = df["sma20"] + 2.0 * df["std20"]

            # EMAs (9, 20, 50, 200)
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["ema200"] = df["close"].ewm(span=200, adjust=False).mean() if len(df) >= 120 else np.nan

            # ATR 14
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        ltp = float(last["close"])
        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        vol = float(last["volume"])
        vol_sma = float(last["vol_sma20"]) if not pd.isna(last["vol_sma20"]) else 1.0
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        atr = float(last["atr"]) if not pd.isna(last["atr"]) and last["atr"] > 0 else max(h - l, 1.0)
        lower_bb = float(last["lower_bb"]) if not pd.isna(last["lower_bb"]) else ltp
        upper_bb = float(last["upper_bb"]) if not pd.isna(last["upper_bb"]) else ltp
        sma20 = float(last["sma20"]) if not pd.isna(last["sma20"]) else ltp
        ema9 = float(last["ema9"]) if not pd.isna(last["ema9"]) else ltp
        ema20 = float(last["ema20"]) if not pd.isna(last["ema20"]) else ltp
        ema50 = float(last["ema50"]) if not pd.isna(last["ema50"]) else ltp
        ema200 = float(last["ema200"]) if not pd.isna(last["ema200"]) else None

        c_range = max(h - l, 0.01)
        body = abs(ltp - o)
        lower_wick = min(o, ltp) - l
        upper_wick = h - max(o, ltp)

        # -------------------------------------------------------------
        # 1. Bullish Reversal Evaluation (Calls / Long)
        # -------------------------------------------------------------
        score_bull = 0
        confluences_bull: List[str] = []
        patterns_bull: List[str] = []

        # A. Candlestick Patterns
        is_hammer = (lower_wick >= 0.45 * c_range) and (body <= 0.40 * c_range) and (ltp >= prev["close"] or ltp > o)
        is_bull_engulf = (prev["close"] < prev["open"]) and (ltp > o) and (ltp >= prev["open"]) and (o <= prev["close"])
        is_piercing = (prev["close"] < prev["open"]) and (ltp > o) and (ltp >= prev["close"] + (prev["open"] - prev["close"]) * 0.5) and (o <= prev["close"])

        if is_hammer:
            score_bull += 25
            confluences_bull.append("Hammer / Rejection Pin Bar")
            patterns_bull.append("Hammer")
        elif is_bull_engulf:
            score_bull += 25
            confluences_bull.append("Bullish Engulfing")
            patterns_bull.append("Bullish Engulfing")
        elif is_piercing:
            score_bull += 20
            confluences_bull.append("Piercing Line Reversal")
            patterns_bull.append("Piercing Pattern")

        # B. RSI Oversold & Bullish Divergence
        if rsi <= 35:
            score_bull += 15
            confluences_bull.append(f"RSI Oversold ({rsi:.0f})")
        elif rsi <= 40:
            score_bull += 10
            confluences_bull.append(f"RSI Value Zone ({rsi:.0f})")

        prev_low_window = df.iloc[-22:-2]
        if len(prev_low_window) >= 5:
            min_p = prev_low_window["low"].min()
            min_rsi = prev_low_window["rsi"].min()
            if l <= min_p and rsi > (min_rsi + 2.5):
                score_bull += 25
                confluences_bull.append("Bullish RSI Divergence")
                patterns_bull.append("RSI Divergence")

        # C. Bollinger Band Snapback / Oversold Extension
        if l <= lower_bb and ltp >= lower_bb:
            score_bull += 20
            confluences_bull.append("Lower Bollinger Band Snapback")
        elif ltp < lower_bb:
            score_bull += 15
            confluences_bull.append("Outside Lower Bollinger Band")

        # D. Volume Surge / Climax
        surge = vol / vol_sma if vol_sma > 0 else 1.0
        if surge >= 1.3:
            score_bull += 15
            confluences_bull.append(f"Volume Surge ({surge:.1f}x 20SMA)")

        # E. 20-Day Low Liquidity Sweep (Turtle Soup)
        lowest_20 = df.iloc[-21:-1]["low"].min() if len(df) >= 22 else l
        if l < lowest_20 and ltp > lowest_20:
            score_bull += 15
            confluences_bull.append("20D Low Liquidity Sweep")
            patterns_bull.append("Liquidity Sweep")

        # F. SMC Fair Value Gap (FVG / BISI) Retest & 50% Consequent Encroachment (CE)
        fvgs_bull = []
        n_bars = len(df)
        for i in range(max(2, n_bars - 35), n_bars - 1):
            c1_h = float(df.loc[i - 2, "high"])
            c3_l = float(df.loc[i, "low"])
            if c3_l > c1_h:
                fvgs_bull.append({"top": c3_l, "bottom": c1_h, "ce": (c3_l + c1_h) / 2.0})

        for fvg in reversed(fvgs_bull[-4:]):
            if l <= fvg["top"] and ltp >= fvg["bottom"]:
                tested_ce = (l <= fvg["ce"])
                ce_tag = " (50% CE Tapped)" if tested_ce else ""
                score_bull += 20
                confluences_bull.append(f"Bullish FVG Retest ₹{fvg['bottom']:.1f}-₹{fvg['top']:.1f}{ce_tag}")
                patterns_bull.append("FVG Retest")
                break

        # G. SMC Bullish Order Block (OB) Tap
        for i in range(max(1, n_bars - 25), n_bars - 2):
            # Down candle followed by displacement rally
            if float(df.loc[i, "close"]) < float(df.loc[i, "open"]):
                # Next 2 candles made a sharp move up
                if float(df.loc[i + 1, "close"]) > float(df.loc[i, "high"]):
                    ob_high = float(df.loc[i, "high"])
                    ob_low = float(df.loc[i, "low"])
                    if l <= ob_high and ltp >= ob_low:
                        score_bull += 20
                        confluences_bull.append(f"Bullish Order Block (OB ₹{ob_low:.1f}-₹{ob_high:.1f})")
                        patterns_bull.append("Order Block (OB)")
                        break

        # H. Change of Character (CHoCH)
        minor_high_5 = df.iloc[-6:-1]["high"].max() if len(df) >= 6 else h
        if ltp > minor_high_5 and o <= minor_high_5:
            score_bull += 15
            confluences_bull.append("Bullish CHoCH (Structure Shift)")
            patterns_bull.append("CHoCH")

        # I. EMA Dynamic Support & Trend Alignments (20, 50, 200 EMA)
        if ema200 is not None and abs(l - ema200) / ema200 <= 0.012 and ltp >= ema200:
            score_bull += 20
            confluences_bull.append(f"200 EMA Macro Baseline Bounce (₹{ema200:.1f})")
            patterns_bull.append("200 EMA Bounce")
        elif abs(l - ema50) / ema50 <= 0.010 and ltp >= ema50:
            score_bull += 15
            confluences_bull.append(f"50 EMA Dynamic Trend Bounce (₹{ema50:.1f})")
            patterns_bull.append("50 EMA Bounce")

        # 20 EMA Overstretch (Rubber Band)
        if (ema20 - l) >= 2.0 * atr and ltp < ema20:
            score_bull += 15
            confluences_bull.append(f"20 EMA Mean Reversion Stretch (EMA: ₹{ema20:.1f})")
            patterns_bull.append("20 EMA Snapback")

        # 9 EMA Fast Momentum Regain
        if ltp > ema9 and float(prev["close"]) <= float(prev["ema9"]):
            score_bull += 10
            confluences_bull.append("9 EMA Fast Momentum Regain")

        # -------------------------------------------------------------
        # 2. Bearish Reversal Evaluation (Puts / Short)
        # -------------------------------------------------------------
        score_bear = 0
        confluences_bear: List[str] = []
        patterns_bear: List[str] = []

        # A. Candlestick Patterns
        is_shooting_star = (upper_wick >= 0.45 * c_range) and (body <= 0.40 * c_range) and (ltp <= prev["close"] or ltp < o)
        is_bear_engulf = (prev["close"] > prev["open"]) and (ltp < o) and (ltp <= prev["open"]) and (o >= prev["close"])
        is_dark_cloud = (prev["close"] > prev["open"]) and (ltp < o) and (ltp <= prev["close"] - (prev["close"] - prev["open"]) * 0.5) and (o >= prev["close"])

        if is_shooting_star:
            score_bear += 25
            confluences_bear.append("Shooting Star / Upper Rejection")
            patterns_bear.append("Shooting Star")
        elif is_bear_engulf:
            score_bear += 25
            confluences_bear.append("Bearish Engulfing")
            patterns_bear.append("Bearish Engulfing")
        elif is_dark_cloud:
            score_bear += 20
            confluences_bear.append("Dark Cloud Cover")
            patterns_bear.append("Dark Cloud Cover")

        # B. RSI Overbought & Bearish Divergence
        if rsi >= 68:
            score_bear += 15
            confluences_bear.append(f"RSI Overbought ({rsi:.0f})")
        elif rsi >= 62:
            score_bear += 10
            confluences_bear.append(f"RSI Overbought Zone ({rsi:.0f})")

        prev_high_window = df.iloc[-22:-2]
        if len(prev_high_window) >= 5:
            max_p = prev_high_window["high"].max()
            max_rsi = prev_high_window["rsi"].max()
            if h >= max_p and rsi < (max_rsi - 2.5):
                score_bear += 25
                confluences_bear.append("Bearish RSI Divergence")
                patterns_bear.append("RSI Divergence")

        # C. Bollinger Band Snapback / Overbought Extension
        if h >= upper_bb and ltp <= upper_bb:
            score_bear += 20
            confluences_bear.append("Upper Bollinger Band Snapback")
        elif ltp > upper_bb:
            score_bear += 15
            confluences_bear.append("Outside Upper Bollinger Band")

        # D. Volume Surge / Climax
        if surge >= 1.3:
            score_bear += 15
            confluences_bear.append(f"Volume Surge ({surge:.1f}x 20SMA)")

        # E. 20-Day High Liquidity Sweep
        highest_20 = df.iloc[-21:-1]["high"].max() if len(df) >= 22 else h
        if h > highest_20 and ltp < highest_20:
            score_bear += 15
            confluences_bear.append("20D High Liquidity Sweep")
            patterns_bear.append("Liquidity Sweep")

        # F. SMC Bearish Fair Value Gap (FVG / SIBI) Retest & 50% CE Tap
        fvgs_bear = []
        for i in range(max(2, n_bars - 35), n_bars - 1):
            c1_l = float(df.loc[i - 2, "low"])
            c3_h = float(df.loc[i, "high"])
            if c1_l > c3_h:
                fvgs_bear.append({"top": c1_l, "bottom": c3_h, "ce": (c1_l + c3_h) / 2.0})

        for fvg in reversed(fvgs_bear[-4:]):
            if h >= fvg["bottom"] and ltp <= fvg["top"]:
                tested_ce = (h >= fvg["ce"])
                ce_tag = " (50% CE Tapped)" if tested_ce else ""
                score_bear += 20
                confluences_bear.append(f"Bearish FVG Retest ₹{fvg['bottom']:.1f}-₹{fvg['top']:.1f}{ce_tag}")
                patterns_bear.append("FVG Retest")
                break

        # G. SMC Bearish Order Block (OB) Tap
        for i in range(max(1, n_bars - 25), n_bars - 2):
            # Up candle followed by displacement drop
            if float(df.loc[i, "close"]) > float(df.loc[i, "open"]):
                if float(df.loc[i + 1, "close"]) < float(df.loc[i, "low"]):
                    ob_high = float(df.loc[i, "high"])
                    ob_low = float(df.loc[i, "low"])
                    if h >= ob_low and ltp <= ob_high:
                        score_bear += 20
                        confluences_bear.append(f"Bearish Order Block (OB ₹{ob_low:.1f}-₹{ob_high:.1f})")
                        patterns_bear.append("Order Block (OB)")
                        break

        # H. Bearish Change of Character (CHoCH)
        minor_low_5 = df.iloc[-6:-1]["low"].min() if len(df) >= 6 else l
        if ltp < minor_low_5 and o >= minor_low_5:
            score_bear += 15
            confluences_bear.append("Bearish CHoCH (Structure Shift)")
            patterns_bear.append("CHoCH")

        # I. EMA Dynamic Resistance & Trend Alignments (20, 50, 200 EMA)
        if ema200 is not None and abs(h - ema200) / ema200 <= 0.012 and ltp <= ema200:
            score_bear += 20
            confluences_bear.append(f"200 EMA Macro Resistance Rejection (₹{ema200:.1f})")
            patterns_bear.append("200 EMA Rejection")
        elif abs(h - ema50) / ema50 <= 0.010 and ltp <= ema50:
            score_bear += 15
            confluences_bear.append(f"50 EMA Dynamic Trend Rejection (₹{ema50:.1f})")
            patterns_bear.append("50 EMA Rejection")

        # 20 EMA Overstretch (Rubber Band)
        if (h - ema20) >= 2.0 * atr and ltp > ema20:
            score_bear += 15
            confluences_bear.append(f"20 EMA Mean Reversion Stretch (EMA: ₹{ema20:.1f})")
            patterns_bear.append("20 EMA Snapback")

        # 9 EMA Fast Momentum Loss
        if ltp < ema9 and float(prev["close"]) >= float(prev["ema9"]):
            score_bear += 10
            confluences_bear.append("9 EMA Fast Momentum Breakdown")

        # Select stronger direction
        if score_bull >= score_bear and score_bull >= 40:
            prob = min(score_bull, 98)
            conf = "VERY HIGH" if prob >= 80 else ("HIGH" if prob >= 65 else "MODERATE")
            sl = round(l - atr * 0.35, 2)
            risk = max(ltp - sl, 0.5)
            t1 = round(ltp + 1.5 * risk, 2)
            t2 = round(ltp + 2.5 * risk, 2)
            rr = round((t1 - ltp) / risk, 2)
            primary = " + ".join(patterns_bull) if patterns_bull else "Mean Reversion Snapback"

            return DailyReversalSetup(
                symbol=symbol,
                sector=self.sector_map.get(symbol, "OTHER"),
                direction="CALL",
                probability=prob,
                confidence=conf,
                primary_pattern=primary,
                ltp=ltp,
                stop_loss=sl,
                target_1=t1,
                target_2=t2,
                risk_reward=rr,
                confluences=confluences_bull,
                metrics={
                    "rsi": round(rsi, 1),
                    "vol_surge": round(surge, 2),
                    "atr": round(atr, 2),
                    "lower_bb": round(lower_bb, 2),
                    "sma20": round(sma20, 2)
                },
                target_date=target_date
            )

        elif score_bear > score_bull and score_bear >= 40:
            prob = min(score_bear, 98)
            conf = "VERY HIGH" if prob >= 80 else ("HIGH" if prob >= 65 else "MODERATE")
            sl = round(h + atr * 0.35, 2)
            risk = max(sl - ltp, 0.5)
            t1 = round(ltp - 1.5 * risk, 2)
            t2 = round(ltp - 2.5 * risk, 2)
            rr = round((ltp - t1) / risk, 2)
            primary = " + ".join(patterns_bear) if patterns_bear else "Mean Reversion Breakdown"

            return DailyReversalSetup(
                symbol=symbol,
                sector=self.sector_map.get(symbol, "OTHER"),
                direction="PUT",
                probability=prob,
                confidence=conf,
                primary_pattern=primary,
                ltp=ltp,
                stop_loss=sl,
                target_1=t1,
                target_2=t2,
                risk_reward=rr,
                confluences=confluences_bear,
                metrics={
                    "rsi": round(rsi, 1),
                    "vol_surge": round(surge, 2),
                    "atr": round(atr, 2),
                    "upper_bb": round(upper_bb, 2),
                    "sma20": round(sma20, 2)
                },
                target_date=target_date
            )

        return None
