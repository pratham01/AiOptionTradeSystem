"""
StockIntelligenceEnricher — Shadow Mode Multi-Layer Signal Engine.

Computes 8 independent signal layers per stock entirely from existing
15-minute and Daily OHLCV data in the database. No additional broker
API calls are required (except for the lightweight news scraper).

Designed to run in shadow mode: outputs are written to JSON files for
a 1-week validation period before being wired into the main dashboard.

Layers:
  1. Multi-TF Supertrend Alignment (15m + Daily)
  2. EMA Stack Order (9/21/50 on Daily)
  3. ADX Trend Strength (Daily)
  4. MACD Momentum (15m)
  5. RSI Divergence (15m)
  6. OBV Divergence (15m)
  7. Price Structure — HH/HL/LH/LL (Daily)
  8. News Catalyst (web scraper, cached)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.strategy.application.indicators.supertrend import SupertrendIndicator
from trade_system.domains.strategy.application.indicators.obv import OBVIndicator

LOGGER = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class LayerSignal:
    """Result from a single signal layer."""
    name: str
    direction: str          # "BULL", "BEAR", "NEUTRAL"
    strength: float         # 0.0 – 1.0
    label: str              # Human-readable short label
    detail: str = ""        # Longer explanation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StockIntelligence:
    """Complete intelligence profile for a single stock."""
    symbol: str
    ltp: float
    pchange: float
    timestamp: str
    layers: List[LayerSignal] = field(default_factory=list)
    composite_score: float = 0.0
    composite_verdict: str = "NO_EDGE"
    composite_direction: str = "NEUTRAL"
    bull_count: int = 0
    bear_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = [l.to_dict() for l in self.layers]
        return d


# ── Main Engine ────────────────────────────────────────────────────────────────

class StockIntelligenceEnricher:
    """
    Standalone engine that computes 8 signal layers per stock from DB data.
    Zero broker API calls. All computation is local.
    """

    def __init__(self) -> None:
        self.engine = get_engine()
        self._news_cache: dict | None = None
        self._news_cache_time: datetime | None = None

    def enrich_symbols(
        self,
        symbols: List[str],
        target_date: date | None = None,
    ) -> List[StockIntelligence]:
        """
        Main entry point. Enriches a list of symbols with all 8 signal layers.
        Returns a list of StockIntelligence objects.
        """
        target_date = target_date or date.today()
        LOGGER.info(f"Shadow Intelligence: enriching {len(symbols)} symbols for {target_date}")

        # Batch-fetch all data from DB (one query per table, not per symbol)
        df_15m = self._fetch_15m_batch(symbols, target_date)
        df_daily = self._fetch_daily_batch(symbols, target_date)

        # Fetch news headlines (cached, cheap)
        news_map = self._fetch_news_for_symbols(symbols)

        results = []
        for sym in symbols:
            try:
                intel = self._enrich_single(sym, df_15m, df_daily, news_map, target_date)
                results.append(intel)
            except Exception as e:
                LOGGER.warning(f"Failed to enrich {sym}: {e}")
                results.append(StockIntelligence(
                    symbol=sym, ltp=0.0, pchange=0.0,
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    composite_verdict="ERROR",
                ))

        LOGGER.info(f"Shadow Intelligence: enriched {len(results)} symbols successfully")
        return results

    def _enrich_single(
        self,
        symbol: str,
        df_15m_all: pd.DataFrame,
        df_daily_all: pd.DataFrame,
        news_map: Dict[str, str],
        target_date: date,
    ) -> StockIntelligence:
        """Compute all 8 layers for a single symbol."""
        # Filter data for this symbol
        sym_15m = df_15m_all[df_15m_all["symbol"] == symbol].copy()
        sym_daily = df_daily_all[df_daily_all["symbol"] == symbol].copy()

        # Get LTP and pChange from latest 15m bar
        ltp = float(sym_15m["close"].iloc[-1]) if not sym_15m.empty else 0.0
        if not sym_daily.empty and len(sym_daily) >= 2:
            prev_close = float(sym_daily["close"].iloc[-2])
            pchange = ((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
        else:
            pchange = 0.0

        layers: List[LayerSignal] = []

        # Layer 1: Multi-TF Supertrend
        layers.append(self._layer_supertrend(sym_15m, sym_daily))

        # Layer 2: EMA Stack
        layers.append(self._layer_ema_stack(sym_daily))

        # Layer 3: ADX Trend Strength
        layers.append(self._layer_adx(sym_daily))

        # Layer 4: MACD Momentum
        layers.append(self._layer_macd(sym_15m))

        # Layer 5: RSI Divergence
        layers.append(self._layer_rsi_divergence(sym_15m))

        # Layer 6: OBV Divergence
        layers.append(self._layer_obv_divergence(sym_15m))

        # Layer 7: Price Structure
        layers.append(self._layer_price_structure(sym_daily))

        # Layer 8: News Catalyst
        clean_sym = symbol.replace("NSE:", "").replace("-EQ", "")
        layers.append(self._layer_news(clean_sym, news_map))

        # Composite Verdict
        bull_count = sum(1 for l in layers if l.direction == "BULL")
        bear_count = sum(1 for l in layers if l.direction == "BEAR")
        total_layers = len(layers)

        # Weighted score
        direction_scores = [l.strength * (1.0 if l.direction == "BULL" else (-1.0 if l.direction == "BEAR" else 0.0)) for l in layers]
        raw_score = sum(direction_scores) / total_layers if total_layers > 0 else 0.0
        composite_score = round((raw_score + 1.0) / 2.0 * 100, 1)  # Normalize to 0-100

        # Check for divergence warnings
        has_rsi_div = any(l.name == "rsi_divergence" and l.direction != "NEUTRAL" for l in layers)
        has_obv_div = any(l.name == "obv_divergence" and "DIVERGING" in l.label for l in layers)
        adx_layer = next((l for l in layers if l.name == "adx_strength"), None)
        adx_strong = adx_layer and adx_layer.strength > 0.5

        max_agree = max(bull_count, bear_count)
        composite_direction = "BULL" if bull_count > bear_count else ("BEAR" if bear_count > bull_count else "NEUTRAL")

        if (has_rsi_div or has_obv_div) and max_agree >= 3:
            verdict = "⚠️ REVERSAL WARNING"
        elif max_agree >= 6 and adx_strong:
            verdict = "🔥 STRONG MOMENTUM"
        elif max_agree >= 4:
            verdict = "⚡ BUILDING"
        elif max_agree >= 3:
            verdict = "🔄 DEVELOPING"
        else:
            verdict = "💤 NO EDGE"

        return StockIntelligence(
            symbol=symbol,
            ltp=round(ltp, 2),
            pchange=round(pchange, 2),
            timestamp=datetime.now().strftime("%H:%M:%S"),
            layers=layers,
            composite_score=composite_score,
            composite_verdict=verdict,
            composite_direction=composite_direction,
            bull_count=bull_count,
            bear_count=bear_count,
        )

    # ── Layer Implementations ──────────────────────────────────────────────────

    def _layer_supertrend(self, df_15m: pd.DataFrame, df_daily: pd.DataFrame) -> LayerSignal:
        """Layer 1: Multi-TF Supertrend Alignment (15m + Daily)."""
        st_15m_dir = 0
        st_daily_dir = 0

        if not df_15m.empty and len(df_15m) >= 10:
            try:
                st_calc = SupertrendIndicator(period=7, multiplier=3)
                st_df = st_calc.calculate(df_15m)
                if not st_df.empty and "supertrend_direction" in st_df.columns:
                    st_15m_dir = int(st_df["supertrend_direction"].iloc[-1])
            except Exception:
                pass

        if not df_daily.empty and len(df_daily) >= 10:
            try:
                st_calc = SupertrendIndicator(period=7, multiplier=3)
                st_df = st_calc.calculate(df_daily)
                if not st_df.empty and "supertrend_direction" in st_df.columns:
                    st_daily_dir = int(st_df["supertrend_direction"].iloc[-1])
            except Exception:
                pass

        if st_15m_dir == 1 and st_daily_dir == 1:
            return LayerSignal("supertrend_alignment", "BULL", 1.0, "🟢 ALIGNED BULL", "Both 15m and Daily Supertrend bullish")
        elif st_15m_dir == -1 and st_daily_dir == -1:
            return LayerSignal("supertrend_alignment", "BEAR", 1.0, "🔴 ALIGNED BEAR", "Both 15m and Daily Supertrend bearish")
        elif st_15m_dir != 0 and st_daily_dir != 0:
            return LayerSignal("supertrend_alignment", "NEUTRAL", 0.3, "⚠️ CONFLICTING", f"15m={'BULL' if st_15m_dir==1 else 'BEAR'}, Daily={'BULL' if st_daily_dir==1 else 'BEAR'}")
        else:
            return LayerSignal("supertrend_alignment", "NEUTRAL", 0.0, "— INSUFFICIENT", "Not enough data for Supertrend")

    def _layer_ema_stack(self, df_daily: pd.DataFrame) -> LayerSignal:
        """Layer 2: EMA Stack Order (9/21/50 on Daily candles)."""
        if df_daily.empty or len(df_daily) < 50:
            return LayerSignal("ema_stack", "NEUTRAL", 0.0, "— INSUFFICIENT", "Need 50+ daily bars")

        close = df_daily["close"]
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        if ema9 > ema21 > ema50:
            return LayerSignal("ema_stack", "BULL", 1.0, "🟢 PERFECT BULL", f"EMA 9>{ema9:.0f} > 21>{ema21:.0f} > 50>{ema50:.0f}")
        elif ema9 < ema21 < ema50:
            return LayerSignal("ema_stack", "BEAR", 1.0, "🔴 PERFECT BEAR", f"EMA 9<{ema9:.0f} < 21<{ema21:.0f} < 50<{ema50:.0f}")
        elif ema9 > ema21 and ema21 < ema50:
            return LayerSignal("ema_stack", "BULL", 0.5, "🟡 INVERTING UP", "Short-term EMA crossing above mid")
        elif ema9 < ema21 and ema21 > ema50:
            return LayerSignal("ema_stack", "BEAR", 0.5, "🟡 INVERTING DOWN", "Short-term EMA crossing below mid")
        else:
            return LayerSignal("ema_stack", "NEUTRAL", 0.3, "⚪ MIXED", "EMA stack order is mixed")

    def _layer_adx(self, df_daily: pd.DataFrame) -> LayerSignal:
        """Layer 3: ADX Trend Strength (Average Directional Index)."""
        if df_daily.empty or len(df_daily) < 20:
            return LayerSignal("adx_strength", "NEUTRAL", 0.0, "— INSUFFICIENT", "Need 20+ daily bars")

        df = df_daily.copy()
        high, low, close = df["high"], df["low"], df["close"]

        # +DM / -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed (14-period Wilder's smoothing)
        period = 14
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        adx_val = float(adx.iloc[-1]) if not adx.empty else 0.0
        plus_di_val = float(plus_di.iloc[-1]) if not plus_di.empty else 0.0
        minus_di_val = float(minus_di.iloc[-1]) if not minus_di.empty else 0.0

        # ADX direction from DI
        if plus_di_val > minus_di_val:
            di_direction = "BULL"
        elif minus_di_val > plus_di_val:
            di_direction = "BEAR"
        else:
            di_direction = "NEUTRAL"

        if adx_val >= 30:
            strength = min(adx_val / 50, 1.0)
            return LayerSignal("adx_strength", di_direction, strength, f"🔥 STRONG ({adx_val:.0f})", f"ADX={adx_val:.1f}, +DI={plus_di_val:.1f}, -DI={minus_di_val:.1f}")
        elif adx_val >= 20:
            return LayerSignal("adx_strength", di_direction, 0.5, f"🟡 MODERATE ({adx_val:.0f})", f"ADX={adx_val:.1f}")
        else:
            return LayerSignal("adx_strength", "NEUTRAL", 0.2, f"💤 RANGING ({adx_val:.0f})", f"ADX={adx_val:.1f} — no trend")

    def _layer_macd(self, df_15m: pd.DataFrame) -> LayerSignal:
        """Layer 4: MACD Momentum (histogram expansion/contraction on 15m)."""
        if df_15m.empty or len(df_15m) < 35:
            return LayerSignal("macd_momentum", "NEUTRAL", 0.0, "— INSUFFICIENT", "Need 35+ 15m bars")

        close = df_15m["close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        hist_now = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        macd_now = float(macd_line.iloc[-1])

        # Detect crossover
        macd_prev = float(macd_line.iloc[-2])
        sig_prev = float(signal_line.iloc[-2])
        sig_now = float(signal_line.iloc[-1])

        if macd_prev <= sig_prev and macd_now > sig_now:
            return LayerSignal("macd_momentum", "BULL", 0.9, "🟢 CROSSOVER BULL", "MACD crossed above signal line")
        elif macd_prev >= sig_prev and macd_now < sig_now:
            return LayerSignal("macd_momentum", "BEAR", 0.9, "🔴 CROSSOVER BEAR", "MACD crossed below signal line")

        # Histogram analysis
        if hist_now > 0 and hist_now > hist_prev:
            return LayerSignal("macd_momentum", "BULL", 0.7, "📈 EXPANDING BULL", f"Histogram: {hist_now:+.2f} (growing)")
        elif hist_now < 0 and hist_now < hist_prev:
            return LayerSignal("macd_momentum", "BEAR", 0.7, "📉 EXPANDING BEAR", f"Histogram: {hist_now:+.2f} (growing)")
        elif hist_now > 0 and hist_now < hist_prev:
            return LayerSignal("macd_momentum", "BULL", 0.3, "⚠️ CONTRACTING BULL", f"Histogram: {hist_now:+.2f} (fading)")
        elif hist_now < 0 and hist_now > hist_prev:
            return LayerSignal("macd_momentum", "BEAR", 0.3, "⚠️ CONTRACTING BEAR", f"Histogram: {hist_now:+.2f} (fading)")
        else:
            return LayerSignal("macd_momentum", "NEUTRAL", 0.1, "⚪ FLAT", "No clear MACD momentum")

    def _layer_rsi_divergence(self, df_15m: pd.DataFrame) -> LayerSignal:
        """Layer 5: RSI Divergence detection on 15m candles."""
        if df_15m.empty or len(df_15m) < 20:
            return LayerSignal("rsi_divergence", "NEUTRAL", 0.0, "— INSUFFICIENT", "Need 20+ 15m bars")

        close = df_15m["close"]
        # Calculate RSI-14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))

        rsi_val = float(rsi.iloc[-1])

        # Check for price-RSI divergence over last 20 bars
        lookback = min(20, len(close) - 1)
        price_recent = close.iloc[-lookback:]
        rsi_recent = rsi.iloc[-lookback:]

        price_high = float(price_recent.max())
        price_low = float(price_recent.min())
        price_now = float(close.iloc[-1])

        # Find RSI at the price extreme points
        price_high_idx = price_recent.idxmax()
        price_low_idx = price_recent.idxmin()
        rsi_at_high = float(rsi.loc[price_high_idx]) if price_high_idx in rsi.index else rsi_val
        rsi_at_low = float(rsi.loc[price_low_idx]) if price_low_idx in rsi.index else rsi_val

        # Bearish divergence: price making higher high but RSI making lower high
        if price_now >= price_high * 0.995 and rsi_val < rsi_at_high - 5:
            return LayerSignal("rsi_divergence", "BEAR", 0.8, "🔴 BEAR DIVERGENCE", f"Price at high but RSI={rsi_val:.0f} < {rsi_at_high:.0f}")

        # Bullish divergence: price making lower low but RSI making higher low
        if price_now <= price_low * 1.005 and rsi_val > rsi_at_low + 5:
            return LayerSignal("rsi_divergence", "BULL", 0.8, "🟢 BULL DIVERGENCE", f"Price at low but RSI={rsi_val:.0f} > {rsi_at_low:.0f}")

        # RSI extremes
        if rsi_val >= 75:
            return LayerSignal("rsi_divergence", "BEAR", 0.4, f"🔥 OVERBOUGHT ({rsi_val:.0f})", "RSI > 75, potential pullback")
        elif rsi_val <= 25:
            return LayerSignal("rsi_divergence", "BULL", 0.4, f"🧊 OVERSOLD ({rsi_val:.0f})", "RSI < 25, potential bounce")

        return LayerSignal("rsi_divergence", "NEUTRAL", 0.0, f"— RSI {rsi_val:.0f}", "No divergence detected")

    def _layer_obv_divergence(self, df_15m: pd.DataFrame) -> LayerSignal:
        """Layer 6: OBV Divergence — is smart money confirming the price move?"""
        if df_15m.empty or len(df_15m) < 15:
            return LayerSignal("obv_divergence", "NEUTRAL", 0.0, "— INSUFFICIENT", "Need 15+ 15m bars")

        try:
            obv_calc = OBVIndicator()
            df_obv = obv_calc.calculate(df_15m)
        except Exception:
            return LayerSignal("obv_divergence", "NEUTRAL", 0.0, "— ERROR", "OBV calculation failed")

        if "obv" not in df_obv.columns:
            return LayerSignal("obv_divergence", "NEUTRAL", 0.0, "— ERROR", "OBV column missing")

        # Compare price slope vs OBV slope over last 10 bars
        lookback = min(10, len(df_obv) - 1)
        price_slope = (float(df_obv["close"].iloc[-1]) - float(df_obv["close"].iloc[-lookback])) / max(float(df_obv["close"].iloc[-lookback]), 1)
        obv_start = float(df_obv["obv"].iloc[-lookback])
        obv_end = float(df_obv["obv"].iloc[-1])
        obv_slope = (obv_end - obv_start) / max(abs(obv_start), 1) if obv_start != 0 else 0

        # Price up but OBV down = bearish divergence (smart money not buying)
        if price_slope > 0.005 and obv_slope < -0.01:
            return LayerSignal("obv_divergence", "BEAR", 0.7, "🔴 DIVERGING (Sell Pressure)", "Price rising but OBV declining — smart money exiting")

        # Price down but OBV up = bullish divergence (smart money accumulating)
        if price_slope < -0.005 and obv_slope > 0.01:
            return LayerSignal("obv_divergence", "BULL", 0.7, "🟢 DIVERGING (Accumulation)", "Price falling but OBV rising — smart money accumulating")

        # Confirming
        if price_slope > 0.005 and obv_slope > 0.01:
            return LayerSignal("obv_divergence", "BULL", 0.6, "✅ CONFIRMING BULL", "Price and OBV both rising")
        if price_slope < -0.005 and obv_slope < -0.01:
            return LayerSignal("obv_divergence", "BEAR", 0.6, "✅ CONFIRMING BEAR", "Price and OBV both falling")

        return LayerSignal("obv_divergence", "NEUTRAL", 0.2, "⚪ NEUTRAL", "No clear OBV divergence")

    def _layer_price_structure(self, df_daily: pd.DataFrame) -> LayerSignal:
        """Layer 7: Price Structure — Higher Highs/Lows or Lower Highs/Lows on Daily."""
        if df_daily.empty or len(df_daily) < 5:
            return LayerSignal("price_structure", "NEUTRAL", 0.0, "— INSUFFICIENT", "Need 5+ daily bars")

        # Use last 5 daily bars to detect swing structure
        recent = df_daily.tail(5)
        highs = recent["high"].values
        lows = recent["low"].values

        # Count consecutive higher highs / higher lows
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])

        # Detect structural break
        if hh_count >= 3 and hl_count >= 3:
            return LayerSignal("price_structure", "BULL", 1.0, "📈 UPTREND (HH/HL)", f"{hh_count} Higher Highs, {hl_count} Higher Lows in last 5 days")
        elif lh_count >= 3 and ll_count >= 3:
            return LayerSignal("price_structure", "BEAR", 1.0, "📉 DOWNTREND (LH/LL)", f"{lh_count} Lower Highs, {ll_count} Lower Lows in last 5 days")
        elif hh_count >= 2 and hl_count >= 2:
            return LayerSignal("price_structure", "BULL", 0.7, "📈 UPTREND BUILDING", f"{hh_count}HH/{hl_count}HL")
        elif lh_count >= 2 and ll_count >= 2:
            return LayerSignal("price_structure", "BEAR", 0.7, "📉 DOWNTREND BUILDING", f"{lh_count}LH/{ll_count}LL")

        # Detect trend break
        if hh_count >= 2 and ll_count >= 1:
            return LayerSignal("price_structure", "NEUTRAL", 0.4, "⚠️ FIRST LL BREAK", "Higher highs but first lower low — potential top")
        if lh_count >= 2 and hl_count >= 1:
            return LayerSignal("price_structure", "NEUTRAL", 0.4, "⚠️ FIRST HH BREAK", "Lower highs but first higher low — potential bottom")

        return LayerSignal("price_structure", "NEUTRAL", 0.2, "⚪ RANGING", "No clear structural trend")

    def _layer_news(self, clean_symbol: str, news_map: Dict[str, str]) -> LayerSignal:
        """Layer 8: News Catalyst — match headlines to stock names."""
        headline = news_map.get(clean_symbol)
        if headline:
            return LayerSignal("news_catalyst", "NEUTRAL", 0.5, f"📰 {headline[:50]}", headline)
        return LayerSignal("news_catalyst", "NEUTRAL", 0.0, "—", "No recent news matched")

    # ── Data Fetching ──────────────────────────────────────────────────────────

    def _fetch_15m_batch(self, symbols: List[str], target_date: date) -> pd.DataFrame:
        """Batch-fetch 15m candles for all symbols (10-day lookback)."""
        if not symbols:
            return pd.DataFrame()
            
        placeholders = ','.join(['?' for _ in symbols])
        query = f"""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv_15m
            WHERE symbol IN ({placeholders})
              AND timestamp >= date(?, '-10 days')
              AND timestamp <= date(?, '+1 day')
            ORDER BY symbol, timestamp ASC
        """
        
        try:
            with self.engine.connect() as conn:
                params = tuple(list(symbols) + [target_date.isoformat(), target_date.isoformat()])
                df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch 15m batch: {e}")
            return pd.DataFrame()

    def _fetch_daily_batch(self, symbols: List[str], target_date: date) -> pd.DataFrame:
        """Batch-fetch daily candles for all symbols (60-day lookback)."""
        if not symbols:
            return pd.DataFrame()
            
        placeholders = ','.join(['?' for _ in symbols])
        query = f"""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE symbol IN ({placeholders})
              AND timestamp >= date(?, '-60 days')
              AND timestamp <= ?
            ORDER BY symbol, timestamp ASC
        """
        try:
            with self.engine.connect() as conn:
                params = tuple(list(symbols) + [target_date.isoformat(), target_date.isoformat()])
                df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except Exception as e:
            LOGGER.error(f"Failed to fetch daily batch: {e}")
            return pd.DataFrame()

    def _fetch_news_for_symbols(self, symbols: List[str]) -> Dict[str, str]:
        """
        Fetch news headlines and match them to stock symbols.
        Uses a simple keyword match. Results are cached for 30 minutes.
        """
        now = datetime.now()
        if self._news_cache is not None and self._news_cache_time is not None:
            elapsed = (now - self._news_cache_time).total_seconds()
            if elapsed < 1800:  # 30-minute cache
                return self._news_cache

        news_map: Dict[str, str] = {}
        try:
            from trade_system.shared.utils.news_scraper import BusinessNewsScraper
            scraper = BusinessNewsScraper()
            headlines = scraper.get_all_headlines()

            for sym in symbols:
                clean = sym.replace("NSE:", "").replace("-EQ", "").upper()
                for headline in headlines:
                    headline_upper = headline.upper()
                    if clean in headline_upper:
                        news_map[clean] = headline
                        break
                    # Try partial match for longer names (e.g., "RELIANCE" in "Reliance Industries")
                    if len(clean) >= 4 and clean[:4] in headline_upper:
                        news_map[clean] = headline
                        break

        except Exception as e:
            LOGGER.warning(f"News scraping failed (non-fatal): {e}")

        self._news_cache = news_map
        self._news_cache_time = now
        return news_map
