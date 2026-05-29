"""
CandidateScreenerAgent — screens FO universe for breakout-ready candidates.

Uses multi-timeframe analysis, volume delta, compression, and agent weights
(from WeightEvolver) to rank stocks. LLM optionally validates the top picks.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from trade_system.core import MarketContext, TradeHorizon
from trade_system.infrastructure.data.fo_universe import get_fo_universe, FO_METADATA
from trade_system.application.indicators.volume_delta import VolumeDeltaIndicator
from trade_system.application.indicators.compression import CompressionIndicator
from trade_system.application.indicators.volume_profile import VolumeProfileIndicator
from trade_system.application.indicators.vwap import VWAPIndicator
from trade_system.application.indicators.retracement import RetracementIndicator
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.application.analysis.mwpl_analyzer import MwplAnalyzer
from trade_system.application.analysis.pre_breakout_predictor import PreBreakoutPredictor
from trade_system.core.ports.weights import WeightsProvider

LOGGER = logging.getLogger(__name__)

# Swing classification thresholds
SWING_RSI_MIN = 55.0
SWING_ADX_MIN = 22.0
SWING_ALIGNMENT_MIN = 0.70


@dataclass
class CandidateScore:
    """Scored candidate for option trade."""
    symbol: str
    intraday_score: float       # 0.0 – 1.0
    swing_score: float          # 0.0 – 1.0
    horizon: TradeHorizon
    direction: str              # "CALL" or "PUT"
    sector: str
    entry_price: float

    # Feature snapshot values
    rsi_daily: float | None = None
    rsi_hourly: float | None = None
    adx: float | None = None
    volume_surge: float | None = None
    atr_pct: float | None = None
    is_compressed: bool = False
    vol_delta_positive: bool = False
    above_poc: bool | None = None
    alignment_score: float | None = None
    breakout_type: str | None = None
    pattern: str | None = None
    
    # Liquidity metadata
    spread_pct: float | None = None
    is_liquid: bool = True

    component_scores: dict[str, float] = field(default_factory=dict)

    def best_score(self) -> float:
        return max(self.intraday_score, self.swing_score)


from trade_system.infrastructure.database.connection import get_engine
from trade_system.infrastructure.database.repository import get_market_data, log_agent_thought
from sqlalchemy.orm import Session

class CandidateScreenerAgent:
    """
    Screens the F&O universe for option-buying candidates.
    Prioritizes the optimized database tables (ohlcv_daily, ohlcv_5m).
    """

    NIFTY_SYMBOL = "NSE:NIFTY50-INDEX"
    BANKNIFTY_SYMBOL = "NSE:NIFTYBANK-INDEX"

    def __init__(
        self,
        broker: Any = None,
        weight_evolver: WeightsProvider | None = None,
        max_universe: int = 200,     # Increased to cover full F&O universe
        settings: any = None
    ) -> None:
        self.broker = broker
        if weight_evolver is None:
            from trade_system.application.evolution.weight_evolver import WeightEvolver
            self.weight_evolver = WeightEvolver()
        else:
            self.weight_evolver = weight_evolver
        self.max_universe = max_universe
        self.engine = get_engine()
        
        if settings is None:
            from trade_system.config import Settings
            self.settings = Settings.load()
        else:
            self.settings = settings

        self.vd_ind = VolumeDeltaIndicator()
        self.comp_ind = CompressionIndicator()
        self.vp_ind = VolumeProfileIndicator(price_step=5.0)
        self.vwap_ind = VWAPIndicator()
        self.ret_ind = RetracementIndicator()
        self.st_ind = SupertrendIndicator(period=7, multiplier=3)
        self.mwpl_analyzer = MwplAnalyzer()
        self.pre_breakout_predictor = PreBreakoutPredictor()

        # Load evolved weights
        self.weights = self.weight_evolver.load_weights("candidate_screener")
        LOGGER.info("CandidateScreenerAgent weights: %s", self.weights)

    def _thought(self, message: str, symbol: str | None = None, action: str | None = None):
        """Log screening logic to DB for dashboard visibility."""
        try:
            with Session(self.engine) as session:
                log_agent_thought(session, "CandidateScreener", message, symbol, action)
        except: pass

    def screen(
        self,
        market_context: MarketContext,
        include_nifty: bool = True,
        include_fo: bool = True,
    ) -> list[CandidateScore]:
        """
        Screen candidates using DATABASE data and day's Top Gainers.
        """
        candidates: list[CandidateScore] = []
        
        # 0. Identify MWPL setups (Ban list, Squeeze, Unwinding)
        mwpl_setups = self.mwpl_analyzer.identify_setups()

        with Session(self.engine) as session:
            # --- Screen Nifty index ---
            if include_nifty:
                nifty_score = self._screen_symbol(session, self.NIFTY_SYMBOL, market_context, is_index=True)
                if nifty_score:
                    candidates.append(nifty_score)

            # --- Screen F&O stocks ---
            if include_fo:
                # 1. Get base F&O universe
                universe = set(get_fo_universe())
                
                # 2. Dynamic Hot-List: Add today's Top Gainers
                try:
                    today_str = date.today().strftime('%Y%m%d')
                    top_gainers_file = Path(self.settings.data_dir) / "top_gainers" / f"top100_{today_str}.json"
                    if top_gainers_file.exists():
                        with open(top_gainers_file, "r") as f:
                            data = json.load(f)
                            top_symbols = [g["symbol"] for g in data.get("gainers", [])]
                            universe.update(top_symbols)
                            LOGGER.info("Added %d top gainers to screening universe.", len(top_symbols))
                except Exception as e:
                    LOGGER.debug("Could not load top gainers for dynamic screening: %s", e)

                universe_list = list(universe)[:self.max_universe]
                LOGGER.info("Screening %d total stocks using database history...", len(universe_list))

                for sym in universe_list:
                    try:
                        # Safety: Skip stocks in F&O Ban
                        clean_sym = sym.replace("NSE:", "").replace("-EQ", "")
                        if clean_sym in mwpl_setups.get("BAN", []):
                            LOGGER.debug(f"Skipping {sym}: In F&O Ban.")
                            continue

                        score = self._screen_symbol(session, sym, market_context)
                        
                        # Apply MWPL setup boosters
                        if score:
                            if clean_sym in mwpl_setups.get("SQUEEZE", []):
                                if score.direction == "CALL":
                                    score.intraday_score = min(1.0, score.intraday_score + 0.2)
                                    score.pattern = (score.pattern or "") + ", MWPL_SQUEEZE"
                            elif clean_sym in mwpl_setups.get("UNWINDING", []):
                                if score.direction == "PUT":
                                    score.intraday_score = min(1.0, score.intraday_score + 0.2)
                                    score.pattern = (score.pattern or "") + ", MWPL_UNWINDING"

                        if score and score.best_score() >= 0.40: # Aggressive threshold
                            candidates.append(score)
                    except Exception as exc:
                        LOGGER.debug("Screening error for %s: %s", sym, exc)

        # Rank by best score descending
        candidates.sort(key=lambda c: c.best_score(), reverse=True)
        LOGGER.info("Screened %d total candidates.", len(candidates))
        return candidates

    def _screen_symbol(
        self,
        session: Session,
        symbol: str,
        market_context: MarketContext,
        is_index: bool = False,
    ) -> CandidateScore | None:
        """Score a single symbol using database data."""
        try:
            # 1. Fetch Daily Data (Last 30 days)
            daily_data = get_market_data(session, symbol, "D", limit=30)
            if not daily_data:
                return None
            
            df_daily = pd.DataFrame([
                {"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume} 
                for d in daily_data
            ])
            
            # 2. Fetch 15m Data (Last 2 days)
            intraday_data = get_market_data(session, symbol, "15", limit=200)
            if not intraday_data:
                # Fallback to daily-only analysis
                df_15m = df_daily 
            else:
                df_15m = pd.DataFrame([
                    {"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume} 
                    for d in intraday_data
                ])
                
            if df_15m.empty:
                return None
                
        except Exception as exc:
            LOGGER.debug("Database read failed for %s: %s", symbol, exc)
            return None

        return self._compute_score(symbol, df_15m, df_daily, market_context, is_index)

    def _compute_score(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        df_daily: pd.DataFrame | None,
        market_context: MarketContext,
        is_index: bool,
    ) -> CandidateScore | None:
        """Compute multi-component score for a symbol."""
        try:
            # Calculate indicators
            df_15m = self.vd_ind.calculate(df_15m)
            df_15m = self.comp_ind.calculate(df_15m)
            df_15m = self.vwap_ind.calculate(df_15m)
            df_15m = self.ret_ind.calculate(df_15m)
            df_15m = self.st_ind.calculate(df_15m)
            profile = self.vp_ind.calculate(df_15m.tail(26))  # ~6.5hrs of 15m bars

            # Calculate Daily Supertrend for Multi-TF Alignment
            prev_day_trend = 0
            if df_daily is not None and not df_daily.empty:
                df_daily = self.st_ind.calculate(df_daily)
                # Previous day's direction (Trend from the last closed daily candle)
                if len(df_daily) >= 2:
                    prev_day_trend = int(df_daily['supertrend_direction'].iloc[-2])
        except Exception as exc:
            LOGGER.debug("Indicator error for %s: %s", symbol, exc)
            return None

        latest = df_15m.iloc[-1]
        entry_price = float(latest.get("close", 0))
        if entry_price <= 0:
            return None

        # --- Liquidity Hardening ---
        # 1. Fetch live quote for spread check
        spread_pct = 0.0
        is_liquid = True
        if self.broker and hasattr(self.broker, "get_quotes"):
            try:
                quote = self.broker.get_quotes([symbol]).get(symbol)
                if quote:
                    bid = quote.bid or entry_price * 0.999
                    ask = quote.ask or entry_price * 1.001
                    spread_pct = ((ask - bid) / entry_price) * 100
                    # Reject if spread is > 0.5% (Illiquid for options)
                    if spread_pct > 0.5:
                        LOGGER.warning(f"Rejecting {symbol}: High spread {spread_pct:.2f}%")
                        is_liquid = False
            except Exception as e:
                LOGGER.debug(f"Liquidity check failed for {symbol}: {e}")

        # 2. Volume Consistency (Avg volume check)
        avg_vol = df_15m["volume"].tail(50).mean()
        if avg_vol < 1000: # Threshold for 15m bars
            LOGGER.warning(f"Rejecting {symbol}: Low average volume {avg_vol:.0f}")
            is_liquid = False

        if not is_liquid:
            return None

        # --- Component Scores (0.0 – 1.0) ---
        scores: dict[str, float] = {}

        # 1. Volume Delta
        delta = float(latest.get("delta", 0))
        cvd = float(latest.get("cvd", 0))
        vol_delta_positive = delta > 0 and cvd > 0
        scores["volume_delta"] = 1.0 if vol_delta_positive else (0.4 if delta > 0 else 0.0)

        # 2. Compression
        is_compressed = bool(latest.get("is_compressed", False))
        comp_ratio = float(latest.get("range_compression", 50)) / 100.0
        scores["compression"] = 1.0 if is_compressed else comp_ratio

        # 3. Value Area
        va_score = 0.3   # Default: inside value area
        above_poc: bool | None = None
        if profile:
            price = entry_price
            if price > profile.value_area_high:
                va_score = 1.0
                above_poc = True
            elif price > profile.point_of_control:
                va_score = 0.7
                above_poc = True
            elif price >= profile.value_area_low:
                va_score = 0.4
                above_poc = False
            else:
                va_score = 0.1
                above_poc = False
        scores["value_area"] = va_score

        # 4. Trend Alignment (simple multi-TF proxy from 15m data)
        rsi_15m = self._calc_rsi(df_15m)
        rsi_daily = self._calc_rsi(df_daily) if (df_daily is not None and len(df_daily) >= 15) else None
        adx = self._calc_adx(df_15m)
        alignment_score = self._calc_alignment(df_15m, rsi_15m, rsi_daily, adx)

        # Incorporate Previous Day trend into alignment
        if prev_day_trend != 0:
            current_15m_trend = latest['supertrend_direction']
            if current_15m_trend == prev_day_trend:
                alignment_score = min(1.0, alignment_score + 0.15)
                LOGGER.debug(f"Trend alignment for {symbol}: 15m matches Daily.")
            else:
                alignment_score = max(0.0, alignment_score - 0.1)

        scores["trend_alignment"] = alignment_score

        # 5. Daily Retracement Confluence
        daily_zones = []
        if df_daily is not None and len(df_daily) >= 20:
            df_daily_calc = self.ret_ind.calculate(df_daily)
            daily_zones = self.ret_ind.get_active_zones(df_daily_calc)
            if daily_zones:
                scores["trend_alignment"] = min(1.0, scores["trend_alignment"] + 0.2)
                LOGGER.info(f"Daily Retracement detected for {symbol}: {[z.type for z in daily_zones]}")

        # 6. Momentum (Price & RSI) & Chop Filter
        momentum_score = 0.5
        if rsi_daily is not None:
            if rsi_daily >= 50:
                momentum_score = min(1.0, (rsi_daily - 40) / 30.0)
            else:
                momentum_score = 0.2

        # --- SIDEYWAYS SHIELD (Buyer's Protection) ---
        if adx is not None:
            if adx < 20:
                # ADX < 20 means NO TREND. Slashing momentum score.
                momentum_score *= 0.5
                self._thought(f"Sideways Shield: Slashing momentum due to low ADX ({adx:.1f}). Change: {price_change_pct:+.2f}%", symbol, "SIDEWAYS")
            elif adx > 25:
                momentum_score = min(1.0, momentum_score + 0.2)


        # 7. Intraday Performance (The "Heat" factor)

        # Calculate how much the stock is up today from the ACTUAL Market Open
        price_change_pct = 0.0
        if df_daily is not None and not df_daily.empty:
            # We want TODAY's open price to calculate accurate intraday % change
            today_open = float(df_daily['open'].iloc[-1])
            if today_open > 0:
                price_change_pct = ((entry_price - today_open) / today_open) * 100
        
        # Performance Score: Linear scale. 3% up = 0.6 score, 5% up = 1.0 score.
        perf_score = min(1.0, max(0.0, price_change_pct / 5.0))
        
        # --- TOP GAINER BOOSTER ---
        # If this stock is in today's Top 100 Gainers, give it a guaranteed minimum performance score
        try:
            today_str = date.today().strftime('%Y%m%d')
            top_gainers_file = Path(self.settings.data_dir) / "top_gainers" / f"top100_{today_str}.json"
            if top_gainers_file.exists():
                with open(top_gainers_file, "r") as f:
                    tg_data = json.load(f)
                    gainers = tg_data.get("gainers", [])
                    # Find rank
                    rank = next((i for i, g in enumerate(gainers) if g["symbol"] == symbol), None)
                    if rank is not None:
                        # Top 10 stocks get at least 0.9 performance score
                        boost = max(0.0, (100 - rank) / 100.0)
                        perf_score = max(perf_score, boost)
                        LOGGER.debug(f"Applied Top Gainer boost for {symbol}: Rank {rank+1}")
        except: pass

        scores["performance"] = perf_score
        scores["momentum"] = momentum_score

        # 8. Market Context alignment
        ctx_score = 0.5
        if market_context.bias == "BULLISH":
            ctx_score = 0.8
        elif market_context.bias == "BEARISH":
            ctx_score = 0.2
        if not market_context.tradeable:
            ctx_score = 0.1
        scores["market_context"] = ctx_score

        # 9. Dynamic Direction Detection (Symbol-Specific)
        # Instead of following index bias, follow the stock's own heat
        vwap_val = float(latest.get("vwap", entry_price))
        
        # Determine direction based on price vs VWAP and overall performance
        if entry_price > vwap_val and perf_score > 0.4:
            direction = "CALL"
        elif entry_price < vwap_val and price_change_pct < -1.5:
            direction = "PUT"
        else:
            # Fallback to index bias only if stock momentum is neutral
            direction = "CALL" if market_context.bias != "BEARISH" else "PUT"
        
        vwap_score = 0.5
        if direction == "CALL":
            vwap_score = 1.0 if entry_price > vwap_val else 0.2
        else:
            vwap_score = 1.0 if entry_price < vwap_val else 0.2
        scores["vwap"] = vwap_score

        # 10. Retracement / Confluence Zones
        active_zones = self.ret_ind.get_active_zones(df_15m)
        ret_score = 0.3 # Default
        if active_zones:
            types = [z.type for z in active_zones]
            if "BREAKOUT" in types or "FIB" in types:
                ret_score = 1.0
            elif "FVG" in types:
                ret_score = 0.8
        scores["retracement"] = ret_score

        # --- OPTION BUYER VIABILITY (Range Check) ---
        # If today's High-Low range is < 1x of average Daily ATR, the stock is "Dormant"
        # Option buyers will lose to theta in dormant stocks.
        range_score = 1.0
        try:
            day_high = float(df_daily['high'].iloc[-1])
            day_low = float(df_daily['low'].iloc[-1])
            curr_range = day_high - day_low
            
            # Get historical ATR
            hist_atr = self._calc_atr(df_daily)
            if hist_atr and curr_range < (hist_atr * 0.8):
                # Stock is moving in a very tight box today
                range_score = 0.3
                self._thought(f"Buyer Viability: Range too tight ({curr_range:.1f} vs ATR {hist_atr:.1f}). Change: {price_change_pct:+.2f}%", symbol, "DORMANT")
        except: pass
        scores["viability"] = range_score

        # --- PRE-BREAKOUT PREDICTION ---
        pre_breakout_res = self.pre_breakout_predictor.analyze(df_15m)
        pb_score = pre_breakout_res.get("accumulation_score", 0.0)
        is_pb_tagged = False
        pb_pattern = ""
        
        if pb_score > 0.6:
            # Huge accumulation happening while price is squeezed/flat
            scores["pre_breakout"] = pb_score
            is_pb_tagged = True
            direction = pre_breakout_res["predictive_direction"] or direction
            pb_pattern = "PRE_BREAKOUT_ACCUMULATION"

        # --- Weighted total ---
        # "Behavioral Shield" Weights
        current_weights = {
            "performance": 0.30,      
            "momentum": 0.20,         
            "viability": 0.15,        
            "volume_delta": 0.10,     
            "pre_breakout": 0.15,     # Added predictive weight
            "trend_alignment": 0.05,  
            "value_area": 0.03,       
            "retracement": 0.02       
        }
        
        total = sum(scores.get(k, 0.0) * current_weights.get(k, 0) for k in current_weights)
        weight_sum = sum(current_weights.values())
        if weight_sum > 0:
            total /= weight_sum

        # --- Horizon classification ---
        is_swing = (
            (rsi_daily or 0) >= SWING_RSI_MIN
            and (adx or 0) >= SWING_ADX_MIN
            and alignment_score >= SWING_ALIGNMENT_MIN
        )
        horizon = TradeHorizon.SWING if is_swing else TradeHorizon.INTRADAY

        # Swing score: boost alignment & momentum, discount compression
        intraday_score = round(min(total, 1.0), 4)
        swing_score = round(min(
            total + (0.1 if is_swing else 0.0),
            1.0
        ), 4)

        # Volume surge
        vol_ma = df_15m["volume"].rolling(20).mean().iloc[-1]
        vol_surge = float(latest.get("volume", 0)) / float(vol_ma) if vol_ma > 0 else 1.0

        # ATR %
        atr = self._calc_atr(df_15m)
        atr_pct = (atr / entry_price * 100) if atr and entry_price > 0 else None

        # Breakout type
        breakout_type = self._detect_breakout_type(df_15m, df_daily)
        
        # Combine patterns
        combined_pattern = ", ".join([z.type for z in active_zones]) if active_zones else ""
        if is_pb_tagged:
            combined_pattern += (", " if combined_pattern else "") + pb_pattern

        return CandidateScore(
            symbol=symbol,
            intraday_score=intraday_score,
            swing_score=swing_score,
            horizon=horizon,
            direction=direction,
            sector=FO_METADATA.get(symbol, "Index" if is_index else "Other"),
            entry_price=entry_price,
            rsi_daily=rsi_daily,
            rsi_hourly=rsi_15m,
            adx=adx,
            volume_surge=round(vol_surge, 2),
            atr_pct=round(atr_pct, 3) if atr_pct else None,
            is_compressed=is_compressed,
            vol_delta_positive=vol_delta_positive,
            above_poc=above_poc,
            alignment_score=round(alignment_score, 3),
            breakout_type=breakout_type,
            pattern=combined_pattern if combined_pattern else None,
            spread_pct=round(spread_pct, 3),
            is_liquid=is_liquid,
            component_scores=scores,
        )

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_rsi(df: pd.DataFrame | None, period: int = 14) -> float | None:
        if df is None or len(df) < period + 1:
            return None
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return round(float(val), 2) if pd.notna(val) else None

    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> float | None:
        if len(df) < period * 2:
            return None
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            dm_plus = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
            dm_minus = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
            atr = tr.ewm(span=period).mean()
            di_plus = 100 * dm_plus.ewm(span=period).mean() / (atr + 1e-9)
            di_minus = 100 * dm_minus.ewm(span=period).mean() / (atr + 1e-9)
            dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
            adx = dx.ewm(span=period).mean()
            val = adx.iloc[-1]
            return round(float(val), 2) if pd.notna(val) else None
        except Exception:
            return None

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float | None:
        if len(df) < period + 1:
            return None
        try:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            val = tr.rolling(period).mean().iloc[-1]
            return round(float(val), 3) if pd.notna(val) else None
        except Exception:
            return None

    @staticmethod
    def _calc_alignment(
        df: pd.DataFrame,
        rsi_15m: float | None,
        rsi_daily: float | None,
        adx: float | None,
    ) -> float:
        """Score multi-timeframe alignment (0-1)."""
        score = 0.0
        count = 0

        # Moving average alignment on 15m
        if len(df) >= 20:
            ma5 = df["close"].rolling(5).mean().iloc[-1]
            ma20 = df["close"].rolling(20).mean().iloc[-1]
            if pd.notna(ma5) and pd.notna(ma20):
                score += 1.0 if ma5 > ma20 else 0.0
                count += 1

        # RSI alignment (15m > 50 is bullish)
        if rsi_15m is not None:
            score += 1.0 if rsi_15m > 52 else (0.5 if rsi_15m > 45 else 0.0)
            count += 1

        # Daily RSI alignment
        if rsi_daily is not None:
            score += 1.0 if rsi_daily > 55 else (0.5 if rsi_daily > 45 else 0.0)
            count += 1

        # ADX (trending = aligned)
        if adx is not None:
            score += min(1.0, adx / 40.0)
            count += 1

        return round(score / count, 3) if count > 0 else 0.0

    @staticmethod
    def _detect_breakout_type(df_15m: pd.DataFrame, df_daily: pd.DataFrame | None) -> str | None:
        """Detect if current price is breaking out of a significant level."""
        if df_15m.empty:
            return None
        current_price = float(df_15m["close"].iloc[-1])

        # Check 20-bar high on 15m (intraday high breakout)
        recent_high = df_15m["high"].tail(20).max()
        if current_price >= recent_high * 0.998:
            return "intraday_high"

        # Check daily high breakout
        if df_daily is not None and len(df_daily) >= 2:
            prev_high = float(df_daily["high"].iloc[-2])
            if current_price >= prev_high * 0.998:
                return "daily_high"

        return None
