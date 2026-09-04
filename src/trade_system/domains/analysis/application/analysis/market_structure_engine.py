"""
MarketStructureEngine — Institutional Market Structure, Liquidity Sweeps & Volatility Regime Analyzer.

Continuously decodes:
1. Market Structure: Break of Structure (BOS), Change of Character (CHoCH),
   and Liquidity Sweeps of swing highs/lows (stop runs).
2. India VIX Dynamics: VIX level, intraday change, volatility regimes,
   and Price-to-VIX divergences.
3. Option Chain Migration: Tracking upward/downward shifts in CE/PE institutional walls
   and Rate of Change of OI (dOI/dt).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class MarketStructureInfo:
    """Consolidated market structure, volatility, and wall migration status."""
    symbol: str
    timestamp: str
    spot_price: float
    # 1. Market Structure
    structure_state: str  # "BULLISH_BOS", "BEARISH_BOS", "BULLISH_SWEEP_RECLAIM", "BEARISH_SWEEP_REJECT", "CHOP_CONSOLIDATION"
    structure_score: int  # -25 (Extreme Bearish) to +25 (Extreme Bullish)
    swing_high: float
    swing_low: float
    day_high: float
    day_low: float
    day_open: float
    # 2. VIX Dynamics
    vix_level: float
    vix_change: float
    vix_change_pct: float
    vix_regime: str       # "COMPLACENT (<12)", "NORMAL (12-16)", "ELEVATED (16-22)", "FEAR_SPIKE (>22)"
    vix_divergence: str   # "NONE", "BULLISH_EXPANSION", "BEARISH_COMPLACENCY", "VOLATILITY_WARNING"
    vix_score: int        # -15 to +15
    # 3. Wall Migration & Flow
    ce_wall: float
    pe_wall: float
    wall_migration: str   # "CE_WALL_LOWERING (Bearish)", "PE_WALL_RISING (Bullish)", "WALLS_COMPRESSING (Squeeze)", "STABLE"
    pcr_oi: float
    atm_volume_delta: int
    taker_aggression: str
    # 4. Expiry Day Dynamics & 0DTE Risk Management
    days_to_expiry: int = 5
    is_expiry_day: bool = False
    is_expiry_eve: bool = False
    expiry_phase: str = "NON_EXPIRY_NORMAL"  # "MORNING_PIN_DECAY_TRAP", "MIDDAY_BREAKEVEN_TEST", "AFTERNOON_GAMMA_SQUEEZE", "SETTLEMENT_CLOSING", "NON_EXPIRY_NORMAL"
    max_pain_strike: float = 0.0
    dist_to_max_pain: float = 0.0
    expiry_risk_alert: str = "NONE"
    recommended_contract_type: str = "ATM_CURRENT"  # "NEXT_WEEK_EXPIRY", "DEEP_ITM", "0DTE_ATM_EXPIRING"
    # 5. Wyckoff / Power of 3 (AMD) Market Cycle
    amd_phase: str = "CONSOLIDATION"       # "ACCUMULATION", "MANIPULATION_SPRING", "MANIPULATION_UTAD", "DISTRIBUTION_BULLISH", "DISTRIBUTION_BEARISH"
    amd_range_high: float = 0.0
    amd_range_low: float = 0.0
    amd_manipulation_level: float = 0.0
    amd_invalidation_stop: float = 0.0
    amd_target_1: float = 0.0
    amd_target_2: float = 0.0
    amd_confidence: int = 50
    amd_action: str = "STAND_ASIDE_ACCUMULATION"
    amd_description: str = ""


class MarketStructureEngine:
    """
    Evaluates live price series, VIX quotes, option chain delta, and Wyckoff/PO3 cycles
    to establish the dynamic market structure regime.
    """

    def __init__(self, broker: Optional[Any] = None) -> None:
        self.broker = broker
        self.last_oc_snapshot: Dict[str, pd.DataFrame] = {}
        from trade_system.domains.analysis.application.analysis.amd_phase_engine import AMDPhaseEngine
        self.amd_engine = AMDPhaseEngine()

    def analyze(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        oc_df: pd.DataFrame,
        vix_quote: Optional[Dict[str, Any]] = None,
        spot_price: Optional[float] = None,
        expiry_data: Optional[List[Dict[str, Any]]] = None,
    ) -> MarketStructureInfo:
        """Execute full market structure analysis."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Normalize Spot Price
        if spot_price is None or spot_price == 0.0:
            if not price_df.empty and "close" in price_df.columns:
                spot_price = float(price_df["close"].iloc[-1])
            elif not oc_df.empty and "strike" in oc_df.columns:
                spot_price = float(oc_df["strike"].mean())
            else:
                spot_price = 0.0

        # 1. Evaluate Price Action & Market Structure (BOS / Liquidity Sweeps)
        struct_state, struct_score, s_high, s_low, d_high, d_low, d_open = self._evaluate_price_structure(price_df, spot_price)

        # 2. Evaluate India VIX Dynamics & Divergence
        vix_level, vix_ch, vix_ch_pct, vix_regime, vix_div, vix_score = self._evaluate_vix(vix_quote, price_df, spot_price)

        # 3. Evaluate Option Chain Wall Migration & Order Flow
        ce_wall, pe_wall, wall_mig, pcr, atm_delta, taker_agg = self._evaluate_walls_and_migration(symbol, oc_df, spot_price)

        # 4. Evaluate Expiry Day Dynamics & 0DTE Risk Management
        dte, is_exp, is_eve, exp_phase, max_pain, dist_mp, exp_alert, rec_contract = self._evaluate_expiry_dynamics(
            symbol=symbol,
            oc_df=oc_df,
            spot_price=spot_price,
            expiry_data=expiry_data
        )

        # 5. Evaluate Wyckoff / Power of 3 (AMD) Market Cycle
        strike_step = 50.0
        if "BANKNIFTY" in symbol or "SENSEX" in symbol:
            strike_step = 100.0
        elif "MIDCPNIFTY" in symbol:
            strike_step = 25.0
        elif spot_price > 5000:
            strike_step = 50.0
        elif spot_price > 1000:
            strike_step = 20.0
        elif spot_price > 500:
            strike_step = 10.0
        else:
            strike_step = 5.0

        amd_info = self.amd_engine.analyze(
            price_df=price_df,
            spot_price=spot_price,
            strike_step=strike_step,
            atm_vol_delta=atm_delta,
        )

        return MarketStructureInfo(
            symbol=symbol,
            timestamp=now_str,
            spot_price=round(spot_price, 2),
            structure_state=struct_state,
            structure_score=struct_score,
            swing_high=round(s_high, 2),
            swing_low=round(s_low, 2),
            day_high=round(d_high, 2),
            day_low=round(d_low, 2),
            day_open=round(d_open, 2),
            vix_level=round(vix_level, 2),
            vix_change=round(vix_ch, 2),
            vix_change_pct=round(vix_ch_pct, 2),
            vix_regime=vix_regime,
            vix_divergence=vix_div,
            vix_score=vix_score,
            ce_wall=round(ce_wall, 2),
            pe_wall=round(pe_wall, 2),
            wall_migration=wall_mig,
            pcr_oi=round(pcr, 2),
            atm_volume_delta=atm_delta,
            taker_aggression=taker_agg,
            days_to_expiry=dte,
            is_expiry_day=is_exp,
            is_expiry_eve=is_eve,
            expiry_phase=exp_phase,
            max_pain_strike=round(max_pain, 2),
            dist_to_max_pain=round(dist_mp, 2),
            expiry_risk_alert=exp_alert,
            recommended_contract_type=rec_contract,
            amd_phase=amd_info.phase,
            amd_range_high=amd_info.range_high,
            amd_range_low=amd_info.range_low,
            amd_manipulation_level=amd_info.manipulation_level,
            amd_invalidation_stop=amd_info.invalidation_stop,
            amd_target_1=amd_info.target_1,
            amd_target_2=amd_info.target_2,
            amd_confidence=amd_info.confidence,
            amd_action=amd_info.recommended_action,
            amd_description=amd_info.description,
        )

    def _evaluate_expiry_dynamics(
        self,
        symbol: str,
        oc_df: pd.DataFrame,
        spot_price: float,
        expiry_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, bool, bool, str, float, float, str, str]:
        """
        Calculate DTE, Max Pain strike, and identify Expiry Day phases
        (e.g. Morning Pinning / Decay Trap vs Afternoon Gamma Squeeze).
        """
        now = datetime.now()
        today = date.today()
        days_to_expiry = 5

        # 1. Parse DTE from broker expiryData
        if expiry_data and len(expiry_data) > 0:
            first_exp_str = expiry_data[0].get("date", "")
            if first_exp_str:
                try:
                    target_date = datetime.strptime(first_exp_str, "%d-%m-%Y").date()
                    days_to_expiry = max(0, (target_date - today).days)
                except Exception:
                    pass

        # Fallback to weekday heuristic for major indices if not parsed
        if days_to_expiry > 7:
            # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
            weekday = today.weekday()
            if "NIFTY50" in symbol:
                days_to_expiry = (3 - weekday) % 7
            elif "BANKNIFTY" in symbol:
                days_to_expiry = (2 - weekday) % 7
            elif "FINNIFTY" in symbol:
                days_to_expiry = (1 - weekday) % 7
            elif "SENSEX" in symbol:
                days_to_expiry = (4 - weekday) % 7

        is_expiry_day = (days_to_expiry == 0)
        is_expiry_eve = (days_to_expiry == 1)

        # 2. Compute Max Pain strike
        max_pain = spot_price
        if oc_df is not None and not oc_df.empty and "strike" in oc_df.columns:
            try:
                strikes = sorted(oc_df["strike"].unique())
                ce_map = oc_df[oc_df["option_type"] == "CE"].set_index("strike")["oi"].to_dict()
                pe_map = oc_df[oc_df["option_type"] == "PE"].set_index("strike")["oi"].to_dict()
                losses = {}
                for s in strikes:
                    tot = 0
                    for k in strikes:
                        ce_oi = ce_map.get(k, 0)
                        pe_oi = pe_map.get(k, 0)
                        if s > k:
                            tot += ce_oi * (s - k)
                        elif s < k:
                            tot += pe_map.get(k, 0) * (k - s)
                    losses[s] = tot
                if losses:
                    max_pain = float(min(losses, key=losses.get))
            except Exception as e:
                LOGGER.warning("Max Pain calculation exception: %s", e)

        dist_to_max_pain = spot_price - max_pain

        # 3. Classify Expiry Phase & Actionable Contract Rule
        from datetime import time as dt_time
        curr_time = now.time()

        if is_expiry_day:
            if curr_time < dt_time(12, 30):
                phase = "PHASE 1: MORNING PINNING & THETA DECAY TRAP (09:15 - 12:30)"
                risk_alert = "⚠️ 0DTE THETA TRAP: Institutional writers aggressively defending straddles. Fading breakouts. Do NOT buy naked 0DTE ATM/OTM options."
                recommended_contract = "NEXT_WEEK_EXPIRY (or Deep ITM Delta >= 0.75)"
            elif curr_time < dt_time(13, 30):
                phase = "PHASE 2: MIDDAY STRADDLE BREAKEVEN TESTING (12:30 - 13:30)"
                risk_alert = "⚡ STRADDLE TESTING: Theta 80% decayed. Watch for delta expansion outside Call/Put walls."
                recommended_contract = "DEEP_ITM_DELTA_75"
            elif curr_time <= dt_time(15, 15):
                phase = "PHASE 3: AFTERNOON GAMMA SQUEEZE & UNWINDING WINDOW (13:30 - 15:15)"
                risk_alert = "🚀 GAMMA BLAST WINDOW: Institutional short covering triggers 100-300% explosive directional expansion if walls crack."
                recommended_contract = "0DTE_ATM_EXPIRING"
            else:
                phase = "PHASE 4: SETTLEMENT PINNING & CLOSE (15:15 - 15:30)"
                risk_alert = "🛑 SETTLEMENT PIN: High liquidity friction. Standing aside."
                recommended_contract = "STAND_ASIDE"
        elif is_expiry_eve:
            phase = "EXPIRY EVE (1DTE) — High Gamma Expansion Building"
            risk_alert = "⚡ 1DTE: Option premiums sensitive to overnight delta shifts. Maintain defined risk."
            recommended_contract = "CURRENT_EXPIRY_ATM"
        else:
            phase = f"REGULAR SESSION ({days_to_expiry} Days to Expiry)"
            risk_alert = "STANDARD TRADING: Normal delta/theta dynamics apply."
            recommended_contract = "CURRENT_EXPIRY_ATM"

        return days_to_expiry, is_expiry_day, is_expiry_eve, phase, max_pain, dist_to_max_pain, risk_alert, recommended_contract

    def _evaluate_price_structure(
        self,
        price_df: pd.DataFrame,
        spot_price: float
    ) -> Tuple[str, int, float, float, float, float, float]:
        """Detect Higher Highs/Lows, BOS, and Liquidity Sweeps."""
        if price_df is None or price_df.empty or len(price_df) < 5:
            return "CHOP_CONSOLIDATION", 0, spot_price, spot_price, spot_price, spot_price, spot_price

        df = price_df.sort_values("timestamp").copy()
        day_open = float(df["open"].iloc[0])
        day_high = float(df["high"].max())
        day_low = float(df["low"].min())

        # Swing pivots using rolling window
        lookback = min(10, len(df) - 1)
        recent = df.tail(lookback)
        swing_high = float(recent["high"].iloc[:-1].max())
        swing_low = float(recent["low"].iloc[:-1].min())

        latest_bar = df.iloc[-1]
        c = float(latest_bar["close"])
        o = float(latest_bar["open"])
        h = float(latest_bar["high"])
        l = float(latest_bar["low"])

        # Liquidity Sweep below swing low reclaimed with green close
        bull_sweep = (l < swing_low) and (c > swing_low) and (c > o)
        # Liquidity Sweep above swing high rejected with red close
        bear_sweep = (h > swing_high) and (c < swing_high) and (c < o)

        # Break of Structure (BOS): decisive close beyond swing pivot
        bull_bos = (c > swing_high) and (c > o)
        bear_bos = (c < swing_low) and (c < o)

        # Classify
        if bull_bos:
            return "BULLISH_BOS (Break of Structure Up)", 25, swing_high, swing_low, day_high, day_low, day_open
        elif bear_bos:
            return "BEARISH_BOS (Break of Structure Down)", -25, swing_high, swing_low, day_high, day_low, day_open
        elif bull_sweep:
            return "BULLISH_SWEEP_RECLAIM (Liquidity Grab)", 20, swing_high, swing_low, day_high, day_low, day_open
        elif bear_sweep:
            return "BEARISH_SWEEP_REJECT (Liquidity Grab)", -20, swing_high, swing_low, day_high, day_low, day_open
        elif c > day_open:
            return "BULLISH_INTRADAY_MOMENTUM", 10, swing_high, swing_low, day_high, day_low, day_open
        elif c < day_open:
            return "BEARISH_INTRADAY_MOMENTUM", -10, swing_high, swing_low, day_high, day_low, day_open
        else:
            return "CHOP_CONSOLIDATION", 0, swing_high, swing_low, day_high, day_low, day_open

    def _evaluate_vix(
        self,
        vix_quote: Optional[Dict[str, Any]],
        price_df: pd.DataFrame,
        spot_price: float
    ) -> Tuple[float, float, float, float, str, int]:
        """Evaluate India VIX regime and Price-VIX divergence."""
        if not vix_quote:
            return 12.0, 0.0, 0.0, "NORMAL (12-16)", "NONE", 0

        vix_level = float(vix_quote.get("lp", vix_quote.get("ltp", 12.0)))
        vix_ch = float(vix_quote.get("ch", 0.0))
        vix_ch_pct = float(vix_quote.get("chp", 0.0))

        # Regime
        if vix_level < 11.5:
            regime = "COMPLACENT (<11.5) — Low Volatility / Grind"
        elif vix_level <= 15.5:
            regime = "NORMAL (11.5-15.5) — Standard Liquidity"
        elif vix_level <= 20.0:
            regime = "ELEVATED (15.5-20.0) — High Directional Velocity"
        else:
            regime = "FEAR_SPIKE (>20.0) — Extreme Gamma Turbulence"

        # Divergence
        divergence = "NONE"
        score = 0
        if not price_df.empty and len(price_df) >= 3:
            price_change = float(price_df["close"].iloc[-1]) - float(price_df["open"].iloc[0])

            # Normal correlations: Price Up + VIX Down, Price Down + VIX Up
            if price_change > 0 and vix_ch_pct < -0.5:
                divergence = "HEALTHY_BULLISH_FLOW (Spot Up + VIX Down)"
                score = 15
            elif price_change < 0 and vix_ch_pct > 0.5:
                divergence = "HEALTHY_BEARISH_FLOW (Spot Down + VIX Up)"
                score = -15
            elif price_change > 0 and vix_ch_pct > 2.0:
                divergence = "⚠️ VOLATILITY_DIVERGENCE: Spot Up but VIX Surging (Institutional Hedging / Imminent Reversal Risk)"
                score = -10  # Penalize long
            elif price_change < 0 and vix_ch_pct < -2.0:
                divergence = "⚡ VOLATILITY_DIVERGENCE: Spot Down but VIX Crushing (Put Selling Exhaustion / Short Squeeze Imminent)"
                score = 10   # Favor bounce

        return vix_level, vix_ch, vix_ch_pct, regime, divergence, score

    def _evaluate_walls_and_migration(
        self,
        symbol: str,
        oc_df: pd.DataFrame,
        spot_price: float
    ) -> Tuple[float, float, str, float, int, str]:
        """Compute institutional walls and compare with prior snapshot to detect migrations."""
        if oc_df is None or oc_df.empty:
            return 0.0, 0.0, "STABLE", 1.0, 0, "NEUTRAL"

        df = oc_df.copy()
        if "strike" not in df.columns and "strike_price" in df.columns:
            df["strike"] = df["strike_price"]

        ce_df = df[df["option_type"] == "CE"]
        pe_df = df[df["option_type"] == "PE"]

        ce_wall = float(ce_df.loc[ce_df["oi"].idxmax()]["strike"]) if not ce_df.empty and ce_df["oi"].max() > 0 else spot_price + 100
        pe_wall = float(pe_df.loc[pe_df["oi"].idxmax()]["strike"]) if not pe_df.empty and pe_df["oi"].max() > 0 else spot_price - 100

        total_ce_oi = ce_df["oi"].sum()
        total_pe_oi = pe_df["oi"].sum()
        pcr = total_pe_oi / max(total_ce_oi, 1)

        # ATM Volume Delta
        all_strikes = sorted(set(ce_df["strike"].tolist() + pe_df["strike"].tolist()))
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if all_strikes else spot_price
        step = abs(all_strikes[1] - all_strikes[0]) if len(all_strikes) >= 2 else 50.0
        atm_cluster = [atm_strike - step, atm_strike, atm_strike + step]

        ce_atm_vol = ce_df[ce_df["strike"].isin(atm_cluster)]["volume"].sum()
        pe_atm_vol = pe_df[pe_df["strike"].isin(atm_cluster)]["volume"].sum()
        atm_delta = int(ce_atm_vol - pe_atm_vol)
        atm_ratio = ce_atm_vol / max(pe_atm_vol, 1)

        if atm_ratio >= 1.4:
            taker_agg = f"AGGRESSIVE_BULLISH ({atm_ratio:.1f}x Call Takers)"
        elif atm_ratio <= 0.7:
            taker_agg = f"AGGRESSIVE_BEARISH ({(1/max(atm_ratio, 0.01)):.1f}x Put Takers)"
        else:
            taker_agg = f"BALANCED ({atm_ratio:.2f}x)"

        # Migration tracking against previous snapshot
        migration = "STABLE"
        prev_df = self.last_oc_snapshot.get(symbol)
        if prev_df is not None and not prev_df.empty:
            prev_ce = prev_df[prev_df["option_type"] == "CE"]
            prev_pe = prev_df[prev_df["option_type"] == "PE"]
            prev_ce_wall = float(prev_ce.loc[prev_ce["oi"].idxmax()]["strike"]) if not prev_ce.empty else ce_wall
            prev_pe_wall = float(prev_pe.loc[prev_pe["oi"].idxmax()]["strike"]) if not prev_pe.empty else pe_wall

            if ce_wall < prev_ce_wall:
                migration = f"🔴 CE WALL LOWERING (₹{prev_ce_wall:.0f} -> ₹{ce_wall:.0f}): Institutional ceiling pressing down"
            elif pe_wall > prev_pe_wall:
                migration = f"🟢 PE WALL RISING (₹{prev_pe_wall:.0f} -> ₹{pe_wall:.0f}): Institutional floor stepping up"
            elif (ce_wall - pe_wall) < (prev_ce_wall - prev_pe_wall):
                migration = "⚡ WALLS COMPRESSING: Volatility Squeeze building up"

        # Cache this snapshot
        self.last_oc_snapshot[symbol] = df

        return ce_wall, pe_wall, migration, pcr, atm_delta, taker_agg
