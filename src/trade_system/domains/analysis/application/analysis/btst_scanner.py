"""
BTST Institutional Scanner — Buy Today, Sell Tomorrow (BTST) & STBT Engine.

Decodes the critical 15:00 – 15:30 IST market microstructure window:
1. Last 30-Minute Volume Concentration vs Daily Average
2. Price Proximity to Day's High (Closing at highs)
3. Open Interest (OI) Quadrant Forensics (Long Build-Up vs Short Covering Trap)
4. Option Chain Support & Resistance Walls
5. Algorithmic BTST Conviction Score (0 - 100) & Actionable Setups
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trade_system.domains.market_data.infrastructure.data.fo_universe import (
    get_fo_universe,
    get_sector_mapping,
)
from trade_system.domains.trading.infrastructure.brokers.legacy import (
    FyersBrokerClient,
    FyersAuthService,
)
from trade_system.shared.config import Settings
from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)


def _normalize_quote(q: Any) -> Dict[str, Any]:
    """Convert either a raw dict or a MarketQuote dataclass into a uniform dict."""
    if not q:
        return {}
    if isinstance(q, dict):
        lp = float(q.get("lp", 0.0) or q.get("ltp", 0.0) or q.get("last_price", 0.0))
        op = float(q.get("open_price", 0.0) or q.get("open", lp))
        hp = float(q.get("high_price", 0.0) or q.get("high", lp))
        low = float(q.get("low_price", 0.0) or q.get("low", lp))
        prev = float(q.get("prev_close_price", 0.0) or q.get("previous_close", lp))
        ch = float(q.get("ch", 0.0) or q.get("change", lp - prev))
        chp = float(q.get("chp", 0.0) or q.get("change_percent", ((lp - prev) / prev * 100) if prev else 0.0))
        vol = int(q.get("v", 0) or q.get("volume", 0))
    elif hasattr(q, "last_price") or hasattr(q, "ltp"):
        lp = float(getattr(q, "last_price", 0.0) or getattr(q, "ltp", 0.0))
        op = float(getattr(q, "open", lp))
        hp = float(getattr(q, "high", lp))
        low = float(getattr(q, "low", lp))
        prev = float(getattr(q, "previous_close", lp))
        ch = float(getattr(q, "change", lp - prev))
        chp = float(getattr(q, "change_percent", ((lp - prev) / prev * 100) if prev else 0.0))
        vol = int(getattr(q, "volume", 0))
    else:
        return {}

    return {
        "lp": lp,
        "open_price": op,
        "high_price": hp,
        "low_price": low,
        "prev_close_price": prev,
        "ch": ch,
        "chp": chp,
        "v": vol,
    }


@dataclass
class BTSTCandidate:
    """Represents a scored BTST / STBT candidate."""
    symbol: str
    clean_symbol: str
    sector: str
    spot_price: float
    day_open: float
    day_high: float
    day_low: float
    prev_close: float
    day_change: float
    day_change_pct: float
    dist_to_day_high_pct: float
    
    # Volume Shockwave Metrics
    total_day_volume: int
    last_30m_volume: int
    vol_30m_ratio_pct: float       # % of day's volume in last 30m
    vol_burst_multiplier: float    # vs typical 30m baseline (8% of day)
    vwap: float
    price_vs_vwap_pct: float

    # Open Interest & Option Chain Metrics
    oi_quadrant: str               # "LONG_BUILDUP", "SHORT_COVERING", "SHORT_BUILDUP", "LONG_UNWINDING", "NEUTRAL"
    total_ce_oi: int
    total_pe_oi: int
    net_ce_oich: int
    net_pe_oich: int
    pcr_oi: float
    immediate_call_wall: float
    immediate_put_wall: float

    # Algorithmic Scoring (0 - 100)
    price_action_score: int        # Max 25
    volume_shockwave_score: int    # Max 25
    oi_flow_score: int             # Max 25
    trend_score: int               # Max 25
    btst_score: int                # Total 0 - 100

    # Classification & Action Plan
    verdict: str                   # "STRONG_BTST", "MODERATE_BTST", "BTST_TRAP_SHORT_COVERING", "STBT_CANDIDATE", "NEUTRAL"
    verdict_badge_color: str
    is_trap: bool
    trap_warning: str
    recommended_entry: str
    stop_loss: str
    target_1: str
    target_2: str
    risk_reward: str
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "clean_symbol": self.clean_symbol,
            "sector": self.sector,
            "spot_price": round(self.spot_price, 2),
            "day_high": round(self.day_high, 2),
            "day_low": round(self.day_low, 2),
            "day_change_pct": round(self.day_change_pct, 2),
            "dist_to_day_high_pct": round(self.dist_to_day_high_pct, 2),
            "vol_30m_ratio_pct": round(self.vol_30m_ratio_pct, 1),
            "vol_burst_multiplier": round(self.vol_burst_multiplier, 2),
            "vwap": round(self.vwap, 2),
            "oi_quadrant": self.oi_quadrant,
            "pcr_oi": round(self.pcr_oi, 2),
            "btst_score": self.btst_score,
            "verdict": self.verdict,
            "is_trap": self.is_trap,
            "trap_warning": self.trap_warning,
            "recommended_entry": self.recommended_entry,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "risk_reward": self.risk_reward,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


@dataclass
class BTSTScanResult:
    """Container for complete universe scan results."""
    scan_time: str
    total_scanned: int
    top_picks: List[BTSTCandidate]
    traps: List[BTSTCandidate]
    stbt_picks: List[BTSTCandidate]
    all_results: List[BTSTCandidate]


class BTSTInstitutionalScanner:
    """
    Scans the F&O Universe for Institutional BTST (Buy Today, Sell Tomorrow)
    and STBT (Sell Today, Buy Tomorrow) setups based on 15:00-15:30 volume & OI forensics.
    """

    def __init__(self, broker: Optional[FyersBrokerClient] = None) -> None:
        self.broker = broker
        self.sector_map = get_sector_mapping()

    def _ensure_broker(self) -> FyersBrokerClient:
        if self.broker is None or self.broker.fyers is None:
            settings = Settings.load()
            auth = FyersAuthService(settings)
            token = auth.get_valid_token()
            self.broker = FyersBrokerClient(
                client_id=settings.fyers.client_id,
                access_token=token,
                user_id=settings.fyers.user_id,
                authenticator=auth.authenticator,
            )
            self.broker.authenticate()
        return self.broker

    # ------------------------------------------------------------------
    # Data Acquisition Helpers
    # ------------------------------------------------------------------

    def fetch_candidate_candles(self, symbol: str, resolution: str = "5") -> pd.DataFrame:
        """Fetch today's 5-minute candles to analyze the 15:00-15:30 window."""
        try:
            broker = self._ensure_broker()
            today_str = date.today().strftime("%Y-%m-%d")
            df = broker.fetch_history(
                symbol=symbol,
                resolution=resolution,
                range_from=today_str,
                range_to=today_str,
            )
            if df is not None and not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df.sort_values("timestamp").reset_index(drop=True)
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.debug("Error fetching candles for %s: %s", symbol, exc)
            return pd.DataFrame()

    def fetch_candidate_option_chain(self, symbol: str, strikecount: int = 10) -> Optional[pd.DataFrame]:
        """Fetch option chain for a candidate stock."""
        try:
            broker = self._ensure_broker()
            res = broker.fyers.optionchain(data={"symbol": symbol, "strikecount": strikecount})
            if not isinstance(res, dict) or res.get("s") != "ok":
                return None
            oc = res.get("data", {}).get("optionsChain", [])
            if not oc:
                return None
            df = pd.DataFrame(oc)
            if "strike_price" in df.columns:
                df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce")
            return df
        except Exception as exc:
            LOGGER.debug("Error fetching option chain for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Core Algorithmic Analyzer
    # ------------------------------------------------------------------

    def analyze_single_stock(
        self,
        symbol: str,
        quote: Optional[Dict[str, Any]] = None,
        candles_df: Optional[pd.DataFrame] = None,
        option_chain_df: Optional[pd.DataFrame] = None,
    ) -> Optional[BTSTCandidate]:
        """
        Conduct deep 15:00-15:30 forensics on a single F&O stock.
        """
        clean_sym = symbol.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-INDEX", "")
        sector = self.sector_map.get(symbol, "GENERAL")

        # 1. Price metrics from quote
        q_norm = _normalize_quote(quote) if quote else {}
        if q_norm:
            spot = q_norm["lp"]
            day_open = q_norm["open_price"]
            day_high = q_norm["high_price"]
            day_low = q_norm["low_price"]
            prev_close = q_norm["prev_close_price"]
            day_chg = q_norm["ch"]
            day_chg_pct = q_norm["chp"]
            total_vol = q_norm["v"]
        else:
            spot = day_open = day_high = day_low = prev_close = day_chg = day_chg_pct = 0.0
            total_vol = 0

        if spot <= 0.0:
            return None

        # 2. 5-minute Candles & 15:00-15:30 Volume Shockwave
        if candles_df is None or candles_df.empty:
            candles_df = self.fetch_candidate_candles(symbol, resolution="5")

        if not candles_df.empty:
            # Fallback for high/low/open from candles if quote missed them
            if day_high == 0.0 or day_high < spot:
                day_high = float(candles_df["high"].max())
            if day_low == 0.0 or day_low > spot:
                day_low = float(candles_df["low"].min())
            if day_open == 0.0:
                day_open = float(candles_df["open"].iloc[0])
            if total_vol == 0:
                total_vol = int(candles_df["volume"].sum())

            # Calculate session VWAP
            tp = (candles_df["high"] + candles_df["low"] + candles_df["close"]) / 3
            cum_vol = candles_df["volume"].cumsum()
            cum_tp_vol = (tp * candles_df["volume"]).cumsum()
            vwap = float((cum_tp_vol / cum_vol.replace(0, 1)).iloc[-1])

            # Filter for last 30 minutes (15:00 - 15:30 IST)
            # Find the most recent date in candles
            target_date = candles_df["timestamp"].dt.date.max()
            day_candles = candles_df[candles_df["timestamp"].dt.date == target_date]
            last_30m_df = day_candles[day_candles["timestamp"].dt.time >= dtime(15, 0)]

            if not last_30m_df.empty:
                last_30m_vol = int(last_30m_df["volume"].sum())
            else:
                # If market is still mid-session (<15:00), take last 6 candles (last 30m of trading)
                last_30m_vol = int(day_candles["volume"].tail(6).sum())
        else:
            vwap = spot
            last_30m_vol = int(total_vol * 0.10)  # Approximation fallback

        # Calculate volume metrics
        actual_total_vol = max(total_vol, last_30m_vol, 1)
        vol_30m_ratio = (last_30m_vol / actual_total_vol) * 100.0
        # Typical 30m baseline is 8% of total volume in a 375m day
        vol_burst_multiplier = vol_30m_ratio / 8.0

        # Proximity to Day's High
        effective_high = max(day_high, spot)
        dist_to_high_pct = ((effective_high - spot) / effective_high) * 100.0 if effective_high > 0 else 0.0
        price_vs_vwap_pct = ((spot - vwap) / vwap) * 100.0 if vwap > 0 else 0.0

        # 3. Option Chain & OI Quadrant Analysis
        if option_chain_df is None or option_chain_df.empty:
            option_chain_df = self.fetch_candidate_option_chain(symbol)

        ce_oi = pe_oi = ce_oich = pe_oich = 0
        call_wall = put_wall = 0.0

        if option_chain_df is not None and not option_chain_df.empty:
            ce_df = option_chain_df[option_chain_df["option_type"].astype(str).str.upper() == "CE"]
            pe_df = option_chain_df[option_chain_df["option_type"].astype(str).str.upper() == "PE"]

            ce_oi = int(pd.to_numeric(ce_df.get("oi", 0), errors="coerce").fillna(0).sum())
            pe_oi = int(pd.to_numeric(pe_df.get("oi", 0), errors="coerce").fillna(0).sum())
            ce_oich = int(pd.to_numeric(ce_df.get("oich", 0), errors="coerce").fillna(0).sum())
            pe_oich = int(pd.to_numeric(pe_df.get("oich", 0), errors="coerce").fillna(0).sum())

            if not ce_df.empty and "strike" in ce_df.columns:
                call_wall = float(ce_df.loc[ce_df["oi"].idxmax()]["strike"]) if "oi" in ce_df.columns and not ce_df["oi"].empty else spot * 1.03
            if not pe_df.empty and "strike" in pe_df.columns:
                put_wall = float(pe_df.loc[pe_df["oi"].idxmax()]["strike"]) if "oi" in pe_df.columns and not pe_df["oi"].empty else spot * 0.97

        pcr_oi = (pe_oi / ce_oi) if ce_oi > 0 else 1.0

        # Determine Institutional OI Quadrant
        is_price_up = day_chg_pct >= 0.20
        is_price_down = day_chg_pct <= -0.20

        if is_price_up:
            if pe_oich > 0 and (ce_oich <= 0 or pe_oich >= ce_oich * 1.2):
                oi_quadrant = "LONG_BUILDUP"
            elif ce_oich < 0 and pe_oich <= 0:
                oi_quadrant = "SHORT_COVERING"
            else:
                oi_quadrant = "LONG_BUILDUP" if pe_oich >= 0 else "SHORT_COVERING"
        elif is_price_down:
            if ce_oich > 0 and (pe_oich <= 0 or ce_oich >= pe_oich * 1.2):
                oi_quadrant = "SHORT_BUILDUP"
            elif pe_oich < 0 and ce_oich <= 0:
                oi_quadrant = "LONG_UNWINDING"
            else:
                oi_quadrant = "SHORT_BUILDUP"
        else:
            oi_quadrant = "NEUTRAL"

        # 4. Algorithmic Scoring (0 to 100)
        # -------------------------------------------------------------
        # Pillar A: Price Action Score (Max 25)
        pa_score = 0
        if dist_to_high_pct <= 0.40:
            pa_score = 25
        elif dist_to_high_pct <= 0.80:
            pa_score = 20
        elif dist_to_high_pct <= 1.40:
            pa_score = 14
        elif dist_to_high_pct <= 2.00:
            pa_score = 8
        else:
            pa_score = 2

        if spot >= vwap:
            pa_score = min(25, pa_score + 3)

        # Pillar B: Volume Shockwave Score (Max 25)
        vol_score = 0
        if vol_30m_ratio >= 24.0:
            vol_score = 25
        elif vol_30m_ratio >= 18.0:
            vol_score = 20
        elif vol_30m_ratio >= 13.0:
            vol_score = 14
        elif vol_30m_ratio >= 9.0:
            vol_score = 8
        else:
            vol_score = 3

        # Pillar C: Institutional OI Flow Score (Max 25)
        oi_score = 0
        is_trap = False
        trap_msg = ""

        if is_price_up:
            if oi_quadrant == "LONG_BUILDUP":
                oi_score = 25
                if pcr_oi >= 0.70:
                    oi_score = min(25, oi_score + 3)
            elif oi_quadrant == "SHORT_COVERING":
                # CRITICAL: Short covering trap! Zero points.
                oi_score = 0
                is_trap = True
                trap_msg = "⚠️ SHORT COVERING TRAP: Price surged into the close, but Call/Futures OI collapsed. Buying is from intraday bears exiting, NOT fresh institutional overnight accumulation."
        elif is_price_down:
            if oi_quadrant == "SHORT_BUILDUP":
                oi_score = 20  # Good for STBT

        # Pillar D: Trend & Momentum Continuum (Max 25)
        trend_score = 0
        if 1.2 <= day_chg_pct <= 4.0:
            trend_score = 25  # Sweet spot for sustainable BTST
        elif 4.0 < day_chg_pct <= 7.0:
            trend_score = 18  # Strong, but slight gap-down profit-taking risk
        elif 0.5 <= day_chg_pct < 1.2:
            trend_score = 14
        elif day_chg_pct > 7.0:
            trend_score = 10  # Over-extended
        else:
            trend_score = 4

        if spot > day_open:
            trend_score = min(25, trend_score + 3)

        btst_total_score = pa_score + vol_score + oi_score + trend_score

        # 5. Verdict Classification
        if is_trap:
            verdict = "BTST_TRAP_SHORT_COVERING"
            badge_color = "#ff4d6d"
        elif btst_total_score >= 75 and oi_quadrant == "LONG_BUILDUP":
            verdict = "STRONG_BTST_ACCUMULATION"
            badge_color = "#00d084"
        elif btst_total_score >= 60 and oi_quadrant == "LONG_BUILDUP":
            verdict = "MODERATE_BTST"
            badge_color = "#00e6ff"
        elif is_price_down and oi_quadrant == "SHORT_BUILDUP" and vol_30m_ratio >= 15.0:
            verdict = "STBT_CANDIDATE"
            badge_color = "#f77f00"
        else:
            verdict = "NEUTRAL_MONITOR"
            badge_color = "#8b949e"

        # 6. Actionable Trade Parameters
        dist_to_low_pct = ((spot - day_low) / day_low * 100.0) if day_low > 0 else 0.0

        if verdict == "STBT_CANDIDATE":
            entry_zone = f"₹{spot:,.1f} — ₹{spot * 0.996:,.1f} (Sell before 15:25 IST)"
            sl_val = round(min(vwap * 1.008, spot * 1.015), 1)
            t1_val = round(spot * 0.982, 1)
            t2_val = round(spot * 0.968, 1)
            rr_ratio = f"1 : {round((spot - t1_val) / max(sl_val - spot, 0.1), 1)}"
            rationale = (
                f"{clean_sym} broke down to within {dist_to_low_pct:.1f}% of Day Low with {vol_30m_ratio:.1f}% "
                f"of daily volume packed into the last 30m ({vol_burst_multiplier:.1f}x burst). "
                f"OI confirms {oi_quadrant.replace('_', ' ')} with PCR at {pcr_oi:.2f}."
            )
        else:
            entry_zone = f"₹{spot:,.1f} — ₹{spot * 1.004:,.1f} (Buy before 15:25 IST)"
            sl_val = round(max(vwap * 0.992, spot * 0.985), 1)
            t1_val = round(spot * 1.018, 1)
            t2_val = round(spot * 1.032, 1)
            rr_ratio = f"1 : {round((t1_val - spot) / max(spot - sl_val, 0.1), 1)}"
            rationale = (
                f"{clean_sym} closed within {dist_to_high_pct:.1f}% of Day High with {vol_30m_ratio:.1f}% "
                f"of daily volume packed into the last 30m ({vol_burst_multiplier:.1f}x burst). "
                f"OI confirms {oi_quadrant.replace('_', ' ')} with PCR at {pcr_oi:.2f}."
            )

        return BTSTCandidate(
            symbol=symbol,
            clean_symbol=clean_sym,
            sector=sector,
            spot_price=spot,
            day_open=day_open,
            day_high=effective_high,
            day_low=day_low,
            prev_close=prev_close,
            day_change=day_chg,
            day_change_pct=day_chg_pct,
            dist_to_day_high_pct=dist_to_high_pct,
            total_day_volume=actual_total_vol,
            last_30m_volume=last_30m_vol,
            vol_30m_ratio_pct=vol_30m_ratio,
            vol_burst_multiplier=vol_burst_multiplier,
            vwap=vwap,
            price_vs_vwap_pct=price_vs_vwap_pct,
            oi_quadrant=oi_quadrant,
            total_ce_oi=ce_oi,
            total_pe_oi=pe_oi,
            net_ce_oich=ce_oich,
            net_pe_oich=pe_oich,
            pcr_oi=pcr_oi,
            immediate_call_wall=call_wall,
            immediate_put_wall=put_wall,
            price_action_score=pa_score,
            volume_shockwave_score=vol_score,
            oi_flow_score=oi_score,
            trend_score=trend_score,
            btst_score=btst_total_score,
            verdict=verdict,
            verdict_badge_color=badge_color,
            is_trap=is_trap,
            trap_warning=trap_msg,
            recommended_entry=entry_zone,
            stop_loss=f"₹{sl_val:,.1f}",
            target_1=f"₹{t1_val:,.1f} (At 09:20 AM)",
            target_2=f"₹{t2_val:,.1f}",
            risk_reward=rr_ratio,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Universe Scanner Pipeline
    # ------------------------------------------------------------------

    def scan_btst(
        self,
        symbols: Optional[List[str]] = None,
        universe: Optional[List[str]] = None,
        top_candidates_limit: int = 30,
        progress_callback: Optional[Any] = None,
    ) -> BTSTScanResult:
        """
        Execute two-tier BTST universe scan:
        Tier 1: Fast batch quotes to filter top momentum candidates.
        Tier 2: Deep 5m candle volume & option chain OI forensics.
        """
        target_universe = symbols or universe or get_fo_universe()
        LOGGER.info("Starting BTST Universe scan across %d F&O stocks...", len(target_universe))
        broker = self._ensure_broker()

        # Step 1: Tier 1 Fast Quotes Fetch (Batches of 50)
        quotes_map: Dict[str, Dict[str, Any]] = {}
        for chunk_start in range(0, len(target_universe), 50):
            chunk = target_universe[chunk_start : chunk_start + 50]
            try:
                q_res = broker.get_quotes(chunk)
                if q_res:
                    for k, v in q_res.items():
                        norm = _normalize_quote(v)
                        if norm:
                            quotes_map[k] = norm
            except Exception as exc:
                LOGGER.warning("Quotes chunk failed: %s", exc)
            time.sleep(0.1)

        # Step 2: Filter Tier 1 Candidates
        # Prioritize: Day gainers (>0.5%) close to day high, plus top day losers for STBT
        tier1_candidates: List[Tuple[str, Dict[str, Any]]] = []
        for sym in target_universe:
            q = quotes_map.get(sym, {})
            if not q:
                continue
            chp = float(q.get("chp", 0.0))
            lp = float(q.get("lp", 0.0))
            hp = float(q.get("high_price", lp))
            dist_high = ((hp - lp) / hp * 100) if hp > 0 else 999.0

            # Filter criterion: Positive momentum near highs OR strong breakdown
            if (chp >= 0.4 and dist_high <= 2.5) or (chp <= -1.5):
                tier1_candidates.append((sym, q))

        # Sort Tier 1: highest change % near highs
        tier1_candidates.sort(key=lambda item: float(item[1].get("chp", 0.0)), reverse=True)
        tier1_selected = tier1_candidates[:top_candidates_limit]

        LOGGER.info(
            "Tier 1 filtered %d momentum candidates for deep forensics.",
            len(tier1_selected),
        )

        # Step 3: Tier 2 Deep Forensics
        candidates: List[BTSTCandidate] = []
        total_tier2 = len(tier1_selected)

        for idx, (sym, q) in enumerate(tier1_selected, 1):
            cand = self.analyze_single_stock(sym, quote=q)
            if cand:
                candidates.append(cand)
            if progress_callback:
                progress_callback(idx / max(1, total_tier2), f"Analyzing {sym} ({idx}/{total_tier2})...")

        # Segregate into Top BTST, Traps, and STBT
        top_picks = [c for c in candidates if "BTST" in c.verdict and not c.is_trap]
        top_picks.sort(key=lambda c: c.btst_score, reverse=True)

        traps = [c for c in candidates if c.is_trap]
        traps.sort(key=lambda c: c.day_change_pct, reverse=True)

        stbt = [c for c in candidates if c.verdict == "STBT_CANDIDATE"]
        stbt.sort(key=lambda c: c.day_change_pct, reverse=False)

        candidates.sort(key=lambda c: c.btst_score, reverse=True)

        return BTSTScanResult(
            scan_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_scanned=len(target_universe),
            top_picks=top_picks,
            traps=traps,
            stbt_picks=stbt,
            all_results=candidates,
        )

    # ------------------------------------------------------------------
    # Telegram Alert Generator & Dispatcher
    # ------------------------------------------------------------------

    def format_telegram_alert(self, candidate: BTSTCandidate) -> str:
        """Format high-conviction BTST candidate for Telegram alert."""
        icon = "🔥" if candidate.btst_score >= 80 else "⚡"
        return (
            f"{icon} <b>INSTITUTIONAL BTST ALERT: {candidate.clean_symbol}</b>\n\n"
            f"• <b>Sector:</b> {candidate.sector}\n"
            f"• <b>Spot LTP:</b> ₹{candidate.spot_price:,.2f} ({candidate.day_change_pct:+.2f}%)\n"
            f"• <b>BTST Conviction Score:</b> <b>{candidate.btst_score}/100</b>\n"
            f"• <b>Last 30m Volume:</b> {candidate.vol_30m_ratio_pct:.1f}% of Day's Volume ({candidate.vol_burst_multiplier:.1f}x surge)\n"
            f"• <b>OI Alignment:</b> {candidate.oi_quadrant.replace('_', ' ')} (PCR: {candidate.pcr_oi:.2f})\n"
            f"• <b>Day High Proximity:</b> {candidate.dist_to_day_high_pct:.2f}% from High\n\n"
            f"🎯 <b>Actionable Setup:</b>\n"
            f"• <b>Entry:</b> {candidate.recommended_entry}\n"
            f"• <b>Stop Loss:</b> {candidate.stop_loss}\n"
            f"• <b>Target 1:</b> {candidate.target_1}\n"
            f"• <b>Risk-Reward:</b> {candidate.risk_reward}\n\n"
            f"💡 <i>{candidate.rationale}</i>\n"
            f"⏰ <i>Scan Time: {candidate.timestamp}</i>"
        )

    def dispatch_telegram_alerts(self, top_candidates: List[BTSTCandidate], max_alerts: int = 3) -> int:
        """Send formatted Telegram alerts for top BTST picks."""
        notifier = TelegramNotifier()
        sent_count = 0
        for cand in top_candidates[:max_alerts]:
            msg = self.format_telegram_alert(cand)
            if notifier.send_message(msg):
                sent_count += 1
                time.sleep(0.5)
        return sent_count
