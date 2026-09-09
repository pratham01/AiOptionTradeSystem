"""
ChiefTradingAgent — Autonomous Master Confluence Arbiter & Daily Trade Governor.

Acts as the highest-level decision engine for the trading system:
- Synthesizes 5 pillars:
    1. Price Action & Momentum (SuperTrend 3m/15m + VWAP) [25%]
    2. Option Chain Smart Money Positioning (4-Quadrant OI + PCR) [25%]
    3. Order Flow & Volume Taker Aggression (ATM Volume Delta + V/OI) [20%]
    4. Structural Boundaries & Liquidity Walls (CE/PE Walls + Max Pain) [15%]
    5. Institutional Divergence & Trap Filter (Zero-tolerance Veto) [15%]
- Enforces institutional risk management:
    - Maximum 2 to 3 sniper trades per day on Indices (NIFTY / BANKNIFTY / SENSEX).
    - Maximum 2 sniper trades per day on F&O Stocks.
    - Requires Confluence Score >= 80/100 and ZERO veto violations.
    - 45-minute cooldown between trades.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trade_system.domains.analysis.application.analysis.smart_oi_analyzer import SmartOIAnalyzer
from trade_system.domains.analysis.application.analysis.market_structure_engine import (
    MarketStructureEngine,
    MarketStructureInfo,
)
from trade_system.domains.strategy.application.indicators import calculate_supertrend
from trade_system.domains.trading.infrastructure.brokers.legacy import (
    FyersBrokerClient,
    FyersAuthService,
)
from trade_system.shared.config import Settings
from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)


@dataclass
class ChiefTradeSignal:
    """A qualified, institutional-grade sniper trade signal."""
    signal_id: str
    timestamp: str
    symbol: str
    asset_type: str  # "INDEX" or "STOCK"
    direction: str   # "BUY_CALL" or "BUY_PUT"
    strike_name: str
    option_type: str # "CE" or "PE"
    spot_entry_trigger: float
    spot_stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    confluence_score: int  # 0 to 100
    pillar_scores: Dict[str, int]
    primary_thesis: str
    veto_passed: bool
    status: str = "QUALIFIED_PENDING_TRIGGER"  # QUALIFIED, EXECUTED, CANCELLED


class ChiefTradingAgent:
    """
    Autonomous Chief Trading Agent enforcing daily trade limits and
    high-conviction multi-pillar confluence.
    """

    MAX_DAILY_INDEX_TRADES = 3
    MAX_DAILY_STOCK_TRADES = 2
    MIN_CONFLUENCE_THRESHOLD = 80
    COOLDOWN_MINUTES = 45

    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_file: Optional[Path] = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.auth_service = FyersAuthService(self.settings)
        self.broker: Optional[FyersBrokerClient] = None
        self.market_engine = MarketStructureEngine()
        self.state_file = state_file or (Path(__file__).resolve().parents[6] / "data" / "chief_agent_state.json")
        self._ensure_state_file()

    def _ensure_state_file(self) -> None:
        """Ensure state file exists with valid structure."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            initial_state = {
                "date": date.today().isoformat(),
                "index_trades_count": 0,
                "stock_trades_count": 0,
                "executed_trades": [],
                "last_index_trade_time": None,
                "last_stock_trade_time": None,
            }
            with open(self.state_file, "w") as f:
                json.dump(initial_state, f, indent=2)

    def _load_state(self) -> Dict[str, Any]:
        """Load state and reset counters if a new day has begun."""
        self._ensure_state_file()
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
        except Exception:
            state = {}

        today_str = date.today().isoformat()
        if state.get("date") != today_str:
            state = {
                "date": today_str,
                "index_trades_count": 0,
                "stock_trades_count": 0,
                "executed_trades": [],
                "last_index_trade_time": None,
                "last_stock_trade_time": None,
            }
            self._save_state(state)
        return state

    def _save_state(self, state: Dict[str, Any]) -> None:
        """Persist state to JSON file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            LOGGER.error("Failed to save Chief Agent state: %s", e)

    def get_broker(self) -> FyersBrokerClient:
        """Lazy-initialize authenticated broker client."""
        if self.broker is None:
            token = self.auth_service.get_valid_token()
            self.broker = FyersBrokerClient(
                client_id=self.settings.fyers.client_id,
                access_token=token,
                user_id=self.settings.fyers.user_id,
                authenticator=self.auth_service.authenticator,
            )
            self.broker.authenticate()
        return self.broker

    def check_daily_trade_budget(self, asset_type: str) -> Tuple[bool, int, int, str]:
        """
        Check if daily trade budget is available.
        Returns: (can_trade, trades_taken, max_allowed, reason)
        """
        state = self._load_state()
        now = datetime.now()

        if asset_type == "INDEX":
            count = state.get("index_trades_count", 0)
            max_trades = self.MAX_DAILY_INDEX_TRADES
            last_time_str = state.get("last_index_trade_time")
        else:
            count = state.get("stock_trades_count", 0)
            max_trades = self.MAX_DAILY_STOCK_TRADES
            last_time_str = state.get("last_stock_trade_time")

        if count >= max_trades:
            return False, count, max_trades, f"Daily limit reached ({count}/{max_trades} trades executed today). Capital protected."

        if last_time_str:
            try:
                last_time = datetime.fromisoformat(last_time_str)
                elapsed = (now - last_time).total_seconds() / 60.0
                if elapsed < self.COOLDOWN_MINUTES:
                    rem = int(self.COOLDOWN_MINUTES - elapsed)
                    return False, count, max_trades, f"Cooldown active ({rem} mins remaining from last trade). Standing aside."
            except Exception:
                pass

        return True, count, max_trades, f"Budget available ({count}/{max_trades} trades taken)."

    def record_signal_execution(self, signal: ChiefTradeSignal) -> None:
        """Record executed trade into daily ledger."""
        state = self._load_state()
        now_str = datetime.now().isoformat()

        if signal.asset_type == "INDEX":
            state["index_trades_count"] = state.get("index_trades_count", 0) + 1
            state["last_index_trade_time"] = now_str
        else:
            state["stock_trades_count"] = state.get("stock_trades_count", 0) + 1
            state["last_stock_trade_time"] = now_str

        state["executed_trades"].append(asdict(signal))
        self._save_state(state)
        LOGGER.info(
            "Chief Agent recorded trade for %s (%s). Daily count: %s",
            signal.symbol, signal.direction,
            state["index_trades_count"] if signal.asset_type == "INDEX" else state["stock_trades_count"]
        )

    def evaluate_symbol(
        self,
        symbol: str,
        strikecount: int = 15,
        force_evaluation: bool = False
    ) -> Tuple[Optional[ChiefTradeSignal], Dict[str, Any]]:
        """
        Master evaluation pipeline for any Index or F&O Stock.
        Returns: (QualifiedSignal or None, DiagnosticScorecard)
        """
        is_index = "INDEX" in symbol
        asset_type = "INDEX" if is_index else "STOCK"

        # Check daily budget
        can_trade, count, max_trades, budget_msg = self.check_daily_trade_budget(asset_type)
        if not can_trade and not force_evaluation:
            return None, {
                "symbol": symbol,
                "status": "BLOCKED_BY_GOVERNOR",
                "reason": budget_msg,
                "confluence_score": 0,
                "pillar_scores": {},
                "veto_reasons": [budget_msg]
            }

        broker = self.get_broker()

        # 1. Fetch Option Chain & Live Quotes
        try:
            res = broker.fyers.optionchain(data={"symbol": symbol, "strikecount": strikecount})
            if not isinstance(res, dict) or res.get("s") != "ok":
                return None, {"symbol": symbol, "status": "ERROR", "reason": "Failed to fetch live option chain."}
            oc_data = res.get("data", {})
            options_chain = oc_data.get("optionsChain", [])
            if not options_chain:
                return None, {"symbol": symbol, "status": "ERROR", "reason": "Option chain empty."}
            oc_df = pd.DataFrame(options_chain)
            if "strike" not in oc_df.columns and "strike_price" in oc_df.columns:
                oc_df["strike"] = oc_df["strike_price"]
        except Exception as e:
            LOGGER.error("Option chain fetch error for %s: %s", symbol, e)
            return None, {"symbol": symbol, "status": "ERROR", "reason": str(e)}

        # Extract Spot Price & Expiry
        spot_price = 0.0
        for item in options_chain:
            if item.get("underlying_value"):
                spot_price = float(item["underlying_value"])
                break

        # Fetch live quotes for spot and India VIX
        vix_quote = None
        try:
            quotes = broker.get_quotes([symbol, "NSE:INDIAVIX-INDEX"])
            if quotes:
                vix_quote = quotes.get("NSE:INDIAVIX-INDEX")
                if symbol in quotes:
                    val = quotes[symbol]
                    spot_price = float(val.get("lp", 0.0) if isinstance(val, dict) else getattr(val, "ltp", 0.0))
        except Exception as ex:
            LOGGER.warning("Quotes fetch warning for %s: %s", symbol, ex)

        expiry_date = str(options_chain[0].get("expiry_date", "")) if options_chain else ""

        # 2. Fetch Intraday 5m Price Candles
        today_str = date.today().strftime("%Y-%m-%d")
        try:
            price_df = broker.fetch_history(symbol, "5", today_str, today_str)
            if price_df is not None and not price_df.empty:
                price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], format="mixed")
                price_df = price_df.sort_values("timestamp").reset_index(drop=True)
            else:
                price_df = pd.DataFrame()
        except Exception:
            price_df = pd.DataFrame()

        # 3. Analyze Market Structure (BOS, Sweeps, VIX, Wall Migration, Expiry Day Dynamics)
        market_structure = self.market_engine.analyze(
            symbol=symbol,
            price_df=price_df,
            oc_df=oc_df,
            vix_quote=vix_quote,
            spot_price=spot_price,
            expiry_data=oc_data.get("expiryData", []),
        )

        # 4. Initialize SmartOI Analyzer & Detect Forensics
        analyzer = SmartOIAnalyzer(symbol)
        strike_step = analyzer.update_strike_step_from_df(oc_df)
        smart_analysis = analyzer.analyze_smart_oi(oc_df, None, spot_price=spot_price)
        divergences = analyzer.detect_institutional_divergences(price_df, oc_df, None, spot_price=spot_price)
        vol_profile = analyzer.compute_strike_volume_profile(oc_df, spot_price=spot_price)

        # 5. Compute 5-Pillar Confluence Score & Veto Flags
        scorecard = self._compute_confluence(
            symbol=symbol,
            spot_price=spot_price,
            strike_step=strike_step,
            price_df=price_df,
            oc_df=oc_df,
            smart_analysis=smart_analysis,
            divergences=divergences,
            vol_profile=vol_profile,
            is_index=is_index,
            market_structure=market_structure,
        )

        total_score = scorecard["total_confluence_score"]
        direction = scorecard["direction"]
        veto_reasons = scorecard["veto_reasons"]
        pillar_scores = scorecard["pillar_scores"]

        # Check qualification
        if total_score >= self.MIN_CONFLUENCE_THRESHOLD and not veto_reasons and direction != "NEUTRAL":
            atm_strike = scorecard["atm_strike"]
            ce_wall = scorecard["ce_wall"]
            pe_wall = scorecard["pe_wall"]

            if direction == "BUY_CALL":
                option_type = "CE"
                if market_structure and market_structure.is_expiry_day:
                    if "NEXT_WEEK" in market_structure.recommended_contract_type:
                        strike_name = f"{int(atm_strike)} CE [Next Week Expiry — Anti-Theta Shield]"
                    elif "DEEP_ITM" in market_structure.recommended_contract_type:
                        strike_name = f"{int(atm_strike - strike_step * 2)} CE [Deep ITM Delta 0.75 — Anti-Theta]"
                    else:
                        strike_name = f"{int(atm_strike)} CE [0DTE Gamma Blast]"
                else:
                    strike_name = f"{int(atm_strike)} CE"

                stop_loss = round(spot_price - (strike_step * 0.8), 2)
                t1 = round(ce_wall if ce_wall > spot_price else spot_price + (strike_step * 2), 2)
                t2 = round(t1 + (strike_step * 1.5), 2)

                # Structural AMD stop-loss & target optimization
                if market_structure and market_structure.amd_phase == "MANIPULATION_SPRING":
                    if 0 < market_structure.amd_invalidation_stop < spot_price:
                        stop_loss = market_structure.amd_invalidation_stop
                    if market_structure.amd_target_1 > spot_price:
                        t1 = market_structure.amd_target_1
                    if market_structure.amd_target_2 > t1:
                        t2 = market_structure.amd_target_2
                    rr = f"1:{((t1 - spot_price) / max(spot_price - stop_loss, 0.1)):.1f}"
                    thesis = (
                        f"⚡ WYCKOFF SPRING / AMD LIQUIDITY RECLAIM ({total_score}/100). Smart money swept retail sell-stops "
                        f"below ₹{market_structure.amd_range_low:,.1f} (Sweep Low: ₹{market_structure.amd_manipulation_level:,.1f}) "
                        f"and reclaimed the range. Ultra-tight invalidation at ₹{stop_loss} with asymmetric {rr} R:R targeting ₹{t1}."
                    )
                else:
                    rr = f"1:{((t1 - spot_price) / max(spot_price - stop_loss, 1)):.1f}"
                    thesis = (
                        f"Institutional Long Confluence ({total_score}/100). Supertrend/VWAP bullish alignment, "
                        f"positive ATM Volume Delta (+{divergences['atm_volume_delta']:,}), and zero trap vetoes. "
                        f"Targeting overhead resistance wall at ₹{t1}."
                    )
            else:
                option_type = "PE"
                if market_structure and market_structure.is_expiry_day:
                    if "NEXT_WEEK" in market_structure.recommended_contract_type:
                        strike_name = f"{int(atm_strike)} PE [Next Week Expiry — Anti-Theta Shield]"
                    elif "DEEP_ITM" in market_structure.recommended_contract_type:
                        strike_name = f"{int(atm_strike + strike_step * 2)} PE [Deep ITM Delta 0.75 — Anti-Theta]"
                    else:
                        strike_name = f"{int(atm_strike)} PE [0DTE Gamma Blast]"
                else:
                    strike_name = f"{int(atm_strike)} PE"

                stop_loss = round(spot_price + (strike_step * 0.8), 2)
                t1 = round(pe_wall if pe_wall < spot_price else spot_price - (strike_step * 2), 2)
                t2 = round(t1 - (strike_step * 1.5), 2)

                # Structural AMD stop-loss & target optimization
                if market_structure and market_structure.amd_phase == "MANIPULATION_UTAD":
                    if market_structure.amd_invalidation_stop > spot_price:
                        stop_loss = market_structure.amd_invalidation_stop
                    if 0 < market_structure.amd_target_1 < spot_price:
                        t1 = market_structure.amd_target_1
                    if 0 < market_structure.amd_target_2 < t1:
                        t2 = market_structure.amd_target_2
                    rr = f"1:{((spot_price - t1) / max(stop_loss - spot_price, 0.1)):.1f}"
                    thesis = (
                        f"⚡ WYCKOFF UTAD / AMD UPTHRUST REJECTION ({total_score}/100). Smart money trapped breakout buyers "
                        f"above ₹{market_structure.amd_range_high:,.1f} (Sweep High: ₹{market_structure.amd_manipulation_level:,.1f}) "
                        f"and rejected back down. Ultra-tight invalidation at ₹{stop_loss} with asymmetric {rr} R:R targeting ₹{t1}."
                    )
                else:
                    rr = f"1:{((spot_price - t1) / max(stop_loss - spot_price, 1)):.1f}"
                    thesis = (
                        f"Institutional Short Confluence ({total_score}/100). Supertrend/VWAP bearish breakdown, "
                        f"aggressive put buying / call writing flow, negative volume delta, and zero trap vetoes. "
                        f"Targeting support floor at ₹{t1}."
                    )

            signal = ChiefTradeSignal(
                signal_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol=symbol,
                asset_type=asset_type,
                direction=direction,
                strike_name=strike_name,
                option_type=option_type,
                spot_entry_trigger=round(spot_price, 2),
                spot_stop_loss=stop_loss,
                target_1=t1,
                target_2=t2,
                risk_reward=rr,
                confluence_score=total_score,
                pillar_scores=pillar_scores,
                primary_thesis=thesis,
                veto_passed=True,
            )
            scorecard["qualified_signal"] = signal
            return signal, scorecard

        scorecard["qualified_signal"] = None
        return None, scorecard

    def _compute_confluence(
        self,
        symbol: str,
        spot_price: float,
        strike_step: float,
        price_df: pd.DataFrame,
        oc_df: pd.DataFrame,
        smart_analysis: Dict[str, Any],
        divergences: Dict[str, Any],
        vol_profile: pd.DataFrame,
        is_index: bool,
        market_structure: Optional[MarketStructureInfo] = None,
    ) -> Dict[str, Any]:
        """Compute the strict 5-pillar confluence score and veto check."""
        bull_points = 0
        bear_points = 0
        veto_reasons = []

        # ── PILLAR 1: Price Action & Momentum Structure (Max 25 pts) ─────────
        p1_bull = 0
        p1_bear = 0

        # Market structure state (BOS & Sweeps)
        if market_structure:
            if "BULLISH_BOS" in market_structure.structure_state:
                p1_bull += 15
            elif "BULLISH_SWEEP_RECLAIM" in market_structure.structure_state:
                p1_bull += 12
            elif "BEARISH_BOS" in market_structure.structure_state:
                p1_bear += 15
            elif "BEARISH_SWEEP_REJECT" in market_structure.structure_state:
                p1_bear += 12

        if not price_df.empty and len(price_df) >= 7:
            # Supertrend 7, 3 on 5m
            st_df = calculate_supertrend(price_df, period=7, multiplier=3.0)
            latest_bar = st_df.iloc[-1]
            st_dir = latest_bar.get("supertrend_direction", 0)  # +1 Bullish, -1 Bearish

            if st_dir == 1:
                p1_bull += 8
            elif st_dir == -1:
                p1_bear += 8

            # VWAP check
            price_df["typical_price"] = (price_df["high"] + price_df["low"] + price_df["close"]) / 3
            price_df["tp_vol"] = price_df["typical_price"] * price_df["volume"]
            cum_vol = price_df["volume"].cumsum()
            cum_tp_vol = price_df["tp_vol"].cumsum()
            vwap = (cum_tp_vol / cum_vol.replace(0, 1)).iloc[-1]

            if spot_price > vwap:
                p1_bull += 5
            elif spot_price < vwap:
                p1_bear += 5

            # Price Range / Day Open Check
            open_price = price_df["open"].iloc[0]
            if spot_price > open_price:
                p1_bull += 3
            elif spot_price < open_price:
                p1_bear += 3
        else:
            p1_bull += 5
            p1_bear += 5

        # AMD (Accumulation, Manipulation, Distribution) Structure Confluence
        if market_structure:
            if market_structure.amd_phase == "MANIPULATION_SPRING":
                p1_bull += 15  # High-conviction Spring sweep reclaim
            elif market_structure.amd_phase == "MANIPULATION_UTAD":
                p1_bear += 15  # High-conviction UTAD upthrust rejection
            elif market_structure.amd_phase == "DISTRIBUTION_BULLISH":
                p1_bull += 8
            elif market_structure.amd_phase == "DISTRIBUTION_BEARISH":
                p1_bear += 8

        # ── PILLAR 2: Option Chain Smart Money Flow (Max 25 pts) ─────────────
        p2_bull = 0
        p2_bear = 0

        # Wall migration boost
        if market_structure:
            if "PE_WALL_RISING" in market_structure.wall_migration:
                p2_bull += 5
            elif "CE_WALL_LOWERING" in market_structure.wall_migration:
                p2_bear += 5

        ce_df = oc_df[oc_df["option_type"] == "CE"]
        pe_df = oc_df[oc_df["option_type"] == "PE"]

        total_ce_oi = ce_df["oi"].sum()
        total_pe_oi = pe_df["oi"].sum()
        pcr_oi = total_pe_oi / max(total_ce_oi, 1)

        summary = smart_analysis.get("summary", {})
        smart_signal = smart_analysis.get("signal", "NEUTRAL").upper()

        if "LONG BUILDUP" in smart_signal:
            p2_bull += 15
        elif "SHORT BUILDUP" in smart_signal:
            p2_bear += 15
        elif "SHORT COVERING" in smart_signal:
            p2_bull += 10
        elif "LONG UNWINDING" in smart_signal:
            p2_bear += 10

        # PCR Scoring
        if is_index:
            if 0.85 <= pcr_oi <= 1.25:
                p2_bull += 5
                p2_bear += 5
            elif pcr_oi > 1.25:
                p2_bull += 10  # Heavy put writing support
            elif pcr_oi < 0.70:
                p2_bear += 10  # Heavy call writing resistance
        else:
            # Equities
            if pcr_oi >= 0.80:
                p2_bull += 10
            elif pcr_oi <= 0.50:
                p2_bear += 10

        # Expiry Day Max Pain Gravitation & Pinning
        if market_structure and market_structure.is_expiry_day:
            dist_mp = market_structure.dist_to_max_pain
            if "PHASE 1" in market_structure.expiry_phase:
                # Morning Pinning: Price drawn magnetically to Max Pain
                if dist_mp > strike_step * 0.8:
                    p2_bear += 8  # Spot stretched above Max Pain -> gravity pull downward
                elif dist_mp < -strike_step * 0.8:
                    p2_bull += 8  # Spot stretched below Max Pain -> gravity pull upward
                elif abs(dist_mp) <= strike_step * 0.5:
                    # Tightly pinned right at Max Pain! High risk of chop
                    p1_bull = max(0, p1_bull - 8)
                    p1_bear = max(0, p1_bear - 8)
            elif "PHASE 3" in market_structure.expiry_phase:
                # Afternoon Gamma Blast: Once pin breaks, violent momentum
                if dist_mp > strike_step * 1.2:
                    p2_bull += 10
                elif dist_mp < -strike_step * 1.2:
                    p2_bear += 10

        # ── PILLAR 3: Order Flow & ATM Volume Delta (Max 20 pts) ─────────────
        p3_bull = 0
        p3_bear = 0

        atm_delta = divergences.get("atm_volume_delta", 0)
        atm_ratio = divergences.get("atm_volume_ratio", 1.0)
        aggression = divergences.get("taker_aggression", "")

        if "BULLISH" in aggression or atm_ratio >= 1.40:
            p3_bull += 15
        elif "BEARISH" in aggression or atm_ratio <= 0.70:
            p3_bear += 15
        else:
            p3_bull += 5
            p3_bear += 5

        # Volume Delta Sign
        if atm_delta > 0:
            p3_bull += 5
        elif atm_delta < 0:
            p3_bear += 5

        # ── PILLAR 4: Structural Boundaries & Liquidity Walls (Max 15 pts) ───
        p4_bull = 0
        p4_bear = 0

        all_strikes = sorted(set(ce_df["strike"].tolist() + pe_df["strike"].tolist()))
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if all_strikes else spot_price

        ce_wall = ce_df.loc[ce_df["oi"].idxmax()]["strike"] if not ce_df.empty and ce_df["oi"].max() > 0 else spot_price + strike_step * 5
        pe_wall = pe_df.loc[pe_df["oi"].idxmax()]["strike"] if not pe_df.empty and pe_df["oi"].max() > 0 else spot_price - strike_step * 5

        dist_to_ce = ce_wall - spot_price
        dist_to_pe = spot_price - pe_wall

        # Ensure room to run before hitting resistance wall
        min_headroom = strike_step * 1.5
        if dist_to_ce >= min_headroom:
            p4_bull += 10
        if dist_to_pe >= min_headroom:
            p4_bear += 10

        # Proximity to Support / Resistance bounce
        if dist_to_pe < min_headroom and spot_price >= pe_wall:
            p4_bull += 5  # Bouncing off primary support PE wall
        if dist_to_ce < min_headroom and spot_price <= ce_wall:
            p4_bear += 5  # Reversing from primary resistance CE wall

        # ── PILLAR 5: Institutional Divergence & Trap Veto Filter (Max 15 pts)
        p5_bull = 0
        p5_bear = 0

        trap_alerts = divergences.get("trap_alerts", [])
        div_list = divergences.get("divergences", [])

        # VETO CHECKS
        has_bull_trap = any("BULL TRAP" in a for a in trap_alerts)
        has_bear_trap = any("BEAR TRAP" in a for a in trap_alerts)
        has_exhaustion = any("Exhaustion" in d for d in div_list)

        # Bullish points if no traps and bullish divergences present
        if not has_bull_trap:
            p5_bull += 10
            if any("Bullish" in d for d in div_list):
                p5_bull += 5

        # Bearish points if no traps and bearish divergences present
        if not has_bear_trap:
            p5_bear += 10
            if any("Bearish" in d for d in div_list):
                p5_bear += 5

        # VIX Divergence check & Vetoes
        if market_structure:
            if market_structure.vix_score > 0:
                p5_bull += 5
            elif market_structure.vix_score < 0:
                p5_bear += 5

            if "VOLATILITY_DIVERGENCE: Spot Up but VIX Surging" in market_structure.vix_divergence:
                veto_reasons.append(f"VETO: Volatility Divergence! India VIX is surging (+{market_structure.vix_change_pct:.1f}%) while spot is rising. Institutional smart money is hedging downside risk.")
            elif "VOLATILITY_DIVERGENCE: Spot Down but VIX Crushing" in market_structure.vix_divergence:
                veto_reasons.append(f"VETO: Volatility Divergence! India VIX is collapsing (-{abs(market_structure.vix_change_pct):.1f}%) into sell-off. Indicates put selling exhaustion and imminent short squeeze.")

            # Expiry Day Pinning VETO
            if market_structure.is_expiry_day:
                if "PHASE 1" in market_structure.expiry_phase and abs(market_structure.dist_to_max_pain) <= strike_step * 0.4:
                    veto_reasons.append(f"VETO: 0DTE Morning Pinning Active! Spot is locked at Max Pain ₹{market_structure.max_pain_strike:.0f}. False breakouts & theta decay traps prevalent.")
                elif "PHASE 4" in market_structure.expiry_phase:
                    veto_reasons.append("VETO: Expiry Day Settlement Window (post-15:15). Standing aside.")

            # AMD Accumulation Breakout Trap VETO
            if market_structure.amd_phase == "ACCUMULATION":
                veto_reasons.append(
                    f"VETO: Wyckoff Phase 1 Accumulation active [₹{market_structure.amd_range_low:,.0f} - ₹{market_structure.amd_range_high:,.0f}]. "
                    f"Breakouts are false traps until the Manipulation sweep occurs."
                )

        # VETO APPLICATION
        total_bull = p1_bull + p2_bull + p3_bull + p4_bull + p5_bull
        total_bear = p1_bear + p2_bear + p3_bear + p4_bear + p5_bear

        if total_bull > total_bear and total_bull >= 65:
            direction = "BUY_CALL"
            final_score = total_bull
            pillar_scores = {
                "Price Action & Momentum": p1_bull,
                "Smart Money Flow": p2_bull,
                "Order Flow Aggression": p3_bull,
                "Structural Walls": p4_bull,
                "Divergence & Traps": p5_bull,
            }
            if has_bull_trap:
                veto_reasons.append("VETO: Bull Trap detected! Institutional writers dumping calls into resistance.")
            if has_exhaustion:
                veto_reasons.append("VETO: Volume Delta Exhaustion detected on recent high.")
            if dist_to_ce < strike_step:
                veto_reasons.append(f"VETO: Inadequate headroom to Resistance CE Wall (only {dist_to_ce:.1f} pts away).")
        elif total_bear > total_bull and total_bear >= 65:
            direction = "BUY_PUT"
            final_score = total_bear
            pillar_scores = {
                "Price Action & Momentum": p1_bear,
                "Smart Money Flow": p2_bear,
                "Order Flow Aggression": p3_bear,
                "Structural Walls": p4_bear,
                "Divergence & Traps": p5_bear,
            }
            if has_bear_trap:
                veto_reasons.append("VETO: Bear Trap detected! Institutional writers absorbing puts at support.")
            if dist_to_pe < strike_step:
                veto_reasons.append(f"VETO: Inadequate headroom to Support PE Wall (only {dist_to_pe:.1f} pts away).")
        else:
            direction = "NEUTRAL"
            final_score = max(total_bull, total_bear)
            pillar_scores = {
                "Price Action & Momentum": max(p1_bull, p1_bear),
                "Smart Money Flow": max(p2_bull, p2_bear),
                "Order Flow Aggression": max(p3_bull, p3_bear),
                "Structural Walls": max(p4_bull, p4_bear),
                "Divergence & Traps": max(p5_bull, p5_bear),
            }
            veto_reasons.append("Conflicting signals between Price Action and Smart Money Flow. High-conviction threshold not met.")

        return {
            "symbol": symbol,
            "direction": direction,
            "total_confluence_score": final_score,
            "pillar_scores": pillar_scores,
            "veto_reasons": veto_reasons,
            "atm_strike": atm_strike,
            "ce_wall": ce_wall,
            "pe_wall": pe_wall,
            "pcr_oi": round(pcr_oi, 2),
            "spot_price": round(spot_price, 2),
            "market_structure": asdict(market_structure) if market_structure else {},
        }

    def dispatch_telegram_alert(self, signal: ChiefTradeSignal) -> bool:
        """Send a pristine, high-priority Telegram alert for an approved sniper trade."""
        try:
            bot_token = self.settings.st_confirmed_telegram.bot_token or self.settings.telegram.bot_token
            chat_id = self.settings.st_confirmed_telegram.chat_id or self.settings.telegram.chat_id
            notifier = TelegramNotifier(bot_token, chat_id)

            dir_icon = "🟢" if "CALL" in signal.direction else "🔴"
            arrow = "📈 BUY CALL" if "CALL" in signal.direction else "📉 BUY PUT"

            message = (
                f"🎯 <b>CHIEF TRADING AGENT: SNIPER TRADE SIGNAL</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{dir_icon} <b>{signal.symbol}</b> — <b>{arrow}</b>\n"
                f"💎 <b>Contract:</b> <code>{signal.strike_name}</code>\n"
                f"🔥 <b>Confluence Score:</b> <b>{signal.confluence_score}/100</b> [HIGH CONVICTION]\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📍 <b>Spot Entry Trigger:</b> ₹{signal.spot_entry_trigger:,.2f}\n"
                f"🛑 <b>Spot Stop Loss:</b> ₹{signal.spot_stop_loss:,.2f}\n"
                f"🎯 <b>Target 1:</b> ₹{signal.target_1:,.2f}\n"
                f"🚀 <b>Target 2:</b> ₹{signal.target_2:,.2f}\n"
                f"⚖️ <b>Risk:Reward:</b> {signal.risk_reward}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <b>Pillars:</b>\n"
                f" • Price Action: {signal.pillar_scores.get('Price Action & Momentum', 0)}/25\n"
                f" • Smart Money Flow: {signal.pillar_scores.get('Smart Money Flow', 0)}/25\n"
                f" • Order Flow Aggression: {signal.pillar_scores.get('Order Flow Aggression', 0)}/20\n"
                f" • Structural Walls: {signal.pillar_scores.get('Structural Walls', 0)}/15\n"
                f" • Trap & Divergence Filter: {signal.pillar_scores.get('Divergence & Traps', 0)}/15\n"
                f"\n🧠 <i>Thesis: {signal.primary_thesis}</i>"
            )
            res = notifier.send(message)
            if res:
                self.record_signal_execution(signal)
            return bool(res)
        except Exception as e:
            LOGGER.error("Failed to dispatch Telegram alert: %s", e)
            return False

    def dispatch_max_pain_alert(
        self,
        symbol: str,
        new_max_pain: float,
        prev_max_pain: float,
        spot_price: float,
        market_structure: Dict[str, Any],
    ) -> bool:
        """Dispatch instant Telegram notification on option chain Max Pain level shift."""
        try:
            telegram_cfg = self.settings.telegram
            if not telegram_cfg.enabled:
                return False

            notifier = TelegramNotifier(telegram_cfg.bot_token, telegram_cfg.chat_id)
            short_sym = symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")
            pts_diff = new_max_pain - prev_max_pain
            is_bullish = pts_diff > 0
            direction_badge = "📈 Bullish (Floor Migrated Upward)" if is_bullish else "📉 Bearish (Ceiling Lowered Downward)"
            diff_sign = f"+{pts_diff:,.1f}" if is_bullish else f"{pts_diff:,.1f}"

            dist = spot_price - new_max_pain
            dist_sign = f"+{dist:,.1f}" if dist >= 0 else f"{dist:,.1f}"
            dte = market_structure.get("days_to_expiry", "N/A")
            phase = market_structure.get("expiry_phase", "ACTIVE")

            if is_bullish:
                implication = (
                    f"• <b>Smart Money Writers:</b> Heavy Put writing / Call short covering has pushed the strike of minimum writer payout HIGHER.\n"
                    f"• <b>Market Structure:</b> Underlying price floor has risen. Pullbacks toward ₹{new_max_pain:,.0f} likely to find institutional support & magnet pull.\n"
                    f"• <b>Option Action:</b> Bullish bias. Favorable for ATM Call buying on dips."
                )
            else:
                implication = (
                    f"• <b>Smart Money Writers:</b> Heavy Call writing / Put long buildup has pushed the strike of minimum writer payout LOWER.\n"
                    f"• <b>Market Structure:</b> Price ceiling is lowering. Bounces toward ₹{new_max_pain:,.0f} face heavy institutional resistance.\n"
                    f"• <b>Option Action:</b> Bearish bias. Favorable for ATM Put buying on rallies."
                )

            message = (
                f"🧲 <b>MAX PAIN LEVEL SHIFT — {short_sym}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>New Max Pain Strike:</b> <b>₹{new_max_pain:,.1f}</b>\n"
                f"⏮️ <b>Previous Level:</b> ₹{prev_max_pain:,.1f}\n"
                f"📊 <b>Shift Delta:</b> <b>{diff_sign} pts</b> ({direction_badge})\n"
                f"📌 <b>Current Spot:</b> ₹{spot_price:,.2f} ({dist_sign} pts from Max Pain)\n"
                f"⏳ <b>Expiry Dynamics:</b> {dte} DTE ({phase})\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <b>Institutional Forensics:</b>\n"
                f"{implication}\n\n"
                f"⏰ <i>Time: {datetime.now().strftime('%H:%M:%S IST')}</i>"
            )
            return bool(notifier.send(message))
        except Exception as e:
            LOGGER.error("Failed to dispatch Max Pain alert for %s: %s", symbol, e)
            return False


class ContinuousChiefAgent:
    """
    Continuous Autonomous Chief Trading Agent daemon that runs during market hours,
    tracks dynamic market structure, VIX, option chain, and dispatches high-conviction
    sniper trades to Telegram.
    """

    def __init__(
        self,
        agent: Optional[ChiefTradingAgent] = None,
        live_state_file: Optional[Path] = None,
    ) -> None:
        self.agent = agent or ChiefTradingAgent()
        self.live_state_file = live_state_file or (Path(__file__).resolve().parents[6] / "data" / "chief_agent_live_state.json")
        self.last_max_pain: Dict[str, float] = {}
        self.is_running = False

    def is_market_hours(self) -> bool:
        """Check if current time is within Indian market hours (09:15 to 15:30 IST on weekdays)."""
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        from datetime import time as dt_time
        return dt_time(9, 15) <= now.time() <= dt_time(15, 30)

    def run_cycle(
        self,
        symbols: Optional[List[str]] = None,
        dispatch_telegram: bool = True,
    ) -> Dict[str, Any]:
        """Run a single autonomous evaluation cycle across primary symbols."""
        target_symbols = symbols or ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
        cycle_results: Dict[str, Any] = {}
        qualified_signals: List[Dict[str, Any]] = []

        for sym in target_symbols:
            try:
                sig, scorecard = self.agent.evaluate_symbol(sym, strikecount=15)
                cycle_results[sym] = scorecard

                # Check for Max Pain shift
                ms = scorecard.get("market_structure", {})
                spot = scorecard.get("spot_price", 0.0)
                new_mp = ms.get("max_pain_strike", 0.0)
                if new_mp > 0:
                    prev_mp = self.last_max_pain.get(sym)
                    if prev_mp is not None and prev_mp > 0 and new_mp != prev_mp:
                        if dispatch_telegram:
                            LOGGER.info("Continuous Chief Agent: Max Pain shifted for %s: %s -> %s", sym, prev_mp, new_mp)
                            self.agent.dispatch_max_pain_alert(sym, new_mp, prev_mp, spot, ms)
                    self.last_max_pain[sym] = new_mp

                if sig:
                    qualified_signals.append(asdict(sig))
                    if dispatch_telegram:
                        LOGGER.info("Continuous Chief Agent: Dispatched sniper trade for %s!", sym)
                        self.agent.dispatch_telegram_alert(sig)
            except Exception as e:
                LOGGER.error("Error in continuous cycle for %s: %s", sym, e)

        # Extract VIX details from cycle
        vix_info: Dict[str, Any] = {}
        for sym, sc in cycle_results.items():
            ms = sc.get("market_structure", {})
            if ms:
                vix_info = {
                    "vix_level": ms.get("vix_level", 0.0),
                    "vix_change": ms.get("vix_change", 0.0),
                    "vix_change_pct": ms.get("vix_change_pct", 0.0),
                    "vix_regime": ms.get("vix_regime", "NORMAL"),
                    "vix_divergence": ms.get("vix_divergence", "NONE"),
                }
                break

        state_payload = {
            "timestamp": datetime.now().isoformat(),
            "is_running": True,
            "market_hours": self.is_market_hours(),
            "vix": vix_info,
            "indices_status": cycle_results,
            "qualified_signals": qualified_signals,
            "daily_budget": self.agent._load_state(),
        }

        try:
            self.live_state_file.parent.mkdir(parents=True, exist_ok=True)
            def _json_default(obj: Any) -> Any:
                if isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, (np.floating, float)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray, list)):
                    return list(obj)
                elif hasattr(obj, "isoformat"):
                    return obj.isoformat()
                return str(obj)

            with open(self.live_state_file, "w") as f:
                json.dump(state_payload, f, indent=2, default=_json_default)
        except Exception as ex:
            LOGGER.error("Failed to write continuous live state file: %s", ex)

        return state_payload

    def run_forever(
        self,
        interval_seconds: int = 120,
        dispatch_telegram: bool = True,
    ) -> None:
        """Run continuous autonomous monitoring loop."""
        self.is_running = True
        LOGGER.info("🚀 Starting Continuous Chief Trading Agent daemon (interval: %ss)...", interval_seconds)

        import time
        while self.is_running:
            try:
                in_hours = self.is_market_hours()
                if not in_hours:
                    LOGGER.info("Outside market hours. Standing by and taking maintenance snapshot...")
                    self.run_cycle(dispatch_telegram=False)
                    time.sleep(300)
                    continue

                LOGGER.info("Evaluating live market structure and confluence across indices...")
                self.run_cycle(dispatch_telegram=dispatch_telegram)
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                LOGGER.info("Stopping Continuous Chief Agent daemon...")
                self.is_running = False
                break
            except Exception as e:
                LOGGER.exception("Unexpected error in Continuous Chief Agent loop: %s", e)
                time.sleep(60)

