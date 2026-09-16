"""
OptionChainMonitorAgent — Specialized AI Agent for real-time dynamic Option Chain differential forensics.

Monitors snapshot-to-snapshot changes across:
1. Strike-level Delta OI Velocity & Acceleration (dOI/dt)
2. Max Pain Migration Drift Vector
3. Dynamic Call/Put Wall Shifts
4. PCR Velocity (dPCR/dt)
5. Institutional Writer Traps (Bull Trap, Bear Trap) vs Real Short Squeezes / Long Liquidations
6. Real-time Telegram alerting & State persistence for Dashboard
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from trade_system.shared.config import Settings
from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger("OptionChainMonitorAgent")
IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class OptionChainSnapshotMeta:
    timestamp: datetime
    spot_price: float
    atm_strike: float
    max_pain: float
    ce_wall: float
    pe_wall: float
    ce_wall_oi: int
    pe_wall_oi: int
    pcr_oi: float
    pcr_vol: float
    total_ce_oi: int
    total_pe_oi: int
    atm_vol_delta: int
    strikes_data: Dict[float, Dict[str, Any]]  # {strike: {"CE_oi": x, "PE_oi": y, "CE_vol": a, "PE_vol": b}}


@dataclass
class DynamicForensicResult:
    symbol: str
    timestamp: datetime
    spot_price: float
    atm_strike: float
    max_pain: float
    prev_max_pain: float
    max_pain_shifted: bool
    max_pain_shift_pts: float
    ce_wall: float
    pe_wall: float
    ce_wall_shifted: bool
    pe_wall_shifted: bool
    pcr_oi: float
    pcr_velocity: float  # dPCR/dt (per min)
    atm_vol_ratio: float
    top_ce_build_strike: float
    top_ce_build_oi: int
    top_pe_build_strike: float
    top_pe_build_oi: int
    top_ce_unwind_strike: float
    top_ce_unwind_oi: int
    top_pe_unwind_strike: float
    top_pe_unwind_oi: int
    regime: str
    traps: List[str]
    squeezes: List[str]
    signals: List[Dict[str, Any]]


class OptionChainMonitorAgent:
    """
    Autonomous, real-time agent dedicated to continuous Option Chain Data Change analysis.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        notifier: TelegramNotifier | None = None,
        state_file: Path | None = None,
        enable_telegram: bool = True,
        max_history_len: int = 40,
    ) -> None:
        self.settings = settings or Settings.load()
        
        # Configure Telegram Notifier (defaults to primary personal bot)
        if notifier:
            self.notifier = notifier
        else:
            token = self.settings.telegram.bot_token
            chat_id = self.settings.telegram.chat_id
            self.notifier = TelegramNotifier(token, chat_id) if token and chat_id else None

        self.enable_telegram = enable_telegram
        self.state_file = state_file or (self.settings.data_dir / "option_chain_monitor_state.json")
        self.max_history_len = max_history_len

        # In-memory history per symbol: deque of OptionChainSnapshotMeta
        self.history: Dict[str, deque] = {}
        # Tracking debounce timestamps: {key: timestamp}
        self.alert_debounce: Dict[str, datetime] = {}
        # Last known states
        self.latest_results: Dict[str, DynamicForensicResult] = {}

    def clean_symbol(self, symbol: str) -> str:
        return symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "").replace("-EQ", "")

    def process_snapshot(
        self,
        symbol: str,
        oc_df: pd.DataFrame,
        spot_price: float,
        timestamp: datetime | None = None,
        expiry: str | None = None,
    ) -> Optional[DynamicForensicResult]:
        """
        Process a new snapshot dataframe, compare against historical snapshots,
        and generate differential metrics, trap alerts, and trade setups.
        """
        if oc_df is None or oc_df.empty:
            return None

        now = timestamp or datetime.now(IST)
        clean_sym = self.clean_symbol(symbol)

        # Standardize columns
        df = oc_df.copy()
        if "strike_price" in df.columns and "strike" not in df.columns:
            df["strike"] = df["strike_price"]
        if "open_interest" in df.columns and "oi" not in df.columns:
            df["oi"] = df["open_interest"]

        df["strike"] = pd.to_numeric(df.get("strike", 0.0), errors="coerce").fillna(0.0)
        df["oi"] = pd.to_numeric(df.get("oi", 0), errors="coerce").fillna(0).astype(int)
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
        df["ltp"] = pd.to_numeric(df.get("ltp", 0.0), errors="coerce").fillna(0.0)
        df["option_type"] = df.get("option_type", "").astype(str).str.upper()

        ce_df = df[df["option_type"] == "CE"].sort_values("strike").copy()
        pe_df = df[df["option_type"] == "PE"].sort_values("strike").copy()

        if ce_df.empty or pe_df.empty:
            return None

        total_ce_oi = int(ce_df["oi"].sum())
        total_pe_oi = int(pe_df["oi"].sum())
        total_ce_vol = int(ce_df["volume"].sum())
        total_pe_vol = int(pe_df["volume"].sum())

        pcr_oi = float(total_pe_oi / max(total_ce_oi, 1))
        pcr_vol = float(total_pe_vol / max(total_ce_vol, 1))

        # Walls
        ce_wall_row = ce_df.loc[ce_df["oi"].idxmax()] if ce_df["oi"].max() > 0 else None
        pe_wall_row = pe_df.loc[pe_df["oi"].idxmax()] if pe_df["oi"].max() > 0 else None

        ce_wall = float(ce_wall_row["strike"]) if ce_wall_row is not None else 0.0
        ce_wall_oi = int(ce_wall_row["oi"]) if ce_wall_row is not None else 0
        pe_wall = float(pe_wall_row["strike"]) if pe_wall_row is not None else 0.0
        pe_wall_oi = int(pe_wall_row["oi"]) if pe_wall_row is not None else 0

        # Max Pain
        all_strikes = sorted(set(ce_df["strike"].tolist() + pe_df["strike"].tolist()))
        max_pain = self._calculate_max_pain(ce_df, pe_df, all_strikes)

        # ATM Strike
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if all_strikes and spot_price > 0 else spot_price
        strike_step = self._infer_strike_step(all_strikes)

        # ATM Cluster Volume Delta
        atm_cluster = [atm_strike - strike_step, atm_strike, atm_strike + strike_step]
        ce_atm_vol = ce_df[ce_df["strike"].isin(atm_cluster)]["volume"].sum()
        pe_atm_vol = pe_df[pe_df["strike"].isin(atm_cluster)]["volume"].sum()
        atm_vol_delta = int(ce_atm_vol - pe_atm_vol)
        atm_vol_ratio = float(ce_atm_vol / max(pe_atm_vol, 1))

        # Build strike map
        strikes_data: Dict[float, Dict[str, Any]] = {}
        for s in all_strikes:
            ce_s = ce_df[ce_df["strike"] == s]
            pe_s = pe_df[pe_df["strike"] == s]
            strikes_data[s] = {
                "CE_oi": int(ce_s["oi"].iloc[0]) if not ce_s.empty else 0,
                "PE_oi": int(pe_s["oi"].iloc[0]) if not pe_s.empty else 0,
                "CE_vol": int(ce_s["volume"].iloc[0]) if not ce_s.empty else 0,
                "PE_vol": int(pe_s["volume"].iloc[0]) if not pe_s.empty else 0,
                "CE_ltp": float(ce_s["ltp"].iloc[0]) if not ce_s.empty else 0.0,
                "PE_ltp": float(pe_s["ltp"].iloc[0]) if not pe_s.empty else 0.0,
            }

        curr_snap = OptionChainSnapshotMeta(
            timestamp=now,
            spot_price=spot_price,
            atm_strike=atm_strike,
            max_pain=max_pain,
            ce_wall=ce_wall,
            pe_wall=pe_wall,
            ce_wall_oi=ce_wall_oi,
            pe_wall_oi=pe_wall_oi,
            pcr_oi=pcr_oi,
            pcr_vol=pcr_vol,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            atm_vol_delta=atm_vol_delta,
            strikes_data=strikes_data,
        )

        if clean_sym not in self.history:
            self.history[clean_sym] = deque(maxlen=self.max_history_len)

        history_queue = self.history[clean_sym]
        prev_snap: Optional[OptionChainSnapshotMeta] = history_queue[-1] if history_queue else None

        # ── Cross-Snapshot Differential Analysis ─────────────────────────────
        max_pain_shifted = False
        max_pain_shift_pts = 0.0
        prev_max_pain = max_pain

        ce_wall_shifted = False
        pe_wall_shifted = False
        pcr_velocity = 0.0

        top_ce_build_s = 0.0
        top_ce_build_oi = 0
        top_pe_build_s = 0.0
        top_pe_build_oi = 0

        top_ce_unwind_s = 0.0
        top_ce_unwind_oi = 0
        top_pe_unwind_s = 0.0
        top_pe_unwind_oi = 0

        traps: List[str] = []
        squeezes: List[str] = []
        signals: List[Dict[str, Any]] = []

        if prev_snap is not None:
            dt_minutes = max((curr_snap.timestamp - prev_snap.timestamp).total_seconds() / 60.0, 0.5)

            # 1. Max Pain Migration
            if prev_snap.max_pain > 0 and curr_snap.max_pain != prev_snap.max_pain:
                max_pain_shifted = True
                max_pain_shift_pts = curr_snap.max_pain - prev_snap.max_pain
                prev_max_pain = prev_snap.max_pain

            # 2. Wall Shifts
            if prev_snap.ce_wall > 0 and curr_snap.ce_wall != prev_snap.ce_wall:
                ce_wall_shifted = True
            if prev_snap.pe_wall > 0 and curr_snap.pe_wall != prev_snap.pe_wall:
                pe_wall_shifted = True

            # 3. PCR Velocity (rate of change per minute)
            pcr_velocity = (curr_snap.pcr_oi - prev_snap.pcr_oi) / dt_minutes

            # 4. Strike-by-Strike ΔOI calculation
            ce_deltas: List[Tuple[float, int]] = []
            pe_deltas: List[Tuple[float, int]] = []

            for s, curr_vals in curr_snap.strikes_data.items():
                if s in prev_snap.strikes_data:
                    prev_vals = prev_snap.strikes_data[s]
                    d_ce = curr_vals["CE_oi"] - prev_vals["CE_oi"]
                    d_pe = curr_vals["PE_oi"] - prev_vals["PE_oi"]
                    ce_deltas.append((s, d_ce))
                    pe_deltas.append((s, d_pe))

            if ce_deltas:
                ce_deltas_sorted = sorted(ce_deltas, key=lambda x: x[1], reverse=True)
                top_ce_build_s, top_ce_build_oi = ce_deltas_sorted[0]
                top_ce_unwind_s, top_ce_unwind_oi = ce_deltas_sorted[-1]

            if pe_deltas:
                pe_deltas_sorted = sorted(pe_deltas, key=lambda x: x[1], reverse=True)
                top_pe_build_s, top_pe_build_oi = pe_deltas_sorted[0]
                top_pe_unwind_s, top_pe_unwind_oi = pe_deltas_sorted[-1]

            # 5. Trap & Squeeze Detection
            min_defense_qty = 8000 if "BANKNIFTY" in symbol else 12000

            # Bull Trap: Spot at or above CE Wall, but CE OI adds heavily (institutions defending/absorbing)
            if spot_price >= (curr_snap.ce_wall - strike_step * 0.25):
                ce_wall_delta = curr_snap.strikes_data.get(curr_snap.ce_wall, {}).get("CE_oi", 0) - \
                                prev_snap.strikes_data.get(curr_snap.ce_wall, {}).get("CE_oi", 0)
                if ce_wall_delta >= min_defense_qty:
                    traps.append(
                        f"🚨 <b>BULL TRAP ALERT:</b> Spot (₹{spot_price:,.1f}) testing Resistance CE Wall (₹{curr_snap.ce_wall:,.0f}), "
                        f"but Call writers added +{ce_wall_delta:,} contracts! High risk of rejection."
                    )
                elif ce_wall_delta <= -min_defense_qty:
                    squeezes.append(
                        f"🚀 <b>SHORT SQUEEZE ALERT:</b> Spot breached CE Wall (₹{curr_snap.ce_wall:,.0f}) "
                        f"with massive Call panic unwinding ({ce_wall_delta:,} contracts). Strong continuation expected!"
                    )

            # Bear Trap: Spot at or below PE Wall, but PE OI adds heavily (institutions absorbing sells)
            if spot_price <= (curr_snap.pe_wall + strike_step * 0.25):
                pe_wall_delta = curr_snap.strikes_data.get(curr_snap.pe_wall, {}).get("PE_oi", 0) - \
                                prev_snap.strikes_data.get(curr_snap.pe_wall, {}).get("PE_oi", 0)
                if pe_wall_delta >= min_defense_qty:
                    traps.append(
                        f"🪤 <b>BEAR TRAP ALERT:</b> Spot (₹{spot_price:,.1f}) testing Support PE Wall (₹{curr_snap.pe_wall:,.0f}), "
                        f"but Put writers added +{pe_wall_delta:,} contracts! High risk of bounce."
                    )
                elif pe_wall_delta <= -min_defense_qty:
                    squeezes.append(
                        f"🩸 <b>LONG LIQUIDATION ALERT:</b> Spot broke Support PE Wall (₹{curr_snap.pe_wall:,.0f}) "
                        f"with Put panic unwinding ({pe_wall_delta:,} contracts). Downward cascade active!"
                    )

            # 6. Actionable Trade Setup Generation
            if max_pain_shifted and abs(max_pain_shift_pts) >= strike_step:
                if max_pain_shift_pts > 0:
                    signals.append({
                        "setup": "MAX_PAIN_UPWARD_MIGRATION",
                        "bias": "BULLISH",
                        "action": "BUY_CALL",
                        "contract": f"{int(curr_snap.atm_strike)} CE",
                        "entry": f"Pullback towards ₹{prev_max_pain:,.1f}",
                        "stop_loss": f"₹{prev_max_pain - strike_step:,.1f}",
                        "target_1": f"₹{curr_snap.max_pain:,.1f}",
                        "target_2": f"₹{curr_snap.ce_wall:,.1f}",
                        "confidence": 85,
                        "rationale": f"Max Pain jumped +{max_pain_shift_pts:.0f} pts from ₹{prev_max_pain:.0f} to ₹{curr_snap.max_pain:.0f}. Institutional writers lifted floor.",
                    })
                else:
                    signals.append({
                        "setup": "MAX_PAIN_DOWNWARD_MIGRATION",
                        "bias": "BEARISH",
                        "action": "BUY_PUT",
                        "contract": f"{int(curr_snap.atm_strike)} PE",
                        "entry": f"Bounce towards ₹{prev_max_pain:,.1f}",
                        "stop_loss": f"₹{prev_max_pain + strike_step:,.1f}",
                        "target_1": f"₹{curr_snap.max_pain:,.1f}",
                        "target_2": f"₹{curr_snap.pe_wall:,.1f}",
                        "confidence": 85,
                        "rationale": f"Max Pain dropped {max_pain_shift_pts:.0f} pts from ₹{prev_max_pain:.0f} to ₹{curr_snap.max_pain:.0f}. Institutional writers depressed ceiling.",
                    })

            if squeezes and "SHORT SQUEEZE" in squeezes[0]:
                signals.append({
                    "setup": "SHORT_SQUEEZE_BREAKOUT",
                    "bias": "BULLISH",
                    "action": "BUY_CALL",
                    "contract": f"{int(curr_snap.atm_strike)} CE",
                    "entry": f"Above ₹{curr_snap.ce_wall:,.1f}",
                    "stop_loss": f"₹{curr_snap.ce_wall - strike_step * 0.5:,.1f}",
                    "target_1": f"₹{curr_snap.ce_wall + strike_step:,.1f}",
                    "target_2": f"₹{curr_snap.ce_wall + strike_step * 2:,.1f}",
                    "confidence": 88,
                    "rationale": "CE Wall breached with confirmed panic Call covering.",
                })

            if traps and "BEAR TRAP" in traps[0] and pcr_velocity > 0.005:
                signals.append({
                    "setup": "BEAR_TRAP_REVERSAL",
                    "bias": "BULLISH",
                    "action": "BUY_CALL",
                    "contract": f"{int(curr_snap.atm_strike)} CE",
                    "entry": f"Reclaim above ₹{curr_snap.pe_wall:,.1f}",
                    "stop_loss": f"₹{spot_price - strike_step * 0.3:,.1f}",
                    "target_1": f"₹{curr_snap.max_pain:,.1f}",
                    "target_2": f"₹{curr_snap.ce_wall:,.1f}",
                    "confidence": 82,
                    "rationale": "Support broken but smart money put writing absorbed retail panic with positive PCR velocity.",
                })

            if traps and "BULL TRAP" in traps[0] and pcr_velocity < -0.005:
                signals.append({
                    "setup": "BULL_TRAP_REVERSAL",
                    "bias": "BEARISH",
                    "action": "BUY_PUT",
                    "contract": f"{int(curr_snap.atm_strike)} PE",
                    "entry": f"Rejection below ₹{curr_snap.ce_wall:,.1f}",
                    "stop_loss": f"₹{spot_price + strike_step * 0.3:,.1f}",
                    "target_1": f"₹{curr_snap.max_pain:,.1f}",
                    "target_2": f"₹{curr_snap.pe_wall:,.1f}",
                    "confidence": 82,
                    "rationale": "Resistance tested but smart money call writing absorbed retail breakout with negative PCR velocity.",
                })

        # Append to history queue
        history_queue.append(curr_snap)

        # Market Regime
        if curr_snap.pcr_oi >= 1.3:
            regime = "OVERBOUGHT_PUT_HEAVY"
        elif curr_snap.pcr_oi <= 0.7:
            regime = "OVERSOLD_CALL_HEAVY"
        else:
            regime = "BALANCED"

        result = DynamicForensicResult(
            symbol=symbol,
            timestamp=now,
            spot_price=spot_price,
            atm_strike=atm_strike,
            max_pain=max_pain,
            prev_max_pain=prev_max_pain,
            max_pain_shifted=max_pain_shifted,
            max_pain_shift_pts=max_pain_shift_pts,
            ce_wall=ce_wall,
            pe_wall=pe_wall,
            ce_wall_shifted=ce_wall_shifted,
            pe_wall_shifted=pe_wall_shifted,
            pcr_oi=pcr_oi,
            pcr_velocity=pcr_velocity,
            atm_vol_ratio=atm_vol_ratio,
            top_ce_build_strike=top_ce_build_s,
            top_ce_build_oi=top_ce_build_oi,
            top_pe_build_strike=top_pe_build_s,
            top_pe_build_oi=top_pe_build_oi,
            top_ce_unwind_strike=top_ce_unwind_s,
            top_ce_unwind_oi=top_ce_unwind_oi,
            top_pe_unwind_strike=top_pe_unwind_s,
            top_pe_unwind_oi=top_pe_unwind_oi,
            regime=regime,
            traps=traps,
            squeezes=squeezes,
            signals=signals,
        )

        self.latest_results[clean_sym] = result

        # Alert dispatch
        if self.enable_telegram and self.notifier:
            self._dispatch_alerts_if_needed(clean_sym, result, expiry)

        # Persist state
        self._persist_state()

        return result

    def _calculate_max_pain(self, ce_df: pd.DataFrame, pe_df: pd.DataFrame, strikes: List[float]) -> float:
        """Calculate the strike price where total writer payout is minimized."""
        if not strikes:
            return 0.0
        ce_map = dict(zip(ce_df["strike"], ce_df["oi"]))
        pe_map = dict(zip(pe_df["strike"], pe_df["oi"]))

        min_loss = float("inf")
        max_pain_strike = strikes[0]

        for s_candidate in strikes:
            loss = 0.0
            for k, oi_val in ce_map.items():
                if s_candidate > k:
                    loss += (s_candidate - k) * oi_val
            for k, oi_val in pe_map.items():
                if s_candidate < k:
                    loss += (k - s_candidate) * oi_val
            if loss < min_loss:
                min_loss = loss
                max_pain_strike = s_candidate

        return float(max_pain_strike)

    def _infer_strike_step(self, strikes: List[float]) -> float:
        if len(strikes) < 2:
            return 50.0
        diffs = [abs(strikes[i+1] - strikes[i]) for i in range(len(strikes)-1)]
        valid_diffs = [d for d in diffs if d > 0]
        return float(min(valid_diffs) if valid_diffs else 50.0)

    def _debounce(self, key: str, window_seconds: int = 180) -> bool:
        """Returns True if allowed to fire, False if debounced."""
        now = datetime.now()
        last_time = self.alert_debounce.get(key)
        if last_time and (now - last_time).total_seconds() < window_seconds:
            return False
        self.alert_debounce[key] = now
        return True

    def _dispatch_alerts_if_needed(self, symbol: str, result: DynamicForensicResult, expiry: str | None) -> None:
        """Dispatches prioritized alerts for Max Pain shifts, Traps, and Squeezes."""
        if not self.notifier:
            return

        # 1. Max Pain Shift Alert
        if result.max_pain_shifted and abs(result.max_pain_shift_pts) > 0:
            key = f"{symbol}_MAX_PAIN_{result.max_pain}"
            if self._debounce(key, window_seconds=60):
                is_bullish = result.max_pain_shift_pts > 0
                badge = "📈 BULLISH (Floor Lifted)" if is_bullish else "📉 BEARISH (Ceiling Lowered)"
                sign = f"+{result.max_pain_shift_pts:,.0f}" if is_bullish else f"{result.max_pain_shift_pts:,.0f}"

                implication = (
                    f"• <b>Smart Money Intent:</b> Heavy Put accumulation / Call unwinding has shifted the minimum writer payout level <b>HIGHER</b>.\n"
                    f"• <b>Institutional Anchor:</b> Dips to ₹{result.prev_max_pain:,.0f} offer strong support. Favorable for ATM Call buying."
                    if is_bullish else
                    f"• <b>Smart Money Intent:</b> Heavy Call accumulation / Put unwinding has shifted the minimum writer payout level <b>LOWER</b>.\n"
                    f"• <b>Institutional Anchor:</b> Bounces to ₹{result.prev_max_pain:,.0f} face heavy supply. Favorable for ATM Put buying."
                )

                expiry_info = f"\n🗓️ <b>Expiry:</b> {expiry}" if expiry else ""

                msg = (
                    f"🧲 <b>MAX PAIN LEVEL SHIFT — {symbol}</b>\n\n"
                    f"🎯 <b>New Max Pain:</b> ₹{result.max_pain:,.1f}\n"
                    f"⏮️ <b>Previous Level:</b> ₹{result.prev_max_pain:,.1f}\n"
                    f"📊 <b>Shift Delta:</b> <b>{sign} pts</b> ({badge})\n"
                    f"📌 <b>Current Spot:</b> ₹{result.spot_price:,.1f} (PCR: {result.pcr_oi:.2f}){expiry_info}\n\n"
                    f"💡 <b>Microstructure Forensics:</b>\n"
                    f"{implication}\n\n"
                    f"⏰ <i>Time: {result.timestamp.strftime('%H:%M:%S IST')}</i>"
                )
                try:
                    self.notifier.send_message(msg)
                    LOGGER.info("Dispatched Max Pain alert for %s: %s -> %s", symbol, result.prev_max_pain, result.max_pain)
                except Exception as e:
                    LOGGER.error("Failed to send Max Pain alert: %s", e)

        # 2. Institutional Traps Alert
        for trap in result.traps:
            key = f"{symbol}_TRAP_{hash(trap) % 10000}"
            if self._debounce(key, window_seconds=600):
                msg = (
                    f"⚡ <b>INSTITUTIONAL TRAP DETECTED — {symbol}</b>\n\n"
                    f"{trap}\n\n"
                    f"🎯 <b>Spot Price:</b> ₹{result.spot_price:,.1f}\n"
                    f"🧱 <b>Walls:</b> CE ₹{result.ce_wall:,.0f} | PE ₹{result.pe_wall:,.0f}\n"
                    f"📊 <b>PCR:</b> {result.pcr_oi:.2f} (Velocity: {result.pcr_velocity:+.3f}/min)\n\n"
                    f"⏰ <i>Time: {result.timestamp.strftime('%H:%M:%S IST')}</i>"
                )
                try:
                    self.notifier.send_message(msg)
                    LOGGER.info("Dispatched Trap alert for %s", symbol)
                except Exception as e:
                    LOGGER.error("Failed to send Trap alert: %s", e)

        # 3. Short Squeeze / Liquidation Breakout Alert
        for sq in result.squeezes:
            key = f"{symbol}_SQUEEZE_{hash(sq) % 10000}"
            if self._debounce(key, window_seconds=600):
                msg = (
                    f"🔥 <b>EXPLOSIVE SQUEEZE DETECTED — {symbol}</b>\n\n"
                    f"{sq}\n\n"
                    f"🎯 <b>Spot:</b> ₹{result.spot_price:,.1f} | <b>Max Pain:</b> ₹{result.max_pain:,.0f}\n"
                    f"📊 <b>ATM Volume Ratio:</b> {result.atm_vol_ratio:.2f}x\n\n"
                    f"⏰ <i>Time: {result.timestamp.strftime('%H:%M:%S IST')}</i>"
                )
                try:
                    self.notifier.send_message(msg)
                    LOGGER.info("Dispatched Squeeze alert for %s", symbol)
                except Exception as e:
                    LOGGER.error("Failed to send Squeeze alert: %s", e)

    def _persist_state(self) -> None:
        """Save latest forensics state to JSON for Dashboard and Chief Agent consumption."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            serializable: Dict[str, Any] = {}
            for sym, res in self.latest_results.items():
                data = asdict(res)
                data["timestamp"] = res.timestamp.isoformat()
                serializable[sym] = data

            with open(self.state_file, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            LOGGER.error("Failed to persist Option Chain Monitor state: %s", e)

    def get_latest_result(self, symbol: str) -> Optional[DynamicForensicResult]:
        clean_sym = self.clean_symbol(symbol)
        return self.latest_results.get(clean_sym)
