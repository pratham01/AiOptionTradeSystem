"""
Sector Conflict Resolver — Institutional Sector Trend & Alignment Gatekeeper
=============================================================================
Prevents broadcasting conflicting trade setups to Telegram by cross-referencing
individual stock signals with real-time / daily sector momentum:

Rules:
1. BUY CALL (Bullish) on a stock whose parent Sector is DOWN / LAGGING (e.g. COFORGE when IT is -1.6%)
   -> DISQUALIFIED & SUPPRESSED from Telegram (Sector Headwind Trap).
2. BUY PUT (Bearish) on a stock whose parent Sector is UP / LEADING
   -> DISQUALIFIED & SUPPRESSED from Telegram (Sector Tailwind Trap).
3. Setups are only broadcast to Telegram if Sector is ALIGNED or NEUTRAL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_sector_mapping

LOGGER = logging.getLogger(__name__)


@dataclass
class SectorState:
    """Represents the performance and trend of an industry sector."""
    sector: str
    pct_change: float             # Average percentage change of sector stocks
    rs_slope: float               # Relative Strength slope vs Nifty 50
    status: str                   # "🔥 LEADING", "❄️ LAGGING", "⚖️ NEUTRAL"
    trend: str                    # "BULLISH", "BEARISH", "NEUTRAL"
    stock_count: int = 0
    advancing_count: int = 0
    declining_count: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.trend == "BULLISH" or self.pct_change >= 0.3

    @property
    def is_bearish(self) -> bool:
        return self.trend == "BEARISH" or self.pct_change <= -0.3


@dataclass
class SectorAlignmentResult:
    """Evaluation of an individual stock setup against its parent sector."""
    symbol: str
    sector: str
    direction: int                # 1 for CALL (Long), -1 for PUT (Short)
    is_aligned: bool              # True if sector supports or is neutral to trade
    has_conflict: bool            # True if sector directly opposes trade
    should_send_telegram: bool    # True only if no conflict
    sector_pct: float             # Sector average return
    sector_status: str            # "🔥 LEADING", "❄️ LAGGING", etc.
    verdict: str                  # "APPROVED_TAILWIND", "APPROVED_NEUTRAL", "SUPPRESSED_SECTOR_HEADWIND"
    reason: str


class SectorConflictResolver:
    """
    Evaluates and enforces Sector-Stock Alignment across all Telegram dispatchers.
    """

    def __init__(self, conflict_threshold_pct: float = 0.3) -> None:
        self.engine = get_engine()
        self.mapping = get_sector_mapping()
        self.conflict_threshold_pct = conflict_threshold_pct
        self._cached_sector_states: Optional[Dict[str, SectorState]] = None
        self._cache_timestamp: Optional[datetime] = None

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Returns the sector for a given symbol."""
        clean_sym = symbol.strip()
        if not clean_sym.startswith("NSE:"):
            clean_sym = f"NSE:{clean_sym}"
        if not clean_sym.endswith("-EQ") and not clean_sym.endswith("-INDEX"):
            clean_sym = f"{clean_sym}-EQ"
        return self.mapping.get(clean_sym, "UNKNOWN")

    def get_all_sector_states(self, force_refresh: bool = False) -> Dict[str, SectorState]:
        """
        Calculates real-time / latest daily sector performance and relative strength.
        Cached for 5 minutes during live sessions.
        """
        now = datetime.now()
        if (
            not force_refresh
            and self._cached_sector_states is not None
            and self._cache_timestamp is not None
            and (now - self._cache_timestamp).total_seconds() < 300
        ):
            return self._cached_sector_states

        states: Dict[str, SectorState] = {}
        try:
            with self.engine.connect() as conn:
                # Query the latest trading day from ohlcv_daily
                df = pd.read_sql(
                    text("""
                        SELECT timestamp, symbol, open, high, low, close, volume
                        FROM ohlcv_daily
                        WHERE timestamp >= (
                            SELECT min(timestamp) FROM (
                                SELECT DISTINCT timestamp FROM ohlcv_daily ORDER BY timestamp DESC LIMIT 20
                            )
                        )
                        ORDER BY symbol, timestamp ASC
                    """),
                    conn,
                )

                if df.empty:
                    LOGGER.warning("SectorConflictResolver: No daily data found in database.")
                    return states

                df["date"] = pd.to_datetime(df["timestamp"], format="mixed").dt.date.astype(str)
                df["sector"] = df["symbol"].map(self.mapping).fillna("UNKNOWN")
                df["pct_change"] = (df["close"] - df["open"]) / df["open"] * 100

                latest_date = df["date"].max()
                latest_df = df[df["date"] == latest_date]

                # Group by sector on latest date
                for sector, grp in latest_df.groupby("sector"):
                    if sector in ("UNKNOWN", ""):
                        continue

                    avg_pct = float(grp["pct_change"].mean())
                    advancing = int((grp["pct_change"] > 0).sum())
                    declining = int((grp["pct_change"] < 0).sum())
                    total_stocks = len(grp)

                    # Determine Trend & Status
                    if avg_pct >= self.conflict_threshold_pct:
                        trend = "BULLISH"
                        status = "🔥 LEADING"
                    elif avg_pct <= -self.conflict_threshold_pct:
                        trend = "BEARISH"
                        status = "❄️ LAGGING"
                    else:
                        trend = "NEUTRAL"
                        status = "⚖️ NEUTRAL"

                    states[sector] = SectorState(
                        sector=sector,
                        pct_change=round(avg_pct, 2),
                        rs_slope=0.0,
                        status=status,
                        trend=trend,
                        stock_count=total_stocks,
                        advancing_count=advancing,
                        declining_count=declining,
                    )

            self._cached_sector_states = states
            self._cache_timestamp = now
            LOGGER.debug("SectorConflictResolver: Updated sector states across %d sectors.", len(states))

        except Exception as exc:
            LOGGER.error("Failed to compute sector states: %s", exc)

        return states

    def evaluate_alignment(self, symbol: str, direction: int) -> SectorAlignmentResult:
        """
        Evaluates whether a trade signal is aligned with its sector or has a conflict.
        
        direction: 1 for BUY CALL (Long), -1 for BUY PUT (Short)
        """
        # Indices (NIFTY50, BANKNIFTY, SENSEX) have no parent sector headwind
        if "INDEX" in symbol.upper():
            return SectorAlignmentResult(
                symbol=symbol,
                sector="INDEX",
                direction=direction,
                is_aligned=True,
                has_conflict=False,
                should_send_telegram=True,
                sector_pct=0.0,
                sector_status="INDEX",
                verdict="APPROVED_INDEX",
                reason="Index setup (Direct Market Driver)",
            )

        sector = self.get_sector_for_symbol(symbol)
        states = self.get_all_sector_states()
        sec_state = states.get(sector)

        if not sec_state:
            # Unknown sector -> neutral pass-through
            return SectorAlignmentResult(
                symbol=symbol,
                sector=sector,
                direction=direction,
                is_aligned=True,
                has_conflict=False,
                should_send_telegram=True,
                sector_pct=0.0,
                sector_status="⚖️ NEUTRAL",
                verdict="APPROVED_NEUTRAL",
                reason=f"Sector {sector} state neutral/untracked",
            )

        # ── 1. Check Bullish Setup (BUY CALL / LONG) ─────────────────────────
        if direction == 1:
            if sec_state.is_bearish:
                # SECTOR HEADWIND CONFLICT: Stock says BUY, but Sector is DOWN
                return SectorAlignmentResult(
                    symbol=symbol,
                    sector=sector,
                    direction=direction,
                    is_aligned=False,
                    has_conflict=True,
                    should_send_telegram=False,
                    sector_pct=sec_state.pct_change,
                    sector_status=sec_state.status,
                    verdict="SUPPRESSED_SECTOR_HEADWIND",
                    reason=f"Sector Headwind: {sector} is DOWN ({sec_state.pct_change:+.2f}%, {sec_state.declining_count}/{sec_state.stock_count} stocks falling)",
                )
            elif sec_state.is_bullish:
                # SECTOR TAILWIND CONFIRMATION
                return SectorAlignmentResult(
                    symbol=symbol,
                    sector=sector,
                    direction=direction,
                    is_aligned=True,
                    has_conflict=False,
                    should_send_telegram=True,
                    sector_pct=sec_state.pct_change,
                    sector_status=sec_state.status,
                    verdict="APPROVED_TAILWIND",
                    reason=f"Sector Tailwind Confirmed: {sector} is UP ({sec_state.pct_change:+.2f}%, {sec_state.advancing_count}/{sec_state.stock_count} stocks rising)",
                )
            else:
                return SectorAlignmentResult(
                    symbol=symbol,
                    sector=sector,
                    direction=direction,
                    is_aligned=True,
                    has_conflict=False,
                    should_send_telegram=True,
                    sector_pct=sec_state.pct_change,
                    sector_status=sec_state.status,
                    verdict="APPROVED_NEUTRAL",
                    reason=f"Sector {sector} is Neutral ({sec_state.pct_change:+.2f}%)",
                )

        # ── 2. Check Bearish Setup (BUY PUT / SHORT) ─────────────────────────
        else:
            if sec_state.is_bullish:
                # SECTOR TAILWIND CONFLICT: Stock says SHORT, but Sector is UP
                return SectorAlignmentResult(
                    symbol=symbol,
                    sector=sector,
                    direction=direction,
                    is_aligned=False,
                    has_conflict=True,
                    should_send_telegram=False,
                    sector_pct=sec_state.pct_change,
                    sector_status=sec_state.status,
                    verdict="SUPPRESSED_SECTOR_TAILWIND",
                    reason=f"Sector Tailwind Conflict: {sector} is UP ({sec_state.pct_change:+.2f}%, {sec_state.advancing_count}/{sec_state.stock_count} stocks rising)",
                )
            elif sec_state.is_bearish:
                # SECTOR BREAKDOWN CONFIRMATION
                return SectorAlignmentResult(
                    symbol=symbol,
                    sector=sector,
                    direction=direction,
                    is_aligned=True,
                    has_conflict=False,
                    should_send_telegram=True,
                    sector_pct=sec_state.pct_change,
                    sector_status=sec_state.status,
                    verdict="APPROVED_HEADWIND",
                    reason=f"Sector Breakdown Confirmed: {sector} is DOWN ({sec_state.pct_change:+.2f}%, {sec_state.declining_count}/{sec_state.stock_count} stocks falling)",
                )
            else:
                return SectorAlignmentResult(
                    symbol=symbol,
                    sector=sector,
                    direction=direction,
                    is_aligned=True,
                    has_conflict=False,
                    should_send_telegram=True,
                    sector_pct=sec_state.pct_change,
                    sector_status=sec_state.status,
                    verdict="APPROVED_NEUTRAL",
                    reason=f"Sector {sector} is Neutral ({sec_state.pct_change:+.2f}%)",
                )

    def filter_setups_for_telegram(self, setups: list) -> Tuple[list, list]:
        """
        Filters a list of setups (SMC, Wyckoff, Breakouts), separating verified setups
        from suppressed conflict setups.
        
        Returns:
            (valid_setups_with_sector_notes, suppressed_conflict_setups)
        """
        valid_setups = []
        suppressed_setups = []

        for s in setups:
            sym = getattr(s, "symbol", "")
            direction = getattr(s, "direction", 1)
            eval_res = self.evaluate_alignment(sym, direction)

            if eval_res.should_send_telegram:
                # Attach sector alignment note to reasons if not already present
                reasons = getattr(s, "reasons", [])
                if hasattr(s, "reasons") and isinstance(reasons, list):
                    if eval_res.sector != "INDEX" and not any("Sector" in r for r in reasons):
                        reasons.insert(0, eval_res.reason)
                valid_setups.append(s)
            else:
                LOGGER.info(
                    "🚫 Telegram Suppressed [%s %s]: %s",
                    sym, getattr(s, "action", ""), eval_res.reason
                )
                suppressed_setups.append((s, eval_res))

        return valid_setups, suppressed_setups
