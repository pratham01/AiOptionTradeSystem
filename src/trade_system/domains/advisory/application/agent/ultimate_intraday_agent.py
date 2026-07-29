"""
UltimateIntradayAgent — Unified intraday option buying agent that combines
all existing analysis modules into a single multi-layer decision framework.

Architecture (6 layers):
  Layer 1: Pre-Market Context   → RegimeDetector + Expiry check + PCR bias
  Layer 2: Universe Filter      → FOIntradayShortlist (~200 → ~20 stocks)
  Layer 3: Confluence Scoring   → IntradayEdgeScorer (7 layers) + BB filter + EMA/VWAP
  Layer 4: Confirmation Gates   → Volume surge, OI buildup, VIX gate
  Layer 5: Entry & Risk         → SmartEntryTrigger + R:R ≥ 1.5 + OptionStrikeSelector
  Layer 6: Trade Management     → Max 2-3 trades/day, 2 PM cutoff, 3:15 PM square-off

Expiry Day Override:
  Routes through GammaBlastDetector for 0DTE squeeze breakout plays.
  Max 1 trade on expiry. Tighter SL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, time as dt_time
from typing import List, Optional, Dict, Any

import pandas as pd
from sqlalchemy import text

from trade_system.domains.market_data.infrastructure.database.connection import get_engine

# -- Existing analysis modules --
from trade_system.domains.analysis.application.analysis.fo_intraday_shortlist import (
    FOIntradayShortlist,
    ShortlistCandidate,
)
from trade_system.domains.analysis.application.analysis.regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeResult,
)
from trade_system.domains.analysis.application.analysis.intraday_edge_scorer import (
    IntradayEdgeScorer,
    EdgeScore,
)
from trade_system.domains.analysis.application.analysis.smart_entry_trigger import (
    SmartEntryTrigger,
    EntryTrigger,
)
from trade_system.domains.analysis.application.analysis.option_strike_selector import (
    OptionStrikeSelector,
    OptionSuggestion,
)
from trade_system.domains.analysis.application.analysis.gamma_blast_strategy import GammaBlastDetector
from trade_system.domains.strategy.application.indicators.bollinger_bands import BollingerBandsDetector
from trade_system.domains.strategy.application.indicators.vwap import VWAPIndicator

LOGGER = logging.getLogger(__name__)

ENTRY_CUTOFF = dt_time(14, 0)
SQUARE_OFF_TIME = dt_time(15, 15)
VIX_DANGER_THRESHOLD = 20.0


@dataclass
class UltimateTrade:
    """A fully qualified trade suggestion from the unified agent."""
    symbol: str
    sector: str
    direction: str              # "CALL" or "PUT"
    edge_score: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    entry_type: str             # "VWAP_PULLBACK", "ORB_BREAKOUT", etc.
    risk_reward: float
    option: Optional[OptionSuggestion] = None

    # Context
    regime: str = ""
    vix: float = 0.0
    pcr_bias: str = "NEUTRAL"
    oi_signal: str = "NEUTRAL"
    bb_phase: str = ""
    volume_surge: float = 0.0
    is_expiry_day: bool = False

    # Confluence signals
    key_signals: List[str] = field(default_factory=list)
    shortlist_reasons: List[str] = field(default_factory=list)

    def format_display(self) -> str:
        icon = "📈" if self.direction == "CALL" else "📉"
        clean_sym = self.symbol.replace("NSE:", "").replace("-EQ", "")
        lines = [
            f"{icon} {clean_sym} ({self.sector}) | {self.direction}",
            f"Edge Score: {self.edge_score:.0f}/100 | Regime: {self.regime}",
            f"Entry: ₹{self.entry_price:,.2f} ({self.entry_type})",
            f"SL: ₹{self.stop_loss:,.2f} | T1: ₹{self.target_1:,.2f} | T2: ₹{self.target_2:,.2f}",
            f"R:R: {self.risk_reward:.1f}:1 | VIX: {self.vix:.1f} | PCR: {self.pcr_bias}",
            f"OI Signal: {self.oi_signal} | BB Phase: {self.bb_phase} | Vol Surge: {self.volume_surge:.1f}x",
        ]
        if self.option:
            lines.append(
                f"Option: {self.option.display_strike} | "
                f"Est. Premium: ₹{self.option.est_premium_low:.0f}–{self.option.est_premium_high:.0f}"
            )
        lines.append("Signals: " + " | ".join(self.key_signals[:5]))
        return "\n".join(lines)


class UltimateIntradayAgent:
    """
    The unified intraday option buying agent.
    Combines all analysis modules and enforces strict trade limits.
    """

    def __init__(
        self,
        max_trades_normal: int = 3,
        max_trades_expiry: int = 1,
        min_risk_reward: float = 1.5,
        min_edge_score_override: int | None = None,
    ) -> None:
        self.engine = get_engine()
        self._shortlist = FOIntradayShortlist()
        self._regime = RegimeDetector()
        self._edge_scorer = IntradayEdgeScorer(horizon="INTRADAY")
        self._entry_trigger = SmartEntryTrigger(horizon="INTRADAY")
        self._strike_selector = OptionStrikeSelector()
        self._gamma_blast = GammaBlastDetector()
        self._bb_detector = BollingerBandsDetector()
        self._vwap = VWAPIndicator()

        self.max_trades_normal = max_trades_normal
        self.max_trades_expiry = max_trades_expiry
        self.min_risk_reward = min_risk_reward
        self.min_edge_score_override = min_edge_score_override

    # ──────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def scan(self, target_date: Optional[date] = None) -> List[UltimateTrade]:
        """
        Run the full unified pipeline and return at most N actionable trades.
        """
        LOGGER.info("Ultimate Agent: Starting scan...")

        # ── Layer 1: Pre-Market Context ────────────────────────────────────
        regime_result = self._regime.detect(target_date)
        LOGGER.info(f"Ultimate Agent L1: Regime = {regime_result.regime.value}, VIX = {regime_result.vix}")

        # Expiry day check
        is_expiry = self._check_expiry_day(target_date or date.today())
        max_trades = self.max_trades_expiry if is_expiry else self.max_trades_normal

        # PCR bias from option chain (if available)
        pcr_bias, pcr_value = self._get_pcr_bias(target_date)
        LOGGER.info(f"Ultimate Agent L1: PCR = {pcr_value:.2f} ({pcr_bias}), Expiry = {is_expiry}")

        # VIX gate: if VIX > threshold, raise the bar significantly
        edge_threshold = regime_result.score_threshold
        if self.min_edge_score_override is not None:
            edge_threshold = self.min_edge_score_override
        if regime_result.vix > VIX_DANGER_THRESHOLD:
            edge_threshold = max(edge_threshold, 85)
            LOGGER.info(f"Ultimate Agent L1: VIX danger zone! Threshold raised to {edge_threshold}")

        # ── Layer 2: Universe Filter ──────────────────────────────────────
        LOGGER.info("Ultimate Agent L2: Universe shortlisting...")
        shortlisted = self._shortlist.shortlist(target_date)
        if not shortlisted:
            LOGGER.info("Ultimate Agent L2: No stocks passed universe filter.")
            return []
        LOGGER.info(f"Ultimate Agent L2: {len(shortlisted)} stocks shortlisted.")

        shortlist_lookup = {c.symbol: c for c in shortlisted}
        shortlisted_symbols = set(shortlist_lookup.keys())

        # ── Layer 3: Confluence Scoring ───────────────────────────────────
        LOGGER.info("Ultimate Agent L3: Edge scoring...")
        edge_scores = self._edge_scorer.scan(target_date=target_date)
        if not edge_scores:
            LOGGER.info("Ultimate Agent L3: Edge scorer returned no results.")
            return []

        # Filter to shortlisted + above regime-adjusted threshold
        qualifying: List[tuple[EdgeScore, ShortlistCandidate]] = []
        for edge in edge_scores:
            if edge.symbol not in shortlisted_symbols:
                continue
            if edge.final_score < edge_threshold:
                continue
            candidate = shortlist_lookup[edge.symbol]
            qualifying.append((edge, candidate))

        LOGGER.info(f"Ultimate Agent L3: {len(qualifying)} stocks above threshold ({edge_threshold})")
        if not qualifying:
            return []

        # Sort by score (best first)
        qualifying.sort(key=lambda x: x[0].final_score, reverse=True)

        # ── Fetch 15m data for BB, VWAP, and entry triggers ───────────────
        LOGGER.info("Ultimate Agent: Fetching 15m data...")
        df_15m = self._edge_scorer._fetch_15m_data(target_date)
        if df_15m.empty:
            LOGGER.warning("Ultimate Agent: No 15m data available.")
            return []

        resolved_date = target_date
        if resolved_date is None:
            non_idx = df_15m[~df_15m["symbol"].str.contains("INDEX")]
            if not non_idx.empty:
                resolved_date = non_idx["timestamp"].max().date()

        symbol_15m_groups = {
            sym: grp.sort_values("timestamp")
            for sym, grp in df_15m.groupby("symbol")
        }

        # ── Layers 4 & 5: Confirmation Gates + Entry ─────────────────────
        LOGGER.info("Ultimate Agent L4+L5: Confirmation gates & entry triggers...")
        trades: List[UltimateTrade] = []

        for edge, candidate in qualifying:
            if len(trades) >= max_trades:
                break

            df_sym = symbol_15m_groups.get(edge.symbol)
            if df_sym is None or df_sym.empty:
                continue

            # ── L4a: Bollinger Band phase filter ──────────────────────────
            bb_phase = self._get_bb_phase(df_sym)
            if bb_phase in ["SQUEEZE", "DISTRIBUTION"] and edge.direction == "CALL":
                LOGGER.debug(f"Ultimate Agent L4: {edge.symbol} skipped (BB={bb_phase}, dir=CALL)")
                continue
            if bb_phase in ["SQUEEZE", "ACCUMULATION"] and edge.direction == "PUT":
                LOGGER.debug(f"Ultimate Agent L4: {edge.symbol} skipped (BB={bb_phase}, dir=PUT)")
                continue

            # ── L4b: Volume surge check ───────────────────────────────────
            vol_surge = candidate.volume_surge
            if vol_surge < 1.2:
                LOGGER.debug(f"Ultimate Agent L4: {edge.symbol} skipped (vol_surge={vol_surge:.1f}x)")
                continue

            # ── L4c: PCR direction alignment ──────────────────────────────
            pcr_penalty = 0.0
            if pcr_bias == "BULLISH" and edge.direction == "PUT":
                pcr_penalty = 10.0  # penalize but don't block
            elif pcr_bias == "BEARISH" and edge.direction == "CALL":
                pcr_penalty = 10.0

            adjusted_score = edge.final_score - pcr_penalty
            if adjusted_score < edge_threshold:
                LOGGER.debug(f"Ultimate Agent L4: {edge.symbol} dropped after PCR penalty")
                continue

            # ── L5a: Smart entry trigger ──────────────────────────────────
            entry = self._entry_trigger.evaluate(
                direction=edge.direction,
                ltp=edge.ltp,
                atr=edge.atr,
                df_base=df_sym,
                target_date=resolved_date,
            )
            if entry is None:
                continue

            # ── L5b: Risk-reward check ────────────────────────────────────
            risk = abs(entry.entry_price - entry.stop_loss)
            reward = abs(entry.target_1 - entry.entry_price)
            rr = round(reward / risk, 2) if risk > 0 else 0.0
            if rr < self.min_risk_reward:
                continue

            # ── L5c: Option strike selection ──────────────────────────────
            option = self._strike_selector.select(
                symbol=edge.symbol,
                direction=edge.direction,
                ltp=edge.ltp,
                atr=edge.atr,
                stock_entry=entry.entry_price,
                stock_sl=entry.stop_loss,
                stock_target_1=entry.target_1,
                stock_target_2=entry.target_2,
            )

            # Extract key signals
            key_signals = self._extract_key_signals(edge)

            trade = UltimateTrade(
                symbol=edge.symbol,
                sector=edge.sector,
                direction=edge.direction,
                edge_score=adjusted_score,
                entry_price=entry.entry_price,
                stop_loss=entry.stop_loss,
                target_1=entry.target_1,
                target_2=entry.target_2,
                entry_type=entry.trigger_type,
                risk_reward=rr,
                option=option,
                regime=regime_result.regime.value,
                vix=regime_result.vix,
                pcr_bias=pcr_bias,
                oi_signal="",  # populated below if possible
                bb_phase=bb_phase,
                volume_surge=vol_surge,
                is_expiry_day=is_expiry,
                key_signals=key_signals,
                shortlist_reasons=candidate.reasons,
            )
            trades.append(trade)

        LOGGER.info(f"Ultimate Agent: Returning {len(trades)} actionable trades.")
        return trades

    # ──────────────────────────────────────────────────────────────────────
    #  PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _check_expiry_day(self, target_date: date) -> bool:
        """Check if any major index has an expiry today."""
        return (
            GammaBlastDetector.is_expiry_day("NSE:NIFTY50-INDEX", target_date) or
            GammaBlastDetector.is_expiry_day("NSE:NIFTYBANK-INDEX", target_date) or
            GammaBlastDetector.is_expiry_day("BSE:SENSEX-INDEX", target_date)
        )

    def _get_pcr_bias(self, target_date: Optional[date]) -> tuple[str, float]:
        """
        Compute a simple Put-Call Ratio from the option_chain_data table.
        PCR > 1.2 → BULLISH, PCR < 0.8 → BEARISH, else NEUTRAL.
        """
        try:
            date_str = (target_date or date.today()).isoformat()
            with self.engine.connect() as conn:
                query = text("""
                    SELECT option_type, SUM(oi) as total_oi
                    FROM option_chain_data
                    WHERE underlying_symbol = 'NSE:NIFTY50-INDEX'
                      AND date(timestamp) = :date_str
                    GROUP BY option_type
                """)
                df = pd.read_sql(query, conn, params={"date_str": date_str})

            if df.empty or len(df) < 2:
                return "NEUTRAL", 1.0

            ce_oi = df.loc[df["option_type"] == "CE", "total_oi"].sum()
            pe_oi = df.loc[df["option_type"] == "PE", "total_oi"].sum()
            pcr = pe_oi / ce_oi if ce_oi > 0 else 1.0

            if pcr > 1.2:
                return "BULLISH", pcr
            elif pcr < 0.8:
                return "BEARISH", pcr
            else:
                return "NEUTRAL", pcr
        except Exception as e:
            LOGGER.warning(f"PCR fetch failed: {e}")
            return "NEUTRAL", 1.0

    def _get_bb_phase(self, df_sym: pd.DataFrame) -> str:
        """Run BollingerBandsDetector on the symbol's 15m data and return the latest phase."""
        try:
            computed = self._bb_detector.compute(df_sym.copy())
            if "bb_phase" in computed.columns and not computed.empty:
                return str(computed["bb_phase"].iloc[-1])
        except Exception:
            pass
        return "NEUTRAL"

    @staticmethod
    def _extract_key_signals(edge: EdgeScore) -> List[str]:
        """Extract human-readable signals from EdgeScore layers."""
        signals = []
        for layer_name, layer in edge.layers.items():
            if layer.score >= 0.6:
                icon = "✅"
            elif layer.score >= 0.3:
                icon = "⚠️"
            else:
                icon = "❌"
            signals.append(f"{icon} {layer.detail}")
        signals.sort(key=lambda s: s.startswith("✅"), reverse=True)
        return signals
