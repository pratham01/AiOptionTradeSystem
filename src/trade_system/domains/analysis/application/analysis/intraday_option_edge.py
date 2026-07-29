"""
IntradayOptionEdgePipeline — Unified pipeline for high-conviction intraday
F&O stock selection for option buying.

Chains together 5 stages:
  1. FOIntradayShortlist   → Universe filter (200 → ~20 stocks)
  2. RegimeDetector        → Market regime (TRENDING/NEUTRAL/CHOPPY)
  3. IntradayEdgeScorer    → 7-layer confluence scoring
  4. SmartEntryTrigger     → Precise entry / SL / target levels
  5. OptionStrikeSelector  → Optimal strike + premium estimation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from trade_system.domains.analysis.application.analysis.fo_intraday_shortlist import (
    FOIntradayShortlist,
    ShortlistCandidate,
)
from trade_system.domains.analysis.application.analysis.regime_detector import (
    MarketRegime,
    RegimeDetector,
    RegimeResult,
)
from trade_system.domains.analysis.application.analysis.intraday_edge_scorer import (
    EdgeScore,
    IntradayEdgeScorer,
)
from trade_system.domains.analysis.application.analysis.smart_entry_trigger import (
    EntryTrigger,
    SmartEntryTrigger,
)
from trade_system.domains.analysis.application.analysis.option_strike_selector import (
    OptionStrikeSelector,
    OptionSuggestion,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class OptionEdgeAlert:
    """A fully enriched option buying alert ready for Telegram dispatch."""
    # Stock info
    symbol: str
    sector: str
    ltp: float
    change_pct: float

    # Edge scoring
    edge_score: float
    direction: str                  # "CALL" or "PUT"
    direction_confidence: float
    key_signals: List[str]          # Human-readable top confluence signals

    # Market context
    regime: MarketRegime
    regime_signals: List[str]

    # Entry levels
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    entry_type: str                 # "VWAP_PULLBACK", "ORB_BREAKOUT", etc.
    risk_reward_stock: float        # Stock-level R:R

    # Option suggestion
    option: Optional[OptionSuggestion] = None

    # Shortlist reasons (why this stock was chosen)
    shortlist_reasons: List[str] = field(default_factory=list)

    def format_telegram(self) -> str:
        """Format as an HTML Telegram message."""
        icon = "📈" if self.direction == "CALL" else "📉"
        dir_label = "CALL" if self.direction == "CALL" else "PUT"
        clean_sym = self.symbol.replace("NSE:", "").replace("-EQ", "")

        lines = [
            f"🎯 <b>INTRADAY OPTION EDGE — {dir_label}</b>",
            "",
            f"{icon} <b>{clean_sym}</b> ({self.sector})",
            f"Edge Score: <b>{self.edge_score:.0f}/100</b> | Regime: {self.regime.value}",
            f"Direction: {dir_label} ({self.direction_confidence:.0%} layers agree)",
            "",
            "📍 <b>Entry Levels:</b>",
            f"  Entry: ₹{self.entry_price:,.2f} ({self.entry_type.replace('_', ' ').title()})",
            f"  Stop Loss: ₹{self.stop_loss:,.2f}",
            f"  Target 1: ₹{self.target_1:,.2f}",
            f"  Target 2: ₹{self.target_2:,.2f}",
            f"  Stock R:R: {self.risk_reward_stock:.1f}:1",
        ]

        if self.option:
            lines.extend([
                "",
                "🔔 <b>Option Suggestion:</b>",
                f"  Strike: {self.option.display_strike}",
                f"  Est. Premium: ₹{self.option.est_premium_low:.0f}–{self.option.est_premium_high:.0f}",
                f"  Delta: {self.option.est_delta:.2f} | R:R: {self.option.risk_reward:.1f}:1",
                f"  Lot: {self.option.lot_size} | Cost: ~₹{self.option.lot_value:,.0f}",
            ])

        # Key signals (max 5)
        lines.append("")
        lines.append("📊 <b>Key Signals:</b>")
        for sig in self.key_signals[:5]:
            lines.append(f"  {sig}")

        # Consolidation & Structure (ATH / previous swing)
        structural_reasons = [r for r in self.shortlist_reasons if "Sector" not in r]
        if structural_reasons:
            lines.append("")
            lines.append("🔍 <b>Structure & Consolidation:</b>")
            for reason in structural_reasons:
                lines.append(f"  • {reason}")

        return "\n".join(lines)


class IntradayOptionEdgePipeline:
    """
    Orchestrates the complete intraday option buying pipeline.
    Designed to be called every 5 minutes during market hours.
    """

    def __init__(
        self,
        max_alerts_per_cycle: int = 3,
        min_risk_reward: float = 1.5,
    ) -> None:
        self._shortlist = FOIntradayShortlist()
        self._regime = RegimeDetector()
        self._edge_scorer = IntradayEdgeScorer(horizon="INTRADAY")
        self._entry_trigger = SmartEntryTrigger(horizon="INTRADAY")
        self._strike_selector = OptionStrikeSelector()

        self.max_alerts_per_cycle = max_alerts_per_cycle
        self.min_risk_reward = min_risk_reward

    def scan(self, target_date: Optional[date] = None) -> List[OptionEdgeAlert]:
        """
        Run the complete pipeline and return actionable alerts.

        Returns:
            List of OptionEdgeAlert objects, sorted by edge score (highest first).
            Maximum of max_alerts_per_cycle alerts per invocation.
        """
        # ── Stage 1: Universe Filter ───────────────────────────────────────
        LOGGER.info("IOE Pipeline Stage 1: Universe shortlisting...")
        shortlisted = self._shortlist.shortlist(target_date)
        if not shortlisted:
            LOGGER.info("IOE Pipeline: No stocks passed the universe filter.")
            return []

        LOGGER.info(f"IOE Pipeline: {len(shortlisted)} stocks shortlisted.")

        # ── Stage 2: Regime Detection ──────────────────────────────────────
        LOGGER.info("IOE Pipeline Stage 2: Regime detection...")
        regime_result = self._regime.detect(target_date)
        LOGGER.info(f"IOE Pipeline: Regime = {regime_result.label} (threshold: {regime_result.score_threshold})")

        # ── Stage 3: Edge Scoring ──────────────────────────────────────────
        LOGGER.info("IOE Pipeline Stage 3: Edge scoring...")
        edge_scores = self._edge_scorer.scan(target_date=target_date)
        if not edge_scores:
            LOGGER.info("IOE Pipeline: Edge scorer returned no results.")
            return []

        # Build a lookup for shortlisted symbols and their reasons
        shortlist_lookup = {c.symbol: c for c in shortlisted}
        shortlisted_symbols = set(shortlist_lookup.keys())

        # Filter edge scores to only shortlisted stocks
        # Also apply regime threshold
        qualifying: List[tuple[EdgeScore, ShortlistCandidate]] = []
        for edge in edge_scores:
            if edge.symbol not in shortlisted_symbols:
                continue
            if edge.final_score < regime_result.score_threshold:
                continue
            candidate = shortlist_lookup[edge.symbol]
            qualifying.append((edge, candidate))

        LOGGER.info(
            f"IOE Pipeline: {len(qualifying)} stocks passed regime-adjusted "
            f"threshold ({regime_result.score_threshold})"
        )

        if not qualifying:
            return []

        # ── Fetch 15m data for entry triggers ──────────────────────────────
        # The SmartEntryTrigger needs the actual 15m candle DataFrame to
        # compute VWAP pullbacks, ORB breakouts, and supertrend touches.
        LOGGER.info("IOE Pipeline: Fetching 15m data for entry trigger evaluation...")
        df_15m = self._edge_scorer._fetch_15m_data(target_date)
        if df_15m.empty:
            LOGGER.warning("IOE Pipeline: No 15m data available for entry triggers.")
            return []

        # Resolve target_date from 15m data if not provided
        resolved_date = target_date
        if resolved_date is None:
            non_idx = df_15m[~df_15m["symbol"].str.contains("INDEX")]
            if not non_idx.empty:
                resolved_date = non_idx["timestamp"].max().date()

        # Group 15m data by symbol
        symbol_15m_groups = {
            sym: grp.sort_values("timestamp")
            for sym, grp in df_15m.groupby("symbol")
        }

        # ── Stage 4 & 5: Entry + Option Selection ─────────────────────────
        LOGGER.info("IOE Pipeline Stage 4+5: Entry triggers & option selection...")
        alerts: List[OptionEdgeAlert] = []

        for edge, candidate in qualifying:
            # Extract key signals from layers
            key_signals = self._extract_key_signals(edge)

            # Get per-symbol 15m data
            df_sym_15m = symbol_15m_groups.get(edge.symbol)

            # Smart entry trigger — pass the 15m data and target_date
            entry = self._entry_trigger.evaluate(
                direction=edge.direction,
                ltp=edge.ltp,
                atr=edge.atr,
                df_base=df_sym_15m,
                target_date=resolved_date,
            )

            if entry is None:
                # No clean entry found — skip
                LOGGER.debug(f"IOE Pipeline: No entry trigger for {edge.symbol}")
                continue

            # Risk-reward check
            stock_rr = edge.risk_reward
            if entry.target_1 > 0 and entry.stop_loss > 0 and entry.entry_price > 0:
                risk = abs(entry.entry_price - entry.stop_loss)
                reward = abs(entry.target_1 - entry.entry_price)
                stock_rr = round(reward / risk, 2) if risk > 0 else 0.0

            if stock_rr < self.min_risk_reward:
                continue

            # Option strike selection
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

            alert = OptionEdgeAlert(
                symbol=edge.symbol,
                sector=edge.sector,
                ltp=edge.ltp,
                change_pct=edge.change_pct,
                edge_score=edge.final_score,
                direction=edge.direction,
                direction_confidence=edge.direction_confidence,
                key_signals=key_signals,
                regime=regime_result.regime,
                regime_signals=regime_result.signals,
                entry_price=entry.entry_price,
                stop_loss=entry.stop_loss,
                target_1=entry.target_1,
                target_2=entry.target_2,
                entry_type=entry.trigger_type,
                risk_reward_stock=stock_rr,
                option=option,
                shortlist_reasons=candidate.reasons,
            )
            alerts.append(alert)

        # Sort by edge score and limit
        alerts.sort(key=lambda a: a.edge_score, reverse=True)
        result = alerts[: self.max_alerts_per_cycle]

        LOGGER.info(f"IOE Pipeline: Returning {len(result)} actionable alerts.")
        return result

    # ── Private Helpers ────────────────────────────────────────────────────────

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

        # Sort: strong signals first
        signals.sort(key=lambda s: s.startswith("✅"), reverse=True)
        return signals
