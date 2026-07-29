"""
EvolutionLoop — runs the full EOD self-improvement cycle.

Schedule this to run daily after market close (e.g., 15:45 IST).
It orchestrates: OutcomeTracker → FeatureAnalyzer → WeightEvolver → Report.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.database.models import Base, EvolutionReport
from trade_system.domains.analysis.application.evolution.trade_logger import TradeLogger
from trade_system.domains.analysis.application.evolution.outcome_tracker import OutcomeTracker
from trade_system.domains.analysis.application.evolution.feature_analyzer import FeatureAnalyzer
from trade_system.domains.analysis.application.evolution.weight_evolver import WeightEvolver

LOGGER = logging.getLogger(__name__)


class EvolutionLoop:
    """
    End-of-Day self-evolution pipeline:

    1. Mark outcomes for today's pending trades
    2. Load closed trades from last N days
    3. Run feature analysis
    4. Update agent weights if enough data
    5. Generate performance report
    6. (Optionally) send Telegram summary
    """

    def __init__(
        self,
        broker: Any = None,
        notifier: Any = None,
        lookback_days: int = 30,
        min_trades_for_evolution: int = 10,
        engine: Any = None,
    ) -> None:
        self.broker = broker
        self.notifier = notifier
        self.lookback_days = lookback_days
        self.min_trades_for_evolution = min_trades_for_evolution
        self.engine = engine or get_engine()
        Base.metadata.create_all(self.engine)

        self.trade_logger = TradeLogger(engine=self.engine)
        self.outcome_tracker = OutcomeTracker(broker=broker, engine=self.engine)
        self.feature_analyzer = FeatureAnalyzer()
        self.weight_evolver = WeightEvolver(engine=self.engine)

    def run(self, target_date: str | None = None) -> dict[str, Any]:
        """
        Run the complete evolution loop for a given date.

        Args:
            target_date: Date to evaluate (YYYY-MM-DD). Defaults to yesterday.

        Returns:
            Dict with results: win_rate, new_weights, report_text, etc.
        """
        target_date = target_date or (date.today() - timedelta(days=1)).isoformat()
        LOGGER.info("Starting evolution loop for %s (lookback: %d days).",
                    target_date, self.lookback_days)

        # Step 1: Mark outcomes
        LOGGER.info("Step 1: Marking trade outcomes...")
        outcome_counts = self.outcome_tracker.check_and_mark_outcomes(target_date)

        # Step 2: Load closed trades
        LOGGER.info("Step 2: Loading closed trades...")
        closed_trades = self.trade_logger.get_closed_trades(self.lookback_days)
        LOGGER.info("Loaded %d closed trades.", len(closed_trades))

        # Step 3: Feature analysis
        LOGGER.info("Step 3: Running feature analysis...")
        analysis = self.feature_analyzer.analyze(
            closed_trades,
            min_trades=self.min_trades_for_evolution,
        )

        # Step 4: Evolve weights
        weights_updated = False
        new_weights: dict[str, dict[str, float]] = {}

        if analysis.sufficient_data:
            LOGGER.info("Step 4: Evolving agent weights...")
            new_weights = self.weight_evolver.evolve(
                analysis=analysis,
                win_rate=analysis.win_rate,
                avg_pnl_pct=self._compute_avg_pnl(closed_trades),
                total_trades=len(closed_trades),
            )
            weights_updated = bool(new_weights)
        else:
            LOGGER.info("Step 4: Skipping weight evolution (insufficient data).")

        # Step 5: Generate report
        LOGGER.info("Step 5: Generating performance report...")
        report_text = self._build_report(
            target_date=target_date,
            outcome_counts=outcome_counts,
            analysis=analysis,
            new_weights=new_weights,
            weights_updated=weights_updated,
        )

        # Step 6: Persist report
        self._save_report(analysis, weights_updated, report_text, closed_trades)

        # Step 7: Send notification
        if self.notifier:
            try:
                self.notifier.send(self._format_telegram_report(
                    target_date, analysis, outcome_counts, weights_updated
                ))
            except Exception as exc:
                LOGGER.error("Failed to send evolution report via Telegram: %s", exc)

        LOGGER.info("Evolution loop complete.\n%s", report_text)
        return {
            "date": target_date,
            "outcome_counts": outcome_counts,
            "win_rate": analysis.win_rate,
            "weights_updated": weights_updated,
            "new_weights": new_weights,
            "report": report_text,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_avg_pnl(self, trades: list[Any]) -> float | None:
        pnls = [t.actual_pnl_pct for t in trades if t.actual_pnl_pct is not None]
        if not pnls:
            return None
        return round(sum(pnls) / len(pnls), 3)

    def _build_report(
        self,
        target_date: str,
        outcome_counts: dict[str, int],
        analysis: Any,
        new_weights: dict[str, dict[str, float]],
        weights_updated: bool,
    ) -> str:
        lines = [
            f"# Evolution Report — {target_date}",
            "",
            f"## Today's Outcome Summary",
            f"  Wins: {outcome_counts.get('wins', 0)}",
            f"  Losses: {outcome_counts.get('losses', 0)}",
            f"  Neutrals: {outcome_counts.get('neutrals', 0)}",
            f"  Skipped: {outcome_counts.get('skipped', 0)}",
            "",
            f"## Rolling {self.lookback_days}-Day Performance",
        ]

        if analysis.total_trades > 0:
            lines += [
                f"  Trades analyzed: {analysis.total_trades}",
                f"  Win Rate: {analysis.win_rate:.1%} ({analysis.win_count}W / {analysis.loss_count}L)",
            ]
        else:
            lines.append("  No closed trades in lookback window.")

        if analysis.sufficient_data:
            lines += [
                "",
                "## Top Predictive Conditions",
            ]
            for cond in analysis.top_conditions[:5]:
                lines.append(f"  • {cond}")

        lines += ["", f"## Weight Update: {'YES ✓' if weights_updated else 'NO (unchanged)'}"]
        if weights_updated:
            for agent, weights in new_weights.items():
                lines.append(f"  {agent}:")
                for k, v in weights.items():
                    lines.append(f"    {k}: {v:.3f}")

        return "\n".join(lines)

    def _format_telegram_report(
        self,
        target_date: str,
        analysis: Any,
        outcome_counts: dict[str, int],
        weights_updated: bool,
    ) -> str:
        w = outcome_counts.get("wins", 0)
        l_ = outcome_counts.get("losses", 0)
        n = outcome_counts.get("neutrals", 0)
        wr_str = f"{analysis.win_rate:.1%}" if analysis.total_trades > 0 else "N/A"

        lines = [
            f"🤖 <b>Evolution Loop Complete — {target_date}</b>",
            "",
            f"📊 Today: {w}W / {l_}L / {n}N",
            f"📈 Rolling Win Rate ({self.lookback_days}d): {wr_str}",
            f"⚖️ Weights Updated: {'Yes ✓' if weights_updated else 'No (stable)'}",
        ]

        if analysis.top_conditions:
            lines += ["", "🎯 <b>Top Win Conditions:</b>"]
            for cond in analysis.top_conditions[:3]:
                lines.append(f"  • {cond}")

        return "\n".join(lines)

    def _save_report(
        self,
        analysis: Any,
        weights_updated: bool,
        report_text: str,
        closed_trades: list[Any],
    ) -> None:
        try:
            with Session(self.engine) as session:
                report = EvolutionReport(
                    run_at=datetime.utcnow(),
                    lookback_days=self.lookback_days,
                    total_trades_analyzed=analysis.total_trades,
                    win_count=analysis.win_count,
                    loss_count=analysis.loss_count,
                    neutral_count=analysis.neutral_count,
                    win_rate=analysis.win_rate if analysis.total_trades else None,
                    avg_pnl_pct=self._compute_avg_pnl(closed_trades),
                    weights_updated=int(weights_updated),
                    report_text=report_text,
                )
                session.add(report)
                session.commit()
        except Exception as exc:
            LOGGER.error("Failed to save evolution report to DB: %s", exc)


def run_main():
    """CLI entry point for EvolutionLoop."""
    logging.basicConfig(level=logging.INFO)
    loop = EvolutionLoop()
    loop.run()
