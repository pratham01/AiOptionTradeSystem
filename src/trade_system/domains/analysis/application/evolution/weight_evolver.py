"""
WeightEvolver — updates agent scoring weights based on feature analysis.

Maintains a version history so degrading weight sets can be rolled back.
Weights are persisted to DB (AgentWeightHistory) and to a JSON sidecar file.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from trade_system.domains.market_data.infrastructure.database.models import AgentWeightHistory
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.analysis.application.evolution.feature_analyzer import FeatureImportance

LOGGER = logging.getLogger(__name__)

# Default weights for both agents. Keys match FeatureAnalyzer.feature_to_weight values.
DEFAULT_CANDIDATE_WEIGHTS: dict[str, float] = {
    "volume_delta": 0.30,
    "compression": 0.20,
    "value_area": 0.15,
    "trend_alignment": 0.20,
    "momentum": 0.10,
    "market_context": 0.05,
}

DEFAULT_SETUP_WEIGHTS: dict[str, float] = {
    "volume_delta": 0.25,
    "price_action": 0.30,
    "momentum": 0.20,
    "value_area": 0.15,
    "market_context": 0.10,
}

# Max single-step adjustment (clamp weights)
MAX_STEP = 0.05
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.55


class WeightEvolver:
    """
    Reads current agent weights, applies FeatureImportance adjustments,
    normalizes the result, and persists both to DB and JSON sidecar.
    """

    AGENT_CANDIDATE = "candidate_screener"
    AGENT_SETUP = "setup_validator"

    def __init__(self, weights_dir: Path | None = None, engine: Any = None) -> None:
        self.weights_dir = Path(weights_dir or "data/agent_weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine or get_engine()
        from trade_system.domains.market_data.infrastructure.database.models import Base
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load_weights(self, agent_name: str) -> dict[str, float]:
        """Load current weights from JSON sidecar or return defaults."""
        path = self._weight_path(agent_name)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                LOGGER.debug("Loaded weights for %s: %s", agent_name, data)
                return data
            except Exception as exc:
                LOGGER.warning("Failed to load weights for %s: %s", agent_name, exc)

        defaults = self._get_defaults(agent_name)
        LOGGER.info("Using default weights for %s.", agent_name)
        return defaults.copy()

    def save_weights(
        self,
        agent_name: str,
        weights: dict[str, float],
        win_rate: float | None = None,
        avg_pnl_pct: float | None = None,
        total_trades: int | None = None,
        trigger: str = "evolution_loop",
    ) -> None:
        """Persist weights to JSON sidecar and DB history."""
        # Normalize first
        weights = self._normalize(weights)
        path = self._weight_path(agent_name)
        path.write_text(json.dumps(weights, indent=2))
        LOGGER.info("Saved weights for %s: %s", agent_name, weights)

        # DB history
        with Session(self.engine) as session:
            record = AgentWeightHistory(
                recorded_at=datetime.utcnow(),
                agent_name=agent_name,
                weights_json=json.dumps(weights),
                trigger=trigger,
                win_rate=win_rate,
                avg_pnl_pct=avg_pnl_pct,
                total_trades=total_trades,
            )
            session.add(record)
            session.commit()

    # ------------------------------------------------------------------
    # Core evolution step
    # ------------------------------------------------------------------

    def evolve(
        self,
        analysis: FeatureImportance,
        win_rate: float | None = None,
        avg_pnl_pct: float | None = None,
        total_trades: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Apply feature importance adjustments to both agent weight sets.

        Returns:
            Dict with agent_name → new_weights mappings.
        """
        if not analysis.sufficient_data:
            LOGGER.info("Insufficient data for weight evolution — skipping.")
            return {}

        if not analysis.weight_adjustments:
            LOGGER.info("No weight adjustments from analysis — skipping.")
            return {}

        results: dict[str, dict[str, float]] = {}

        for agent_name in [self.AGENT_CANDIDATE, self.AGENT_SETUP]:
            current = self.load_weights(agent_name)
            new_weights = self._apply_adjustments(current, analysis.weight_adjustments)

            # Only save if weights actually changed meaningfully
            if self._changed_enough(current, new_weights):
                self.save_weights(
                    agent_name,
                    new_weights,
                    win_rate=win_rate,
                    avg_pnl_pct=avg_pnl_pct,
                    total_trades=total_trades,
                    trigger="evolution_loop",
                )
                results[agent_name] = new_weights
                LOGGER.info("Evolved weights for %s: %s → %s", agent_name, current, new_weights)
            else:
                LOGGER.info("Weights for %s unchanged — no update needed.", agent_name)
                results[agent_name] = current

        return results

    def rollback(self, agent_name: str, steps: int = 1) -> dict[str, float] | None:
        """
        Roll back to a previous weight set (N steps back in history).

        Returns: restored weights, or None if no history.
        """
        with Session(self.engine) as session:
            history = (
                session.query(AgentWeightHistory)
                .filter(AgentWeightHistory.agent_name == agent_name)
                .order_by(AgentWeightHistory.recorded_at.desc())
                .all()
            )

        if len(history) <= steps:
            LOGGER.warning("Not enough history to rollback %d steps for %s.", steps, agent_name)
            return None

        target = history[steps]
        weights = json.loads(target.weights_json)
        self.save_weights(agent_name, weights, trigger="rollback")
        LOGGER.info("Rolled back %s to weights from %s.", agent_name, target.recorded_at)
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_adjustments(
        self,
        current: dict[str, float],
        adjustments: dict[str, float],
    ) -> dict[str, float]:
        """Apply multiplier adjustments with step clamping."""
        new_weights = {}
        for key, val in current.items():
            multiplier = adjustments.get(key, 1.0)
            delta = val * (multiplier - 1.0)
            delta = max(-MAX_STEP, min(MAX_STEP, delta))
            new_val = val + delta
            new_weights[key] = round(max(MIN_WEIGHT, min(MAX_WEIGHT, new_val)), 4)
        return new_weights

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return weights
        return {k: round(v / total, 4) for k, v in weights.items()}

    @staticmethod
    def _changed_enough(old: dict[str, float], new: dict[str, float]) -> bool:
        """Return True if any weight changed by more than 0.5%."""
        for key in old:
            if key in new and abs(new[key] - old[key]) >= 0.005:
                return True
        return False

    def _weight_path(self, agent_name: str) -> Path:
        return self.weights_dir / f"{agent_name}_weights.json"

    def _get_defaults(self, agent_name: str) -> dict[str, float]:
        if agent_name == self.AGENT_CANDIDATE:
            return DEFAULT_CANDIDATE_WEIGHTS.copy()
        return DEFAULT_SETUP_WEIGHTS.copy()
