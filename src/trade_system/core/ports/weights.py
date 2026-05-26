"""
Weights Provider Port (Layer 1).

Defines the interface for loading and saving evolved weights,
decoupling high-level agents from physical SQLite/Postgres database engines.
"""

from __future__ import annotations
from typing import Protocol

class WeightsProvider(Protocol):
    """Decouples agents from SQLite/Postgres weight persistence database engines."""
    
    def load_weights(self, agent_name: str) -> dict[str, float]:
        """
        Load weights for a given agent.

        Args:
            agent_name: Name of the agent (e.g., 'candidate_screener').

        Returns:
            Dictionary of feature weight mappings.
        """
        ...
        
    def save_weights(
        self,
        agent_name: str,
        weights: dict[str, float],
        win_rate: float | None = None,
        avg_pnl_pct: float | None = None,
        total_trades: int | None = None,
        trigger: str = "evolution_loop",
    ) -> None:
        """
        Persist weights for a given agent.

        Args:
            agent_name: Name of the agent.
            weights: Dictionary of feature weight mappings.
            win_rate: Optional session win rate performance.
            avg_pnl_pct: Optional session average pnl percentage.
            total_trades: Optional session total trades count.
            trigger: Trigger source name.
        """
        ...
