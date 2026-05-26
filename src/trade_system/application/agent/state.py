"""Agent state management for persistence and context."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from trade_system.core import Signal, SignalType, Trade, TradeStatus

LOGGER = logging.getLogger(__name__)


@dataclass
class Position:
    """Current position state."""

    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: int
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None

    def update_price(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price
        if self.direction == "LONG":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity


@dataclass
class AgentState:
    """Complete agent state for persistence."""

    agent_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Portfolio
    cash: float = 100000.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0

    # Signals history (limited size)
    recent_signals: list[dict[str, Any]] = field(default_factory=list)
    max_signal_history: int = 100

    # Context
    market_regime: str = "unknown"
    last_analysis: dict[str, Any] = field(default_factory=dict)

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self.positions[position.symbol] = position
        self.updated_at = datetime.utcnow()

    def remove_position(self, symbol: str) -> Position | None:
        """Remove and return a position."""
        return self.positions.pop(symbol, None)

    def add_signal(self, signal: Signal) -> None:
        """Add signal to recent history."""
        self.recent_signals.append({
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "type": signal.signal_type.value,
            "confidence": signal.confidence,
        })
        # Trim history
        if len(self.recent_signals) > self.max_signal_history:
            self.recent_signals = self.recent_signals[-self.max_signal_history:]
        self.updated_at = datetime.utcnow()

    def update_daily_stats(self, pnl: float, win: bool) -> None:
        """Update daily statistics."""
        self.daily_pnl += pnl
        self.daily_trades += 1
        if win:
            self.daily_wins += 1
        else:
            self.daily_losses += 1
        self.updated_at = datetime.utcnow()

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "cash": self.cash,
            "positions": {
                symbol: {
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "entry_time": p.entry_time.isoformat(),
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                }
                for symbol, p in self.positions.items()
            },
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "recent_signals": self.recent_signals,
            "market_regime": self.market_regime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Deserialize from dictionary."""
        state = cls(
            agent_id=data["agent_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            cash=data.get("cash", 100000.0),
            daily_pnl=data.get("daily_pnl", 0.0),
            daily_trades=data.get("daily_trades", 0),
            daily_wins=data.get("daily_wins", 0),
            daily_losses=data.get("daily_losses", 0),
            recent_signals=data.get("recent_signals", []),
            market_regime=data.get("market_regime", "unknown"),
        )

        # Restore positions
        for symbol, p_data in data.get("positions", {}).items():
            state.positions[symbol] = Position(
                symbol=p_data["symbol"],
                direction=p_data["direction"],
                entry_price=p_data["entry_price"],
                quantity=p_data["quantity"],
                entry_time=datetime.fromisoformat(p_data["entry_time"]),
                current_price=p_data.get("current_price", 0.0),
                unrealized_pnl=p_data.get("unrealized_pnl", 0.0),
                stop_loss=p_data.get("stop_loss"),
                take_profit=p_data.get("take_profit"),
            )

        return state


class StateManager:
    """Manages agent state persistence."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_path(self, agent_id: str) -> Path:
        """Get path for agent state file."""
        return self.state_dir / f"{agent_id}_state.json"

    def save(self, state: AgentState) -> None:
        """Save agent state to disk."""
        path = self._get_state_path(state.agent_id)
        try:
            with open(path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            LOGGER.debug("Saved state for agent %s to %s", state.agent_id, path)
        except Exception as e:
            LOGGER.error("Failed to save state: %s", e)

    def load(self, agent_id: str) -> AgentState | None:
        """Load agent state from disk."""
        path = self._get_state_path(agent_id)
        if not path.exists():
            LOGGER.debug("No saved state found for agent %s", agent_id)
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            state = AgentState.from_dict(data)
            LOGGER.debug("Loaded state for agent %s from %s", agent_id, path)
            return state
        except Exception as e:
            LOGGER.error("Failed to load state: %s", e)
            return None

    def list_agents(self) -> list[str]:
        """List all saved agent IDs."""
        agents = []
        for path in self.state_dir.glob("*_state.json"):
            agents.append(path.stem.replace("_state", ""))
        return agents
