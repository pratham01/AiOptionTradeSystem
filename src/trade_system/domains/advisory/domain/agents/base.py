"""
BaseAgent — Industry-grade abstract contract for all trading agents.

Every agent in the system MUST inherit from this class.
This ensures consistent lifecycle management, healthchecks,
structured logging, and agentic event communication.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Import from models directly to avoid circular import with core/__init__.py
from trade_system.domains.trading.domain.models.domain import MarketContext


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"  # Blocked after repeated failures


@dataclass
class AgentResult:
    """
    Typed return value from every agent run.
    Forces agents to be explicit about success, data, and errors.
    """
    success: bool
    agent_name: str
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, agent_name: str, data: Any = None, metadata: dict | None = None, duration_ms: float = 0.0) -> "AgentResult":
        return cls(success=True, agent_name=agent_name, data=data, metadata=metadata or {}, duration_ms=duration_ms)

    @classmethod
    def fail(cls, agent_name: str, error: str, metadata: dict | None = None, duration_ms: float = 0.0) -> "AgentResult":
        return cls(success=False, agent_name=agent_name, error=error, metadata=metadata or {}, duration_ms=duration_ms)


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.

    Provides:
      - Consistent name and logger
      - Run lifecycle with automatic timing and error capture
      - Circuit-breaker pattern (disables agent after N consecutive failures)
      - Healthcheck interface

    Usage:
        class MyAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "MyAgent"

            async def _execute(self, context: MarketContext) -> AgentResult:
                # ... your agent logic here ...
                return AgentResult.ok(self.name, data=result)
    """

    # Circuit-breaker: disable after this many consecutive failures
    MAX_CONSECUTIVE_FAILURES: int = 5

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"agent.{self.name}")
        self._status: AgentStatus = AgentStatus.IDLE
        self._consecutive_failures: int = 0
        self._last_run_at: float | None = None
        self._last_error: str | None = None
        self._total_runs: int = 0
        self._total_failures: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable unique identifier for this agent."""
        ...

    @abstractmethod
    async def _execute(self, context: MarketContext) -> AgentResult:
        """
        The core agent logic. Override this — do NOT override `run()`.
        Must return an AgentResult.
        """
        ...

    async def run(self, context: MarketContext) -> AgentResult:
        """
        Public entry point. Wraps _execute() with:
        - Circuit-breaker guard
        - Automatic timing
        - Structured error capture
        - Consecutive failure tracking
        """
        if self._status == AgentStatus.CIRCUIT_OPEN:
            msg = f"[{self.name}] Circuit breaker is OPEN. Agent is disabled after {self._consecutive_failures} consecutive failures."
            self._logger.error(msg)
            return AgentResult.fail(self.name, error=msg)

        self._status = AgentStatus.RUNNING
        self._last_run_at = time.monotonic()
        start = time.monotonic()

        try:
            result = await self._execute(context)
            duration_ms = (time.monotonic() - start) * 1000
            result.duration_ms = duration_ms

            if result.success:
                self._consecutive_failures = 0
                self._status = AgentStatus.IDLE
                self._logger.info(
                    "[%s] ✅ Run succeeded in %.1fms", self.name, duration_ms
                )
            else:
                self._handle_failure(result.error or "Unknown error")

            self._total_runs += 1
            return result

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._handle_failure(str(exc))
            self._total_runs += 1
            self._logger.exception("[%s] ❌ Unhandled exception in agent run", self.name)
            return AgentResult.fail(self.name, error=str(exc), duration_ms=duration_ms)

    def _handle_failure(self, error: str) -> None:
        """Track consecutive failures and open circuit-breaker if needed."""
        self._consecutive_failures += 1
        self._total_failures += 1
        self._last_error = error
        self._logger.warning(
            "[%s] ⚠️ Failure #%d/%d: %s",
            self.name, self._consecutive_failures, self.MAX_CONSECUTIVE_FAILURES, error
        )

        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._status = AgentStatus.CIRCUIT_OPEN
            self._logger.error(
                "[%s] 🔴 CIRCUIT BREAKER OPENED after %d consecutive failures. "
                "Call reset_circuit() to re-enable.",
                self.name, self._consecutive_failures
            )
        else:
            self._status = AgentStatus.ERROR

    def reset_circuit(self) -> None:
        """Manually re-enable an agent after its circuit breaker has opened."""
        self._consecutive_failures = 0
        self._status = AgentStatus.IDLE
        self._logger.info("[%s] ⚡ Circuit breaker reset. Agent re-enabled.", self.name)

    async def healthcheck(self) -> bool:
        """
        Override this in agents with external dependencies (e.g., broker, DB).
        Default implementation checks the circuit-breaker status.
        """
        return self._status != AgentStatus.CIRCUIT_OPEN

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def stats(self) -> dict[str, Any]:
        """Returns runtime statistics for monitoring dashboards."""
        return {
            "name": self.name,
            "status": self._status.value,
            "total_runs": self._total_runs,
            "total_failures": self._total_failures,
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
            "failure_rate_pct": round(
                (self._total_failures / self._total_runs * 100) if self._total_runs > 0 else 0, 1
            ),
        }
