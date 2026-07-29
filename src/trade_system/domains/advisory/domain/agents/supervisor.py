"""
AgentSupervisor — Central health monitor and lifecycle manager for all agents.

In an industry-grade agentic system, no agent should run unmonitored.
The supervisor:
  1. Registers agents
  2. Periodically runs healthchecks on each
  3. Resets circuit breakers when safe to do so
  4. Provides a live status board (for the Dashboard)
  5. Emits a CRITICAL alert via Telegram if any agent is stuck or failing
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from trade_system.domains.advisory.domain.agents.base import BaseAgent, AgentStatus

LOGGER = logging.getLogger("AgentSupervisor")


class AgentSupervisor:
    """
    Manages the lifecycle of all registered agents.

    Example usage:
        supervisor = AgentSupervisor(notifier=telegram_notifier)
        supervisor.register(my_agent)
        supervisor.register(another_agent)

        # In your async main loop:
        asyncio.create_task(supervisor.run_healthcheck_loop())
    """

    def __init__(
        self,
        notifier: Any | None = None,
        healthcheck_interval_seconds: int = 60,
        auto_reset_after_seconds: int = 300,
    ) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._notifier = notifier
        self._healthcheck_interval = healthcheck_interval_seconds
        self._auto_reset_after = auto_reset_after_seconds
        self._circuit_open_since: dict[str, datetime] = {}
        self._is_running = False

    def register(self, agent: BaseAgent) -> None:
        """Register an agent with the supervisor."""
        if agent.name in self._agents:
            LOGGER.warning("Agent '%s' is already registered. Skipping.", agent.name)
            return
        self._agents[agent.name] = agent
        LOGGER.info("✅ Registered agent: %s", agent.name)

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from supervision."""
        self._agents.pop(agent_name, None)
        self._circuit_open_since.pop(agent_name, None)

    async def run_healthcheck_loop(self) -> None:
        """
        Continuously polls all registered agents for health.
        Run this as a background asyncio task.
        """
        self._is_running = True
        LOGGER.info("🏥 AgentSupervisor healthcheck loop started. Interval: %ds", self._healthcheck_interval)

        while self._is_running:
            await self._run_single_healthcheck_pass()
            await asyncio.sleep(self._healthcheck_interval)

    async def _run_single_healthcheck_pass(self) -> None:
        """Run one pass of healthchecks across all agents."""
        now = datetime.now()
        degraded_agents = []

        for name, agent in self._agents.items():
            try:
                is_healthy = await agent.healthcheck()

                if not is_healthy or agent.status == AgentStatus.CIRCUIT_OPEN:
                    if name not in self._circuit_open_since:
                        self._circuit_open_since[name] = now
                        LOGGER.error("🔴 Agent '%s' is UNHEALTHY (circuit open).", name)
                        self._alert(f"🔴 Agent UNHEALTHY: *{name}*\nLast Error: {agent._last_error}")

                    # Auto-reset after a cooldown period
                    since = self._circuit_open_since.get(name)
                    if since and (now - since).total_seconds() >= self._auto_reset_after:
                        LOGGER.info("🔁 Auto-resetting circuit breaker for '%s' after %ds cooldown.", name, self._auto_reset_after)
                        agent.reset_circuit()
                        self._circuit_open_since.pop(name, None)
                        self._alert(f"🟡 Auto-reset agent: *{name}*. Monitoring recovery.")

                    degraded_agents.append(name)
                else:
                    # Recovered
                    if name in self._circuit_open_since:
                        LOGGER.info("🟢 Agent '%s' has recovered.", name)
                        self._circuit_open_since.pop(name)
                        self._alert(f"🟢 Agent RECOVERED: *{name}*")

            except Exception as exc:
                LOGGER.exception("Healthcheck failed for agent '%s': %s", name, exc)

        if not degraded_agents:
            LOGGER.debug("All %d agents are healthy.", len(self._agents))

    def stop(self) -> None:
        """Stop the healthcheck loop."""
        self._is_running = False
        LOGGER.info("AgentSupervisor stopped.")

    def _alert(self, message: str) -> None:
        """Send a critical alert via the notifier (e.g., Telegram)."""
        if self._notifier:
            try:
                self._notifier.send(f"⚙️ <b>AgentSupervisor</b>\n{message}")
            except Exception as e:
                LOGGER.error("Failed to send supervisor alert: %s", e)

    @property
    def status_board(self) -> list[dict]:
        """
        Returns a list of stats for each agent.
        Used by the Streamlit dashboard to render the agent health table.
        """
        return [agent.stats for agent in self._agents.values()]

    @property
    def all_healthy(self) -> bool:
        return not self._circuit_open_since
