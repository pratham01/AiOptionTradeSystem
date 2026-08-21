"""Health check probes for Docker/Kubernetes readiness and liveness.

Provides component-level health status for:
- Database connectivity
- WebSocket connection state
- Broker token validity
- Overall system readiness

Usage:
    from trade_system.shared.observability.health import HealthChecker

    checker = HealthChecker()
    status = checker.check_all()
    # Returns: {"status": "healthy", "components": {...}, "timestamp": "..."}
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


class HealthChecker:
    """Component-level health probe aggregator."""

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._custom_checks: dict[str, callable] = {}

    def register_check(self, name: str, check_fn) -> None:
        """Register a custom health check function.

        Args:
            name: Component name (e.g., "websocket", "broker")
            check_fn: Callable that returns dict with at minimum {"status": "healthy"|"unhealthy"}
        """
        self._custom_checks[name] = check_fn

    def check_database(self) -> dict[str, Any]:
        """Check database connectivity and basic metrics."""
        try:
            from trade_system.domains.market_data.infrastructure.database.connection import get_db_health
            return get_db_health()
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}

    def check_all(self) -> dict[str, Any]:
        """Run all registered health checks and return aggregate status."""
        components: dict[str, Any] = {}
        overall_status = "healthy"

        # Database check (always run)
        components["database"] = self.check_database()
        if components["database"].get("status") != "healthy":
            overall_status = "degraded"

        # Custom checks
        for name, check_fn in self._custom_checks.items():
            try:
                result = check_fn()
                components[name] = result
                if result.get("status") not in ("healthy", "ok"):
                    overall_status = "degraded"
            except Exception as exc:
                components[name] = {"status": "unhealthy", "error": str(exc)}
                overall_status = "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "components": components,
        }

    def is_healthy(self) -> bool:
        """Quick boolean health check."""
        result = self.check_all()
        return result["status"] == "healthy"


# Global singleton
HEALTH_CHECKER = HealthChecker()
