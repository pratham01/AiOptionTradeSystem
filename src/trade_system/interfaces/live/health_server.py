"""Lightweight HTTP health server for Docker/Kubernetes probes.

Runs in a daemon thread, exposing:
- GET /health   — liveness probe (JSON)
- GET /ready    — readiness probe (JSON)
- GET /metrics  — Prometheus text format metrics

Usage:
    from trade_system.interfaces.live.health_server import start_health_server

    # Start in a daemon thread (non-blocking)
    start_health_server(port=9090)
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

LOGGER = logging.getLogger(__name__)


class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health/metrics endpoints."""

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond_health()
        elif self.path == "/ready":
            self._respond_ready()
        elif self.path == "/metrics":
            self._respond_metrics()
        else:
            self.send_error(404, "Not Found")

    def _respond_health(self) -> None:
        """Liveness probe — is the process alive?"""
        try:
            from trade_system.shared.observability.health import HEALTH_CHECKER
            result = HEALTH_CHECKER.check_all()
            status_code = 200 if result["status"] != "unhealthy" else 503
            self._send_json(result, status_code)
        except Exception as exc:
            self._send_json({"status": "unhealthy", "error": str(exc)}, 503)

    def _respond_ready(self) -> None:
        """Readiness probe — is the system ready to receive traffic?"""
        try:
            from trade_system.shared.observability.health import HEALTH_CHECKER
            result = HEALTH_CHECKER.check_all()
            # Ready only if all components are healthy
            status_code = 200 if result["status"] == "healthy" else 503
            self._send_json(result, status_code)
        except Exception as exc:
            self._send_json({"status": "not_ready", "error": str(exc)}, 503)

    def _respond_metrics(self) -> None:
        """Prometheus text format metrics export."""
        try:
            from trade_system.shared.observability.metrics import METRICS
            body = METRICS.to_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging to avoid log spam."""
        pass


def start_health_server(port: int = 9090, bind: str = "0.0.0.0") -> HTTPServer | None:
    """Start the health server in a daemon thread.

    Args:
        port: Port to listen on (default: 9090)
        bind: Address to bind to (default: 0.0.0.0)

    Returns:
        The HTTPServer instance, or None if the port is already in use.
    """
    try:
        server = HTTPServer((bind, port), _HealthHandler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="health-server",
            daemon=True,
        )
        thread.start()
        LOGGER.info("Health server started on %s:%d", bind, port)
        return server
    except OSError as exc:
        LOGGER.warning("Failed to start health server on port %d: %s", port, exc)
        return None
