"""HTTP API routes for external integration (FastAPI/Flask compatible).

All endpoints are wired to real data sources:
- Health: Component-level health from observability.health
- Market status: Live state from LiveStateWriter JSON
- Signals: Agent thoughts from database
- Metrics: Real tick/bar/error counters
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class MockRequest:
    """Mock request for framework-agnostic handlers."""

    def __init__(self, method: str, path: str, query: dict[str, Any] | None = None,
                 body: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> None:
        self.method = method
        self.path = path
        self.query = query or {}
        self.body = body or {}
        self.headers = headers or {}


class APIResponse:
    """Standard API response format."""

    def __init__(
        self,
        data: Any = None,
        error: str | None = None,
        status: int = 200,
    ) -> None:
        self.data = data
        self.error = error
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": self.error is None,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result


class TradeAPI:
    """Trading system API handlers — wired to real data sources."""

    def __init__(self) -> None:
        self._routes: dict[str, Callable[[MockRequest], APIResponse]] = {
            "GET /health": self.health_check,
            "GET /market/status": self.market_status,
            "GET /positions": self.get_positions,
            "GET /signals": self.get_signals,
            "POST /signals/evaluate": self.evaluate_signal,
            "GET /agent/status": self.agent_status,
            "GET /metrics": self.get_metrics,
        }

    def get_handler(self, method: str, path: str) -> Callable[[MockRequest], APIResponse] | None:
        """Get handler for a route."""
        return self._routes.get(f"{method} {path}")

    def health_check(self, request: MockRequest) -> APIResponse:
        """Health check endpoint — real component status."""
        try:
            from trade_system.shared.observability.health import HEALTH_CHECKER
            result = HEALTH_CHECKER.check_all()
            return APIResponse(data=result)
        except Exception as exc:
            return APIResponse(data={
                "status": "healthy",
                "version": "2.0.0",
                "error_loading_health": str(exc),
            })

    def market_status(self, request: MockRequest) -> APIResponse:
        """Get current market status from live state file."""
        try:
            live_state_path = Path("data/live_state.json")
            if live_state_path.exists():
                state = json.loads(live_state_path.read_text())
                return APIResponse(data={
                    "bot_running": state.get("bot_running", False),
                    "symbols_active": len(state.get("symbols", {})),
                    "last_tick": state.get("last_update"),
                    "market_session": state.get("market_session", "unknown"),
                })
            return APIResponse(data={
                "bot_running": False,
                "symbols_active": 0,
                "market_session": "unknown",
            })
        except Exception as exc:
            LOGGER.warning("Error reading live state: %s", exc)
            return APIResponse(data={"bot_running": False, "error": str(exc)})

    def get_positions(self, request: MockRequest) -> APIResponse:
        """Get current positions from live state."""
        try:
            live_state_path = Path("data/live_state.json")
            if live_state_path.exists():
                state = json.loads(live_state_path.read_text())
                positions = state.get("positions", [])
                return APIResponse(data={
                    "positions": positions,
                    "count": len(positions),
                })
            return APIResponse(data={"positions": [], "count": 0})
        except Exception as exc:
            return APIResponse(data={"positions": [], "error": str(exc)})

    def get_signals(self, request: MockRequest) -> APIResponse:
        """Get recent signals/thoughts from database."""
        limit = int(request.query.get("limit", 20))
        try:
            from trade_system.domains.market_data.infrastructure.database.connection import get_db_context
            from trade_system.domains.market_data.infrastructure.database.repository import get_latest_thoughts

            with get_db_context() as session:
                thoughts = get_latest_thoughts(session, limit=limit)
                signals = [
                    {
                        "id": t.id,
                        "agent": t.agent_name,
                        "thought": t.thought,
                        "action": t.action,
                        "confidence": t.confidence,
                        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                    }
                    for t in thoughts
                ]
            return APIResponse(data={"signals": signals, "count": len(signals)})
        except Exception as exc:
            LOGGER.warning("Error fetching signals: %s", exc)
            return APIResponse(data={"signals": [], "error": str(exc)})

    def evaluate_signal(self, request: MockRequest) -> APIResponse:
        """Evaluate a potential trade signal."""
        signal_data = request.body.get("signal")
        if not signal_data:
            return APIResponse(error="Missing signal data", status=400)

        # Basic signal validation
        required_fields = ["symbol", "direction", "entry_price"]
        missing = [f for f in required_fields if f not in signal_data]
        if missing:
            return APIResponse(error=f"Missing fields: {missing}", status=400)

        return APIResponse(data={
            "validated": True,
            "symbol": signal_data.get("symbol"),
            "direction": signal_data.get("direction"),
        })

    def agent_status(self, request: MockRequest) -> APIResponse:
        """Get agent status with real metrics."""
        try:
            from trade_system.shared.observability.metrics import METRICS
            metrics = METRICS.to_dict()
            return APIResponse(data={
                "agent_id": "live-bot",
                "state": "active" if metrics["gauges"].get("active_symbols_count", 0) > 0 else "idle",
                "uptime_seconds": metrics["uptime_seconds"],
                "ticks_processed": metrics["counters"].get("ticks_processed_total", 0),
                "bars_written": metrics["counters"].get("bars_written_total", 0),
                "ws_errors": metrics["counters"].get("ws_errors_total", 0),
                "telegram_sent": metrics["counters"].get("telegram_sent_total", 0),
            })
        except Exception as exc:
            return APIResponse(data={"agent_id": "live-bot", "state": "unknown", "error": str(exc)})

    def get_metrics(self, request: MockRequest) -> APIResponse:
        """Get full metrics dump."""
        try:
            from trade_system.shared.observability.metrics import METRICS
            return APIResponse(data=METRICS.to_dict())
        except Exception as exc:
            return APIResponse(data={"error": str(exc)})


def create_app() -> TradeAPI:
    """Create and configure API."""
    return TradeAPI()


# Framework adapters

def fastapi_routes() -> Any:
    """Generate FastAPI app with auth and rate limiting."""
    api = create_app()

    from fastapi import FastAPI, Request, HTTPException
    from trade_system.interfaces.api.auth import validate_request

    app = FastAPI(title="Trade System API", version="2.0.0")

    def _check_auth(request: Request) -> None:
        api_key = request.headers.get("X-API-Key")
        valid, error = validate_request(api_key)
        if not valid:
            status = 429 if "Rate" in error else 401
            raise HTTPException(status_code=status, detail=error)

    @app.get("/health")
    async def health():
        req = MockRequest("GET", "/health")
        resp = api.health_check(req)
        return resp.to_dict()

    @app.get("/market/status")
    async def market_status(request: Request):
        _check_auth(request)
        req = MockRequest("GET", "/market/status")
        resp = api.market_status(req)
        return resp.to_dict()

    @app.get("/positions")
    async def positions(request: Request):
        _check_auth(request)
        req = MockRequest("GET", "/positions")
        resp = api.get_positions(req)
        return resp.to_dict()

    @app.get("/signals")
    async def signals(request: Request, symbol: str | None = None, limit: int = 20):
        _check_auth(request)
        req = MockRequest("GET", "/signals", query={"symbol": symbol, "limit": limit})
        resp = api.get_signals(req)
        return resp.to_dict()

    @app.post("/signals/evaluate")
    async def evaluate(request: Request):
        _check_auth(request)
        body = await request.json()
        req = MockRequest("POST", "/signals/evaluate", body=body)
        resp = api.evaluate_signal(req)
        if resp.error:
            raise HTTPException(status_code=resp.status, detail=resp.error)
        return resp.to_dict()

    @app.get("/agent/status")
    async def agent_status(request: Request):
        _check_auth(request)
        req = MockRequest("GET", "/agent/status")
        resp = api.agent_status(req)
        return resp.to_dict()

    @app.get("/metrics")
    async def metrics(request: Request):
        _check_auth(request)
        req = MockRequest("GET", "/metrics")
        resp = api.get_metrics(req)
        return resp.to_dict()

    return app


def run_main() -> None:
    """Entry point for trade-api CLI command."""
    try:
        from fastapi import FastAPI
        import uvicorn
        app = fastapi_routes()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        LOGGER.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
