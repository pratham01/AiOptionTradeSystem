"""HTTP API routes for external integration (FastAPI/Flask compatible)."""

from __future__ import annotations

from typing import Any
from datetime import datetime


class MockRequest:
    """Mock request for framework-agnostic handlers."""

    def __init__(self, method: str, path: str, query: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> None:
        self.method = method
        self.path = path
        self.query = query or {}
        self.body = body or {}


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
    """Trading system API handlers."""

    def __init__(self) -> None:
        self._routes: dict[str, Callable[[MockRequest], APIResponse]] = {
            "GET /health": self.health_check,
            "GET /market/status": self.market_status,
            "GET /positions": self.get_positions,
            "GET /signals": self.get_signals,
            "POST /signals/evaluate": self.evaluate_signal,
            "GET /agent/status": self.agent_status,
        }

    def get_handler(self, method: str, path: str) -> Callable[[MockRequest], APIResponse] | None:
        """Get handler for a route."""
        return self._routes.get(f"{method} {path}")

    def health_check(self, request: MockRequest) -> APIResponse:
        """Health check endpoint."""
        return APIResponse(data={
            "status": "healthy",
            "version": "0.1.0",
            "components": {
                "broker": "connected",
                "data_stream": "active",
                "notifications": "ready",
            },
        })

    def market_status(self, request: MockRequest) -> APIResponse:
        """Get current market status."""
        return APIResponse(data={
            "market_open": True,  # Placeholder
            "session": "regular",
            "next_close": "15:30:00",
        })

    def get_positions(self, request: MockRequest) -> APIResponse:
        """Get current positions."""
        return APIResponse(data={
            "positions": [],
            "total_pnl": 0.0,
        })

    def get_signals(self, request: MockRequest) -> APIResponse:
        """Get recent signals."""
        symbol = request.query.get("symbol")
        limit = int(request.query.get("limit", 10))

        # Placeholder - would query signal history
        return APIResponse(data={
            "signals": [],
            "symbol": symbol,
            "limit": limit,
        })

    def evaluate_signal(self, request: MockRequest) -> APIResponse:
        """Evaluate a potential trade signal."""
        signal_data = request.body.get("signal")
        if not signal_data:
            return APIResponse(error="Missing signal data", status=400)

        # Placeholder - would run through decision engine
        return APIResponse(data={
            "approved": True,
            "confidence": 0.75,
            "risk_score": 0.3,
            "suggested_size": 10,
        })

    def agent_status(self, request: MockRequest) -> APIResponse:
        """Get agent status and statistics."""
        return APIResponse(data={
            "agent_id": "main",
            "state": "active",
            "daily_pnl": 0.0,
            "open_positions": 0,
            "signals_today": 0,
        })


def create_app() -> TradeAPI:
    """Create and configure API."""
    return TradeAPI()


# Framework adapters

def fastapi_routes() -> list[dict[str, Any]]:
    """Generate FastAPI route definitions."""
    api = create_app()

    from fastapi import FastAPI, Request, HTTPException

    app = FastAPI(title="Trade System API", version="0.1.0")

    @app.get("/health")
    async def health():
        req = MockRequest("GET", "/health")
        resp = api.health_check(req)
        return resp.to_dict()

    @app.get("/market/status")
    async def market_status():
        req = MockRequest("GET", "/market/status")
        resp = api.market_status(req)
        return resp.to_dict()

    @app.get("/positions")
    async def positions():
        req = MockRequest("GET", "/positions")
        resp = api.get_positions(req)
        return resp.to_dict()

    @app.get("/signals")
    async def signals(symbol: str | None = None, limit: int = 10):
        req = MockRequest("GET", "/signals", query={"symbol": symbol, "limit": limit})
        resp = api.get_signals(req)
        return resp.to_dict()

    @app.post("/signals/evaluate")
    async def evaluate(request: Request):
        body = await request.json()
        req = MockRequest("POST", "/signals/evaluate", body=body)
        resp = api.evaluate_signal(req)
        if resp.error:
            raise HTTPException(status_code=resp.status, detail=resp.error)
        return resp.to_dict()

    @app.get("/agent/status")
    async def agent_status():
        req = MockRequest("GET", "/agent/status")
        resp = api.agent_status(req)
        return resp.to_dict()

    return app


def flask_routes() -> Any:
    """Generate Flask blueprint."""
    api = create_app()

    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        req = MockRequest("GET", "/health")
        resp = api.health_check(req)
        return jsonify(resp.to_dict())

    @app.route("/market/status", methods=["GET"])
    def market_status():
        req = MockRequest("GET", "/market/status")
        resp = api.market_status(req)
        return jsonify(resp.to_dict())

    @app.route("/positions", methods=["GET"])
    def positions():
        req = MockRequest("GET", "/positions")
        resp = api.get_positions(req)
        return jsonify(resp.to_dict())

    @app.route("/signals", methods=["GET"])
    def signals():
        req = MockRequest("GET", "/signals", query=request.args.to_dict())
        resp = api.get_signals(req)
        return jsonify(resp.to_dict())

    @app.route("/signals/evaluate", methods=["POST"])
    def evaluate():
        req = MockRequest("POST", "/signals/evaluate", body=request.get_json())
        resp = api.evaluate_signal(req)
        return jsonify(resp.to_dict()), resp.status

    @app.route("/agent/status", methods=["GET"])
    def agent_status():
        req = MockRequest("GET", "/agent/status")
        resp = api.agent_status(req)
        return jsonify(resp.to_dict())

    return app


def run_main() -> None:
    """Entry point for trade-api CLI command."""
    try:
        from fastapi import FastAPI
        import uvicorn
        app = fastapi_routes()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        from flask import Flask
        app = flask_routes()
        app.run(host="0.0.0.0", port=8000)
