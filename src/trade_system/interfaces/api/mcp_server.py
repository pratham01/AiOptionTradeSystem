"""MCP (Model Context Protocol) server for external agent integration."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Callable
from collections.abc import AsyncIterator

LOGGER = logging.getLogger(__name__)


class MCPTool:
    """An MCP tool definition."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_schema(self) -> dict[str, Any]:
        """Return JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class MCPResource:
    """An MCP resource definition."""

    def __init__(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str,
        handler: Callable[..., Any],
    ) -> None:
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
        self.handler = handler

    def to_schema(self) -> dict[str, Any]:
        """Return schema for this resource."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


class MCPServer:
    """MCP Server implementation for trading system integration."""

    def __init__(self, name: str = "trade-system-mcp") -> None:
        self.name = name
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator to register an MCP tool."""
        def decorator(func: Callable) -> Callable:
            self._tools[name] = MCPTool(
                name=name,
                description=description,
                parameters=parameters or {},
                handler=func,
            )
            LOGGER.info("Registered MCP tool: %s", name)
            return func
        return decorator

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "application/json",
    ) -> Callable:
        """Decorator to register an MCP resource."""
        def decorator(func: Callable) -> Callable:
            self._resources[uri] = MCPResource(
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                handler=func,
            )
            LOGGER.info("Registered MCP resource: %s", uri)
            return func
        return decorator

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle an MCP JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "initialize":
                return self._handle_initialize(req_id)
            elif method == "tools/list":
                return self._handle_tools_list(req_id)
            elif method == "tools/call":
                return await self._handle_tool_call(params, req_id)
            elif method == "resources/list":
                return self._handle_resources_list(req_id)
            elif method == "resources/read":
                return self._handle_resource_read(params, req_id)
            else:
                return self._error_response(f"Unknown method: {method}", req_id)
        except Exception as e:
            LOGGER.error("Error handling request: %s", e)
            return self._error_response(str(e), req_id)

    def _handle_initialize(self, req_id: Any) -> dict[str, Any]:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.name,
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {},
                    "resources": {},
                },
            },
        }

    def _handle_tools_list(self, req_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        tools = [tool.to_schema() for tool in self._tools.values()]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools},
        }

    async def _handle_tool_call(
        self,
        params: dict[str, Any],
        req_id: Any,
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            return self._error_response(f"Tool not found: {tool_name}", req_id)

        tool = self._tools[tool_name]
        try:
            result = await tool.handler(**arguments) if asyncio.iscoroutinefunction(
                tool.handler
            ) else tool.handler(**arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}]
                },
            }
        except Exception as e:
            return self._error_response(f"Tool execution failed: {e}", req_id)

    def _handle_resources_list(self, req_id: Any) -> dict[str, Any]:
        """Handle resources/list request."""
        resources = [res.to_schema() for res in self._resources.values()]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"resources": resources},
        }

    def _handle_resource_read(
        self,
        params: dict[str, Any],
        req_id: Any,
    ) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")

        if uri not in self._resources:
            return self._error_response(f"Resource not found: {uri}", req_id)

        resource = self._resources[uri]
        try:
            result = resource.handler()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [{
                        "uri": uri,
                        "mimeType": resource.mime_type,
                        "text": json.dumps(result) if resource.mime_type == "application/json" else str(result),
                    }]
                },
            }
        except Exception as e:
            return self._error_response(f"Resource read failed: {e}", req_id)

    def _error_response(self, message: str, req_id: Any) -> dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32600,
                "message": message,
            },
        }

    async def run_stdio(self) -> None:
        """Run server in stdio mode for MCP."""
        LOGGER.info("Starting MCP server (stdio mode)")

        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        # Properly connect stdin to the stream reader
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request = json.loads(line.decode().strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                LOGGER.error("Invalid JSON: %s", e)
            except Exception as e:
                LOGGER.error("Server error: %s", e)


# Global server instance
mcp_server = MCPServer()


# Example tool registrations (can be moved to separate module)
@mcp_server.register_tool(
    name="get_market_data",
    description="Get current market data for a symbol",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock symbol"},
            "timeframe": {"type": "string", "description": "Timeframe (1m, 5m, 1h, etc.)"},
        },
        "required": ["symbol"],
    },
)
def get_market_data(symbol: str, timeframe: str = "1m") -> dict[str, Any]:
    """Placeholder for market data tool."""
    return {"symbol": symbol, "timeframe": timeframe, "status": "placeholder"}


@mcp_server.register_resource(
    uri="trade://positions",
    name="Current Positions",
    description="List of current open positions",
)
def get_positions() -> list[dict[str, Any]]:
    """Placeholder for positions resource."""
    return []


def run_main() -> None:
    """Entry point for trade-mcp CLI command."""
    import asyncio
    asyncio.run(mcp_server.run_stdio())
