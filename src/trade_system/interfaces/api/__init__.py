"""API module for MCP server and external integrations."""

from trade_system.interfaces.api.mcp_server import MCPServer, mcp_server
from trade_system.interfaces.api.routes import create_app

__all__ = ["MCPServer", "mcp_server", "create_app"]
