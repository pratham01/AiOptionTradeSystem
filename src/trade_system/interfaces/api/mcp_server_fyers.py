#!/usr/bin/env python3
"""
Fyers MCP Server - Real-time market data and trading via MCP protocol.

This server exposes Fyers API functionality through MCP tools and resources.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trade_system.interfaces.api.mcp_server import MCPServer, mcp_server
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)

# Global broker instance (initialized on first use)
_broker: FyersBroker | None = None
_settings: Settings | None = None


def get_settings() -> Settings:
    """Load settings once."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def get_broker() -> FyersBroker:
    """Get or create Fyers broker instance."""
    global _broker
    if _broker is None:
        settings = get_settings()

        # Load token from file
        token_file = Path(".secrets/fyers_token.json")
        if not token_file.exists():
            raise RuntimeError("Fyers token not found. Run authenticate_fyers_totp.py first.")

        with open(token_file) as f:
            token_data = json.load(f)
            access_token = token_data.get("access_token", "")

        _broker = FyersBroker(
            client_id=settings.fyers.client_id,
            access_token=access_token,
            user_id=settings.fyers.user_id,
        )

        if not _broker.authenticate():
            raise RuntimeError("Fyers authentication failed. Token may be expired.")

        LOGGER.info("Fyers broker initialized for MCP server")

    return _broker


# ============================================================================
# MCP TOOLS - Real Fyers Integration
# ============================================================================

@mcp_server.register_tool(
    name="fyers_get_quotes",
    description="Get real-time quotes for NSE stocks from Fyers",
    parameters={
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols (e.g., ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ'])"
            }
        },
        "required": ["symbols"],
    },
)
def fyers_get_quotes(symbols: list[str]) -> dict[str, Any]:
    """Get real-time quotes for specified symbols from Fyers."""
    try:
        broker = get_broker()
        quotes = broker.get_quotes(symbols)

        # Format the response
        result = {}
        for symbol, data in quotes.items():
            result[symbol] = {
                "symbol": symbol,
                "last_price": float(data.get("lp", 0)),
                "open": float(data.get("open", 0)),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "close": float(data.get("close", 0)),
                "prev_close": float(data.get("prev_close_price", 0)),
                "volume": int(data.get("volume", 0)),
                "change": float(data.get("ch", 0)),
                "change_percent": float(data.get("chp", 0)),
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
                "timestamp": data.get("timestamp", ""),
            }

        return {
            "status": "success",
            "count": len(result),
            "quotes": result,
        }
    except Exception as e:
        LOGGER.error("Error fetching quotes: %s", e)
        return {"status": "error", "error": str(e)}


@mcp_server.register_tool(
    name="fyers_get_historical_data",
    description="Fetch historical OHLC data from Fyers",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock symbol (e.g., NSE:NIFTY50-INDEX)"
            },
            "resolution": {
                "type": "string",
                "description": "Time resolution: 1, 5, 15, 30, 60, D, W, M"
            },
            "range_from": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD)"
            },
            "range_to": {
                "type": "string",
                "description": "End date (YYYY-MM-DD)"
            },
        },
        "required": ["symbol", "resolution", "range_from", "range_to"],
    },
)
def fyers_get_historical_data(
    symbol: str,
    resolution: str,
    range_from: str,
    range_to: str,
) -> dict[str, Any]:
    """Fetch historical data from Fyers."""
    try:
        broker = get_broker()
        df = broker.fetch_history(
            symbol=symbol,
            resolution=resolution,
            range_from=range_from,
            range_to=range_to,
        )

        if df.empty:
            return {"status": "success", "count": 0, "data": []}

        # Convert DataFrame to list of records
        records = df.to_dict(orient="records")

        return {
            "status": "success",
            "symbol": symbol,
            "resolution": resolution,
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        LOGGER.error("Error fetching historical data: %s", e)
        return {"status": "error", "error": str(e)}


@mcp_server.register_tool(
    name="fyers_get_option_chain",
    description="Get option chain data for a symbol",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Underlying symbol (e.g., NSE:NIFTY50-INDEX)"
            },
            "strike_count": {
                "type": "integer",
                "description": "Number of strikes to fetch (default: 10)"
            },
        },
        "required": ["symbol"],
    },
)
def fyers_get_option_chain(symbol: str, strike_count: int = 10) -> dict[str, Any]:
    """Get option chain data from Fyers."""
    try:
        broker = get_broker()
        chain = broker.get_option_chain(symbol)

        if not chain:
            return {"status": "error", "error": "No option chain data available"}

        return {
            "status": "success",
            "symbol": symbol,
            "expiry_dates": chain.get("expiryDates", []),
            "strikes": chain.get("strikes", []),
            "data": chain.get("options", {}),
        }
    except Exception as e:
        LOGGER.error("Error fetching option chain: %s", e)
        return {"status": "error", "error": str(e)}


@mcp_server.register_tool(
    name="fyers_get_market_status",
    description="Check if NSE market is currently open",
    parameters={
        "type": "object",
        "properties": {},
    },
)
def fyers_get_market_status() -> dict[str, Any]:
    """Check NSE market status."""
    try:
        broker = get_broker()
        is_open, status = broker.is_market_open_today()

        return {
            "status": "success",
            "market_open": is_open,
            "market_status": status,
            "timestamp": asyncio.get_event_loop().time(),
        }
    except Exception as e:
        LOGGER.error("Error checking market status: %s", e)
        return {"status": "error", "error": str(e)}


@mcp_server.register_tool(
    name="fyers_get_top_gainers",
    description="Get top N gainers from predefined NSE stock universe",
    parameters={
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "Number of top gainers to return (default: 20)"
            },
            "min_price": {
                "type": "number",
                "description": "Minimum stock price filter (default: 50)"
            },
            "min_volume": {
                "type": "integer",
                "description": "Minimum volume filter (default: 100000)"
            },
        },
    },
)
def fyers_get_top_gainers(
    top_n: int = 20,
    min_price: float = 50.0,
    min_volume: int = 100000,
) -> dict[str, Any]:
    """Get top gainers from NSE stocks."""
    try:
        broker = get_broker()

        # Use the predefined stock list from nse_top_gainers_analysis
        from scripts.nse_top_gainers_analysis import NSE_STOCKS

        quotes = broker.get_quotes(NSE_STOCKS)

        # Calculate gains and filter
        stocks = []
        for symbol, data in quotes.items():
            try:
                close = float(data.get("lp", 0))
                prev_close = float(data.get("prev_close_price", 0))
                volume = int(data.get("volume", 0))

                if close < min_price or volume < min_volume:
                    continue

                change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0

                stocks.append({
                    "symbol": symbol,
                    "close": close,
                    "prev_close": prev_close,
                    "change_percent": round(change_pct, 2),
                    "volume": volume,
                    "open": float(data.get("open", 0)),
                    "high": float(data.get("high", 0)),
                    "low": float(data.get("low", 0)),
                })
            except (KeyError, ValueError):
                continue

        # Sort by change % descending
        stocks.sort(key=lambda x: x["change_percent"], reverse=True)

        return {
            "status": "success",
            "count": len(stocks[:top_n]),
            "top_gainers": stocks[:top_n],
            "timestamp": asyncio.get_event_loop().time(),
        }
    except Exception as e:
        LOGGER.error("Error fetching top gainers: %s", e)
        return {"status": "error", "error": str(e)}


# ============================================================================
# MCP RESOURCES
# ============================================================================

@mcp_server.register_resource(
    uri="fyers://market-status",
    name="Fyers Market Status",
    description="Current NSE market status from Fyers",
)
def resource_market_status() -> dict[str, Any]:
    """Resource: Current market status."""
    return fyers_get_market_status()


@mcp_server.register_resource(
    uri="fyers://nifty50-quote",
    name="NIFTY50 Quote",
    description="Real-time NIFTY50 index quote",
)
def resource_nifty50_quote() -> dict[str, Any]:
    """Resource: NIFTY50 current quote."""
    result = fyers_get_quotes(["NSE:NIFTY50-INDEX"])
    if result["status"] == "success":
        return result["quotes"].get("NSE:NIFTY50-INDEX", {})
    return result


@mcp_server.register_resource(
    uri="fyers://top-gainers",
    name="Top 10 Gainers",
    description="Top 10 gaining stocks right now",
)
def resource_top_gainers() -> dict[str, Any]:
    """Resource: Top 10 gainers."""
    return fyers_get_top_gainers(top_n=10)


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

def run_fyers_mcp_server() -> None:
    """Run the Fyers MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    LOGGER.info("Starting Fyers MCP Server...")

    # Pre-initialize broker to catch errors early
    try:
        get_broker()
        LOGGER.info("Fyers broker initialized successfully")
    except Exception as e:
        LOGGER.error("Failed to initialize Fyers broker: %s", e)
        LOGGER.error("Please run: python scripts/authenticate_fyers_totp.py")
        raise

    LOGGER.info("MCP Server ready. Tools available:")
    LOGGER.info("  - fyers_get_quotes: Get real-time stock quotes")
    LOGGER.info("  - fyers_get_historical_data: Fetch OHLC historical data")
    LOGGER.info("  - fyers_get_option_chain: Get options chain")
    LOGGER.info("  - fyers_get_market_status: Check market open/closed")
    LOGGER.info("  - fyers_get_top_gainers: Get top gaining stocks")

    # Run the server
    asyncio.run(mcp_server.run_stdio())


if __name__ == "__main__":
    run_fyers_mcp_server()
