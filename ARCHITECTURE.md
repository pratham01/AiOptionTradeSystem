# Trade System Architecture

## Overview

A scalable, event-driven algorithmic trading system with support for real-time data processing, signal generation, AI-powered decision making, and MCP server integration.

## Directory Structure

```
trade_system/
├── trade_system/              # Core package
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point
│   ├── cli.py                 # Main CLI commands
│   ├── config.py              # Configuration management
│   ├── models.py              # Pydantic/dataclass models
│   ├── logging_utils.py       # Logging utilities
│   │
│   ├── advisory/              # Advisory/AI LLM integration
│   ├── agent/                 # AI decision engine (NEW)
│   │   ├── decision_engine.py # Signal evaluation & decisions
│   │   ├── risk_manager.py    # Risk assessment
│   │   └── state.py           # Agent state persistence
│   ├── analysis/              # Market analysis tools
│   ├── api/                   # External API/MCP server (NEW)
│   │   ├── mcp_server.py    # MCP protocol implementation
│   │   └── routes.py          # HTTP API routes (FastAPI/Flask)
│   ├── backtesting/           # Backtesting engine
│   ├── brokers/               # Broker integrations (Fyers)
│   ├── core/                  # Core domain layer (NEW)
│   │   ├── events.py          # Event-driven architecture
│   │   ├── models.py          # Domain models (Signal, Trade)
│   │   └── signals.py         # Signal generation abstractions
│   ├── dashboard/             # Streamlit dashboards
│   ├── data/                  # Data management
│   ├── indicators/            # Technical indicators
│   ├── live/                  # Live trading engine
│   ├── notifications/         # Telegram notifications
│   ├── pipeline/              # Data processing pipelines (NEW)
│   ├── research/              # Research utilities
│   ├── signals/               # Signal management (NEW)
│   └── strategies/            # Trading strategies
│
├── scripts/                   # Executable scripts (MOVED from root)
├── tests/                     # Test suite
├── data/                      # Data storage
├── logs/                      # Log files
├── .secrets/                  # Secrets (tokens, keys)
├── docs/                      # Documentation
├── pyproject.toml             # Project configuration
└── README.md                  # Main documentation
```

## Key Architectural Components

### 1. Core Layer (`trade_system/core/`)

**Purpose**: Domain models and event-driven architecture

- **events.py**: Async event bus for decoupled communication
  - `EventBus`: Central pub/sub system
  - `MarketEvent`, `SignalEvent`, `NotificationEvent`: Event types
- **models.py**: Domain entities
  - `Signal`: Trading signals with metadata
  - `Trade`: Trade records with P&L tracking
  - `MarketData`: OHLCV data structure
  - `SignalType`: Enum for BUY/SELL/NEUTRAL
- **signals.py**: Signal generator abstractions
  - `SignalGenerator`: Base class for all generators
  - `CompositeSignalGenerator`: Weighted aggregation of multiple generators

### 2. Agent Module (`trade_system/agent/`)

**Purpose**: AI-powered decision engine for trade evaluation

- **decision_engine.py**: Main decision logic
  - `DecisionEngine`: Evaluates signals using rules + optional LLM
  - `Decision`: Structured decision output
  - `DecisionAction`: EXECUTE, HOLD, MODIFY, REJECT
- **risk_manager.py**: Risk assessment
  - `RiskManager`: Position sizing, daily limits, drawdown protection
  - `RiskAssessment`: Risk evaluation result
  - `RiskLimits`: Configurable risk parameters
- **state.py**: Agent state persistence
  - `AgentState`: Complete agent state (positions, P&L, history)
  - `StateManager`: JSON persistence for agent state

**Usage**:
```python
from trade_system.agent import DecisionEngine, RiskManager

risk_mgr = RiskManager(limits=RiskLimits(max_position_size=100))
engine = DecisionEngine(risk_manager=risk_mgr, enable_llm=True)
decision = await engine.decide(signal, context={"market_regime": "trending"})
```

### 3. API Module (`trade_system/api/`)

**Purpose**: External integration via MCP and HTTP

- **mcp_server.py**: Model Context Protocol server
  - `MCPServer`: MCP protocol implementation
  - `MCPTool`, `MCPResource`: MCP primitives
  - Stdio transport for MCP clients
- **routes.py**: HTTP API routes
  - `TradeAPI`: Framework-agnostic API handlers
  - `fastapi_routes()`: FastAPI application factory
  - `flask_routes()`: Flask application factory

**CLI Commands**:
```bash
trade-mcp     # Start MCP server (stdio mode)
trade-api     # Start HTTP API server (port 8000)
```

### 4. Pipeline Module (`trade_system/pipeline/`)

**Purpose**: Data processing workflows

- **processor.py**: Pipeline orchestration
  - `DataPipeline`: Real-time streaming pipeline
  - `BatchPipeline`: Historical batch processing
  - `PipelineStage`: Base class for stages
- **transformers.py**: Data transformations
  - `IndicatorTransform`: Add technical indicators
  - `ResampleTransform`: Change timeframe
  - `SignalTransform`: Generate signals from conditions
  - `FilterTransform`: Data filtering
  - `LagTransform`: Feature engineering

**Usage**:
```python
from trade_system.pipeline import DataPipeline, IndicatorTransform

pipeline = DataPipeline()
pipeline.add_stage(IndicatorTransform([
    {"name": "supertrend", "params": {"period": 7, "multiplier": 3}},
    {"name": "rsi", "params": {"period": 14}},
]))
result = pipeline.process(data)
```

### 5. Signals Module (`trade_system/signals/`)

**Purpose**: Signal generation management

- **generator.py**: Signal generator registry
  - `SignalGenerator`: Abstract base class
  - `SignalRegistry`: Plugin registry for generators
  - `register_signal_generator`: Decorator
- **manager.py**: Signal orchestration
  - `SignalManager`: Deduplication, cooldown, rate limiting
  - `SignalConfig`: Signal generation configuration

### 6. Scripts Directory (`scripts/`)

**Purpose**: Organized entry point scripts (moved from root)

Contains all previously root-level scripts:
- `fetch_historical_data.py`
- `run_*_backtest.py` (various backtest runners)
- `authenticate_fyers*.py`
- `automation_wrapper.py`

## Data Flow Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Broker    │────▶│  Data Pipeline │───▶│  Indicators │
│  (Fyers)    │     │             │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                       ┌──────────────────────────┼─────────────┐
                       │                          ▼             │
                       │  ┌─────────────┐    ┌─────────────┐   │
                       │  │  Signal     │◄───│  Signal     │   │
                       │  │  Generators │    │  Manager    │   │
                       │  └──────┬──────┘    └─────────────┘   │
                       │         │                              │
                       │         ▼                              │
                       │  ┌─────────────┐                      │
                       │  │   Agent     │◄── Risk Assessment   │
                       │  │  Decision   │                      │
                       │  │   Engine    │◄── LLM (optional)    │
                       │  └──────┬──────┘                      │
                       │         │                             │
                       │         ▼                             │
                       │  ┌─────────────┐                      │
                       └──┤   Event     ├──────────────────────┘
                          │    Bus      │
                          └──────┬──────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
   ┌─────────────┐      ┌─────────────┐       ┌─────────────┐
   │  Telegram   │      │  MCP Server │       │  HTTP API   │
   │  Notifier   │      │             │       │             │
   └─────────────┘      └─────────────┘       └─────────────┘
```

## Configuration

Environment variables in `.env`:

```bash
# Broker
FYERS_CLIENT_ID=xxx
FYERS_SECRET_KEY=xxx
FYERS_TOTP_SECRET=xxx

# Notifications
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx

# Agent (optional)
OPENAI_API_KEY=xxx
ANTHROPIC_API_KEY=xxx

# API Server (optional)
API_HOST=0.0.0.0
API_PORT=8000
```

## Installation

```bash
# Base installation
pip install -e .

# With API server support
pip install -e ".[api]"

# With LLM agent support
pip install -e ".[agent]"

# Development dependencies
pip install -e ".[dev]"

# All extras
pip install -e ".[api,agent,dev]"
```

## Usage Patterns

### 1. Live Trading with Agent

```python
import asyncio
from trade_system.live import LiveMarketDataService
from trade_system.agent import DecisionEngine, RiskManager
from trade_system.core.events import EventBus, EventType
from trade_system.notifications import TelegramNotifier

async def main():
    # Setup event bus
    bus = EventBus()

    # Setup agent
    risk_mgr = RiskManager()
    agent = DecisionEngine(risk_manager=risk_mgr, enable_llm=True)

    # Setup notifications
    notifier = TelegramNotifier(token, chat_id)

    # Subscribe to signals
    async def on_signal(event):
        decision = await agent.decide(event.signal, context)
        if decision.action.value == "EXECUTE":
            notifier.send(f"Trade: {decision.signal.symbol}")

    bus.subscribe(EventType.SIGNAL, on_signal)

    # Start live service
    service = LiveMarketDataService(...)
    await service.run_forever()

asyncio.run(main())
```

### 2. MCP Integration

```python
from trade_system.api import mcp_server

@mcp_server.register_tool(
    name="analyze_setup",
    description="Analyze a trade setup",
    parameters={...}
)
def analyze_setup(symbol: str, setup_type: str) -> dict:
    # Your analysis logic
    return {"recommendation": "proceed", "confidence": 0.85}
```

### 3. Signal Pipeline

```python
from trade_system.pipeline import DataPipeline
from trade_system.pipeline.transformers import (
    IndicatorTransform, SignalTransform
)

pipeline = DataPipeline()
pipeline.add_stage(IndicatorTransform([
    {"name": "supertrend", "params": {"period": 7, "multiplier": 3}},
]))
pipeline.add_stage(SignalTransform([
    {
        "indicator": "supertrend_direction",
        "operator": "gt",
        "value": 0,
        "signal": "bullish"
    },
]))
```

## Future Extensions

1. **Multi-broker support**: Add base broker interface in `brokers/base.py`
2. **WebSocket streaming**: Add WebSocket routes to `api/routes.py`
3. **ML models**: Add `models/` directory for trained model artifacts
4. **Database persistence**: Add `db/` module for SQL persistence
5. **Kubernetes deployment**: Add `deploy/k8s/` manifests
