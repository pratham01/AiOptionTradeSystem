# Trade System v2 — Architecture Reference

## System Overview

A **Domain-Driven Design (DDD)** algorithmic trading platform for Indian derivatives (NSE F&O) with real-time WebSocket ingestion, multi-agent intelligence, and autonomous execution. The codebase follows clean architecture principles with strict separation between domain logic, application services, and infrastructure adapters.

---

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT LAYER                                     │
│                                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  live-bot   │  │  dashboard  │  │ st-scanner   │  │  cron-tasks         │ │
│  │  (Engine)   │  │  (Streamlit)│  │  (Telegram)  │  │  (EOD Batch)        │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘ │
│         │                │                │                     │            │
│         ▼                ▼                ▼                     ▼            │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                        Redis Event Bus                                │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│         │                │                │                     │            │
│         ▼                ▼                ▼                     ▼            │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │              Shared SQLite Volume (WAL mode) + CSV Storage            │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Domain Boundaries (DDD Bounded Contexts)

Each domain follows a strict layered structure: `domain/` → `application/` → `infrastructure/`

```
src/trade_system/
├── core/                           # System lifecycle & orchestration
│   └── orchestration/              # Agent lifecycle, startup sequencing
│
├── domains/
│   ├── advisory/                   # 🧠 Intelligence Layer
│   │   ├── domain/agents/          # Agent interfaces & protocols
│   │   ├── application/
│   │   │   ├── agent/              # Live Alert, Chief, Risk, Sanity agents
│   │   │   └── advisory/           # Advisory service coordination
│   │   └── infrastructure/         # Telegram, WhatsApp adapters
│   │
│   ├── analysis/                   # 📊 Analytical Engines
│   │   ├── domain/                 # Analysis domain models
│   │   ├── application/
│   │   │   ├── analysis/           # Smart OI, Midday Breakout, ISFRE, RRG
│   │   │   ├── backtesting/        # Strategy backtesting framework
│   │   │   ├── evolution/          # Parameter optimization & evolution
│   │   │   └── research/           # Research lab services
│   │   └── infrastructure/         # Data source adapters
│   │
│   ├── market_data/                # 📡 Market Data & Storage
│   │   ├── domain/schemas/         # OHLCV, tick, bar schemas
│   │   ├── application/            # BarStore, data sync services
│   │   └── infrastructure/
│   │       ├── database/           # SQLAlchemy connection (NullPool/WAL)
│   │       └── data/               # FO universe, historical fetchers
│   │
│   ├── strategy/                   # 🎯 Strategy Implementations
│   │   ├── domain/                 # Strategy protocols & registry
│   │   ├── application/
│   │   │   ├── indicators/         # SuperTrend, RSI, Bollinger, VWAP
│   │   │   ├── signals/            # Signal generation & scoring
│   │   │   └── strategies/         # SMC, ST Flip, Gamma Blast, ORB
│   │   └── infrastructure/         # External strategy data adapters
│   │
│   └── trading/                    # 💰 Execution & Risk
│       ├── domain/
│       │   ├── models/             # Position, Order, Trade domain models
│       │   └── ports/              # BrokerPort protocol (hexagonal arch)
│       ├── application/
│       │   ├── execution/          # Autonomous router, trailing stops
│       │   └── pipeline/           # Order flow pipeline
│       └── infrastructure/
│           └── brokers/
│               ├── fyers/          # FYERS API v3 adapter
│               ├── dhan/           # Dhan API adapter (backup)
│               └── legacy/         # Legacy broker implementation
│
├── interfaces/
│   ├── api/                        # REST/MCP API server
│   ├── cli/                        # CLI entrypoint (trade-system command)
│   ├── dashboard/                  # Streamlit Mission Control (20+ pages)
│   └── live/                       # Real-time WebSocket engine
│       ├── collector.py            # Main orchestrator (run_forever)
│       ├── websocket_adapter.py    # Fyers WS lifecycle
│       ├── bar_aggregator.py       # Tick → 1-min OHLCV bars
│       ├── pipelines/
│       │   ├── index_pipeline.py   # Full analytics (ST, RSI, SMC, Sniper)
│       │   └── fo_pipeline.py      # Lightweight F&O (RSI + Volume)
│       ├── option_chain_pipeline.py # OI forensics & delta tracking
│       └── health_server.py        # HTTP /health endpoint (port 9090)
│
├── shared/                         # Cross-cutting concerns
│   ├── config/                     # Settings, feature flags
│   ├── events/                     # Domain event definitions
│   ├── notifications/              # Notification service abstractions
│   ├── observability/              # Logging, metrics, tracing
│   ├── ports/                      # Shared port interfaces
│   └── utils/                      # Date, math, formatting helpers
│
└── modules/                        # Legacy module bridge (being migrated)
    ├── data/                       # Legacy data utilities
    └── processing/                 # Legacy processing utilities
```

---

## Live Data Pipeline

### Symbol Routing

```
Symbol ends with "-INDEX"  →  IndexPipeline (full institutional analytics)
  ● NSE:NIFTY50-INDEX, NSE:NIFTYBANK-INDEX, BSE:SENSEX-INDEX

Symbol does NOT end with "-INDEX"  →  FoPipeline (lightweight screening)
  ● NSE:RELIANCE-EQ, NSE:HDFCBANK-EQ, etc. (180+ F&O stocks)
```

### Index Pipeline Dataflow

```
WS tick
  → WebSocketAdapter._handle_message()
  → LiveMarketDataService._process_tick()  [zone alerts, ORB, position mgr]
  → BarAggregator.ingest_tick()
  → [minute boundary] BarAggregator._flush_symbol()
  → LiveMarketDataService._flush_symbol_minute()
      → BarStore.save_1min_bar()           [CSV + async DB]
      → BarStore.update_strategy_file()    [3-min CSV year file]
      → _on_bar_complete(symbol, bar)
          → IndexPipeline.on_bar()
              → calculate_supertrend()
              → _check_trend_change()       → Telegram (trend flip)
              → _check_rsi_divergence()     → Telegram (RSI div)
              → alert_agent.*              [SMC, Sniper, MWPL]
          → _maybe_set_first_15min_candle() [ORB tracking]
          → _check_sr_channel_touch()       [S/R alert]
          → _evaluate_gamma_blast()         [Gamma strategy]
          → _run_confirmed_strategy()       [Confirmed entry/exit]
```

### F&O Pipeline Dataflow

```
WS tick
  → BarAggregator.ingest_tick()
  → [minute boundary] _flush_symbol_minute()
      → BarStore.save_1min_bar()           [CSV + async DB]
      → _on_bar_complete(symbol, bar)
          → FoPipeline.on_bar()
              → _compute_rsi()             [RSI overbought/oversold]
              → volume surge detection     [20-bar SMA comparison]
              → _live_snapshot update      [feeds sector_scope_dashboard]
              → get_pending_alerts()       [collected, not sent unless flag]
```

---

## Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| `LiveMarketDataService` | `interfaces/live/collector.py` | **Thin orchestrator**: lifecycle, day-init, wires all modules |
| `WebSocketAdapter` | `interfaces/live/websocket_adapter.py` | Fyers WS connect/reconnect, token error detection, tick routing |
| `BarAggregator` | `interfaces/live/bar_aggregator.py` | Per-symbol tick buffer → 1-min OHLCV bar, volume delta |
| `IndexPipeline` | `interfaces/live/pipelines/index_pipeline.py` | Index analytics: SuperTrend, RSI divergence, SMC, Sniper |
| `FoPipeline` | `interfaces/live/pipelines/fo_pipeline.py` | F&O analytics: RSI extreme, volume surge, sector snapshot |
| `BarStore` | `domains/market_data/application/bar_store.py` | All OHLCV persistence: async DB write + CSV (1-min + 3-min) |
| `LiveAlertAgent` | `domains/advisory/application/agent/live_alert_agent.py` | Telegram dispatch with debounce + Thought Stream logging |
| `AutonomousRouter` | `domains/trading/application/execution/autonomous_router.py` | Signal → order routing, trailing stops, risk management |
| `FyersBroker` | `domains/trading/infrastructure/brokers/legacy/fyers.py` | Thread-safe REST client, idempotent token refresh |

---

## Dockerized Deployment Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     DOCKER HOST (VPS)                          │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  live-bot    │  │  dashboard   │  │  st-touch-scanner  │   │
│  │  (Python)    │  │  (Streamlit) │  │  (Telegram Bot)    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────────────┘   │
│         │                 │                  │                 │
│  ┌──────┴─────────────────┴──────────────────┴─────────────┐  │
│  │              trade_net (Bridge Network)                  │  │
│  └──────┬─────────────────┬──────────────────┬─────────────┘  │
│         │                 │                  │                 │
│  ┌──────▼───────┐  ┌──────▼──────┐  ┌───────▼──────────────┐  │
│  │  Redis       │  │ trade_data  │  │  cron-tasks          │  │
│  │  (Event Bus) │  │ (SQLite+CSV)│  │  (EOD Post-Market)   │  │
│  └──────────────┘  └─────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

Services:
  ├── redis           → Inter-agent event bus (port 6379)
  ├── live-bot        → Core real-time trading engine
  ├── sync-15m-db     → 15-minute historical data syncer
  ├── st-touch-scanner→ Supertrend Telegram scanner
  ├── dashboard       → Streamlit Mission Control (port 8502)
  └── cron-tasks      → Post-market EOD batch (4:15 PM IST)
```

---

## Separation of Concerns Summary

| Layer | Responsibility | Example |
|---|---|---|
| **Domain** | Business rules, models, ports | `Position`, `Order`, `BrokerPort` |
| **Application** | Use cases, orchestration | `AutonomousRouter`, `BarStore`, `LiveAlertAgent` |
| **Infrastructure** | External adapters | `FyersBroker`, `SQLAlchemy`, `Telegram` |
| **Interfaces** | User-facing endpoints | Dashboard, CLI, WebSocket, REST API |
| **Shared** | Cross-cutting | Config, events, logging, notifications |

---

## Future Extensions

1. **OptionChainPipeline**: Extract `_start_option_chain_thread` + cycle logic into its own module.
2. **BreakoutPipeline**: Extract `_start_breakout_thread` logic.
3. **GammaPipeline**: Extract `_evaluate_gamma_blast` + trade management.
4. **In-process Event Bus**: Replace callback dicts with a lightweight `EventBus` for fully decoupled publish/subscribe.
5. **Multi-broker support**: Abstract `FyersBroker` behind a `BrokerPort` protocol.
6. **Kubernetes migration**: Helm charts for horizontal scaling of scanners.
