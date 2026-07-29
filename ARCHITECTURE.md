# Trade System Architecture v2

## Overview
A scalable, domain-driven algorithmic trading system with real-time data processing,
predictive signal generation, and Dockerized cloud deployment.

## Containerized Deployment

```
┌────────────────────────────────────────────────────────┐
│                   DOCKER HOST (VPS)                    │
│                                                        │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │ live-bot     │   │ dashboard    │   │ cron-tasks │  │
│  │ (Python)     │   │ (Streamlit)  │   │ (Cron/EOD) │  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬─────┘  │
│         │                  │                  │        │
│         ▼                  ▼                  ▼        │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Shared SQLite Volume (./data)           │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## Live Bot Module Map (v2 — Modular DDD)

The live data collection system is structured as a **pipeline of bounded contexts**:

```
Fyers WebSocket
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  interfaces/live/                                                   │
│                                                                     │
│  collector.py  ◄──── THIN ORCHESTRATOR                             │
│  (wires all components; owns run_forever() and lifecycle)          │
│                                                                     │
│  websocket_adapter.py  ← Connection lifecycle, token refresh       │
│         │                                                           │
│         │ on_tick(symbol, tick)                                     │
│         ▼                                                           │
│  bar_aggregator.py     ← Tick → 1-min OHLCV bar (stateful)        │
│         │                                                           │
│         │ on_bar_complete(symbol, bar)                              │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────┐                  │
│  │  _is_index(symbol)?                          │                  │
│  │       YES                    NO              │                  │
│  │        ▼                      ▼              │                  │
│  │  pipelines/              pipelines/          │                  │
│  │  index_pipeline.py       fo_pipeline.py      │                  │
│  │  (Full analytics)        (RSI + Volume)      │                  │
│  └──────────────────────────────────────────────┘                  │
│         │                      │                                   │
│         └──────────┬───────────┘                                   │
│                    ▼                                               │
│  domains/market_data/application/bar_store.py                      │
│  (BarStore: DB async write + CSV persistence)                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| `LiveMarketDataService` | `interfaces/live/collector.py` | **Thin orchestrator**: lifecycle, day-init, wires all modules |
| `WebSocketAdapter` | `interfaces/live/websocket_adapter.py` | Fyers WS connect/reconnect, token error detection, tick routing |
| `BarAggregator` | `interfaces/live/bar_aggregator.py` | Per-symbol tick buffer → 1-min OHLCV bar, volume delta |
| `IndexPipeline` | `interfaces/live/pipelines/index_pipeline.py` | Index analytics: SuperTrend flip, RSI divergence, SMC, Sniper |
| `FoPipeline` | `interfaces/live/pipelines/fo_pipeline.py` | F&O analytics: RSI extreme, volume surge, sector snapshot |
| `BarStore` | `domains/market_data/application/bar_store.py` | All OHLCV persistence: async DB write + CSV (1-min + 3-min files) |
| `LiveAlertAgent` | `domains/advisory/application/agent/live_alert_agent.py` | Telegram dispatch with debounce + Thought Stream logging |
| `FyersBroker` | `domains/trading/infrastructure/brokers/legacy/fyers.py` | Thread-safe REST client, idempotent token refresh |

---

## Symbol Routing Rules

```
Symbol ends with "-INDEX"  →  IndexPipeline (full treatment)
  ● NSE:NIFTY50-INDEX, NSE:NIFTYBANK-INDEX, BSE:SENSEX-INDEX

Symbol does NOT end with "-INDEX"  →  FoPipeline (lightweight)
  ● NSE:RELIANCE-EQ, NSE:HDFCBANK-EQ, etc.

Telegram alerts:
  ● Index symbols: always sent
  ● F&O symbols:  only if Settings.enable_fo_telegram_alerts = True
```

---

## Data Flow: Index Symbol (e.g. NIFTY50)

```
WS tick
  → WebSocketAdapter._handle_message()
  → LiveMarketDataService._process_tick()  [zone alerts, ORB, position manager]
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

## Data Flow: F&O Stock (e.g. RELIANCE)

```
WS tick
  → BarAggregator.ingest_tick()
  → [minute boundary] _flush_symbol_minute()
      → BarStore.save_1min_bar()           [CSV + async DB]
      → BarStore.update_strategy_file()    [3-min CSV year file]
      → _on_bar_complete(symbol, bar)
          → FoPipeline.on_bar()
              → _compute_rsi()             [RSI overbought / oversold]
              → volume surge detection     [20-bar SMA comparison]
              → _live_snapshot update      [feeds sector_scope_dashboard]
              → get_pending_alerts()       [collected, not sent unless flag=True]
```

---

## Domain Boundaries

```
domains/
├── market_data/          ← OHLCV data, storage, DB, FO universe
│   └── application/
│       └── bar_store.py  ← NEW: centralised persistence service
├── advisory/             ← Agents, alerts, strategies
│   └── application/agent/
│       └── live_alert_agent.py  ← Telegram + Thought Stream dispatcher
├── trading/              ← Broker clients, orders, risk management
│   └── infrastructure/brokers/legacy/fyers.py  ← Thread-safe broker
└── analytics/            ← Indicators, backtesting, screening
```

---

## Future Extensions

1. **OptionChainPipeline**: Extract `_start_option_chain_thread` + cycle logic into its own module.
2. **BreakoutPipeline**: Extract `_start_breakout_thread` logic.
3. **GammaPipeline**: Extract `_evaluate_gamma_blast` + trade management.
4. **In-process Event Bus**: Replace callback dicts with a lightweight `EventBus` for fully decoupled publish/subscribe.
5. **Multi-broker support**: Abstract `FyersBroker` behind a `BrokerPort` protocol.
