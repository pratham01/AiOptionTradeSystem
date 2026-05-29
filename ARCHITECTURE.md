# Trade System Architecture v2

## Overview
A scalable, event-driven algorithmic trading system with support for real-time data processing, highly-tuned predictive signal generation (e.g. 1.5% Momentum Surges), and Dockerized cloud deployment.

## Containerized Deployment Model

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
- **Shared Volume**: Ensures historical OHLCV data fetched by `cron-tasks` is instantly queryable by the `dashboard` and `live-bot`.

## Key Architectural Components

### 1. Agent Module (`src/trade_system/application/agent/`)
**Purpose**: Predictive engines and specialized strategy agents.
- **`NextDayPredictorAgent`**: Evaluates F&O stocks using MTF (Multi-Timeframe) Supertrends, 1.5% Momentum Surge filters, and ATR-based SL/TP mapping. It bypasses systemic market regime filters for exceptional individual momentum.
- **`GammaBlastAgent`** & **`NiftyOptionBuyerAgent`**: Derivatives specific agents focused on index trading.

### 2. Infrastructure Layer (`src/trade_system/infrastructure/`)
**Purpose**: External integrations and persistence.
- **Brokers**: `fyers.py` handles API rate limits, TOTP authentication, and WebSocket streams.
- **Database**: SQLAlchemy models managing `ohlcv_15m`, `ohlcv_daily` tables.

### 3. Interfaces (`src/trade_system/interfaces/`)
**Purpose**: System entrypoints.
- **Live Flow**: `collector.py` aggregates ticks into bars and publishes to the Event Bus.
- **Dashboards**: Streamlit views combining Market Mood, Sector Scopes, and Agent logs.

## Data Flow Architecture

```
┌─────────────┐       ┌─────────────┐      ┌───────────────┐
│   Broker    │──────▶│ Live Stream │─────▶│ Bar Aggregator│
│  (Fyers)    │       │ (WebSocket) │      │ (1m, 5m, 15m) │
└─────────────┘       └─────────────┘      └───────┬───────┘
       ▲                                           │
       │ (EOD Fetch)                               ▼
┌─────────────┐                            ┌───────────────┐
│ Cron Tasks  │                            │ Event Bus     │
│ (Backfill)  │                            └───────┬───────┘
└──────┬──────┘                                    │
       │                                           ▼
┌──────▼──────┐                            ┌───────────────┐
│ SQLite DB   │◄───────────────────────────┤ Agents / Algos│
│ (ohlcv)     │                            └───────┬───────┘
└─────────────┘                                    │
       ▲                                           ▼
┌──────┴──────┐                            ┌───────────────┐
│ Streamlit   │                            │ Telegram      │
│ Dashboards  │                            │ Notifier      │
└─────────────┘                            └───────────────┘
```

## Future Extensions
1. **Multi-broker support**: Abstract Fyers into a generic `Broker` protocol.
2. **Kubernetes deployment**: Graduate from `docker-compose` to K8s StatefulSets for massive scale.
