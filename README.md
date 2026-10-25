# Trade System v2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-Domain--Driven%20Design%20(DDD)-success.svg)](ARCHITECTURE.md)
[![Broker](https://img.shields.io/badge/broker-FYERS%20API%20v3-orange.svg)](https://myapi.fyers.in/)
[![Dashboard](https://img.shields.io/badge/dashboard-Streamlit%20Mission%20Control-red.svg)](src/trade_system/interfaces/dashboard/main.py)
[![Docker](https://img.shields.io/badge/deployment-Docker%20Compose-2496ED.svg)](docker-compose.yml)

An enterprise-grade, institutional algorithmic trading platform and autonomous multi-agent swarm engineered for Indian derivatives (NSE F&O). Built on Domain-Driven Design (DDD), the system features real-time WebSocket ingestion, dynamic $\Delta\text{OI}$ option forensics, institutional market structure analysis, automated order routing with dynamic risk management, and a unified Streamlit Mission Control.

---

## 🏛️ Core Capabilities

### 1. ⚡ Autonomous Execution Cockpit & Router
- **Automated Signal Routing**: Wires institutional strategies directly to live execution via [autonomous_router.py](src/trade_system/domains/trading/application/execution/autonomous_router.py).
- **Two-Stage Trailing Stops**: Automates risk lifecycle:
  - **Stage 1 (Breakeven)**: Shifts stop loss to entry price once Target 1 is achieved.
  - **Stage 2 (Trailing Lock)**: Continuously trails profit dynamically towards Target 2.
- **Pre-Trade Risk Protections**: Duplicate position avoidance, max concurrent position capping, risk-per-share exposure rules, and symbol-specific lot sizing.
- **Emergency Panic Kill Switch**: Instantly pauses execution or flattens all active positions with single-button triggers in the dashboard.

### 2. 🎯 StockMojo Smart OI Terminal & Differential $\Delta\text{OI}$ Forensics
- **Differential $\Delta\text{OI}$ Forensics**: Tracks cross-snapshot open interest velocity ($\frac{\Delta\text{OI}}{\Delta t}$) and acceleration across consecutive intervals to detect institutional positioning before spot moves.
- **Max Pain Drift Vectors**: Dynamic calculation of writer loss surfaces to identify whether Max Pain is migrating up or down throughout the session.
- **Dynamic Call/Put Wall Shifts**: Real-time identification of wall migration, aggressive writer defense, and trapped institutional writers.
- **Price-Volume Divergence Engine**: Automatic classification of Bull Traps (Bearish Divergence) and Institutional Absorption (Bullish Divergence).

### 3. 🧠 Institutional Strategy Suite
- **Smart Money Concepts (SMC)**: Photon Market Structure, Order Blocks (OB), Fair Value Gaps (FVG), Change of Character (CHoCH), and Break of Structure (BOS) with confluence scoring via [smc_strategy.py](src/trade_system/domains/strategy/application/strategies/smc_strategy.py).
- **Midday Breakout Engine**: Detects afternoon volatility expansion following morning consolidation coils (10:30–12:45 IST) via [midday_breakout_engine.py](src/trade_system/domains/analysis/application/analysis/midday_breakout_engine.py).
- **Intraday Flow Reversal Engine (ISFRE)**: 4-quadrant derivative classification (Short Covering, Long Buildup, Long Unwinding, Short Buildup) via [intraday_flow_reversal.py](src/trade_system/domains/analysis/application/analysis/intraday_flow_reversal.py).
- **Supertrend MTF (ST Flip Engine)**: Multi-timeframe trend confluence tracking real-time 1m/3m/5m/15m Supertrend flips on NIFTY, BANKNIFTY, SENSEX, and top F&O stocks.
- **0DTE Gamma Blast & Intraday Option Edge**: Quantitative Gamma Exposure (GEX) and Delta Exposure (DEX) scoring with strike selection.
- **Wyckoff Method & VSA**: Volume Spread Analysis detecting accumulation, spring tests, markup, and distribution phases.
- **Next Day Predictor & BTST Scanner**: Post-market scan of 180+ F&O stocks looking for >1.5% momentum surges, volume spikes, and trend confluence.

### 4. 🚀 Streamlit Mission Control (Multi-Page UI)
The interactive web suite in [main.py](src/trade_system/interfaces/dashboard/main.py) provides 20+ purpose-built analytics surfaces:
- **Live Operations**:
  - `Autonomous Execution`: Live position matrix, order flow, 1-click test simulation, manual close, and emergency kill switches.
  - `Chief Trading Agent`: Daily market regime, macro briefing, and multi-agent coordination.
  - `Agent Dashboard`: Multi-agent swarm thought streams, status monitors, and Telegram dispatch log.
  - `Smart OI`: Real-time StockMojo terminal, strike heatmaps, and divergence alerts.
  - `Strike Supertrend`: Real-time Supertrend indicator on the near 5 strikes around spot price from the option chain, showing Call/Put direction, trailing stop levels, and multi-strike consensus.
  - `Sector Scope`: Live sector heatmaps, Sector Matrix & RRG (Relative Rotation Graph), and universe-wide VWAP.
  - `Broker Health`: FYERS token status, ping latency, and rate limits.
  - `Market Mood`: Advance/Decline ratios, aggregate PCR, and breadth analysis.
  - `Option Edge` & `BTST Scanner`: Real-time edge setups and overnight swing shortlist.
- **Research & Strategy Labs**:
  - `GEX & DEX Engine`, `Nifty Volatility Lab`, `Option Research Lab`, `Strategy Lab`, `Backtest Studio`, `Volumetric Order Flow`, and `Candlestick Pattern Lab`.

### 5. ⚡ Real-Time Pipeline Architecture
- **Zero-Touch Authentication**: Automatic TOTP authentication service with token caching and renewal.
- **Stateful Bar Aggregator**: Converts live Fyers WebSocket ticks into 1-minute and 3-minute OHLCV bars.
- **Dual Pipeline Routing**:
  - `IndexPipeline`: Full institutional treatment (SuperTrend, RSI divergence, SMC, S/R touch, Gamma Blast).
  - `FoPipeline`: Lightweight processing across 180+ F&O equities for sector matrix and surge alerts.
- **Centralized BarStore**: Asynchronous SQLite persistence with CSV fallback.

---

## 📁 Repository Structure (Domain-Driven Design)

```
trade_system_v2/
├── config/                          # Universe definitions & alert templates
│   ├── fo_universe.json             # F&O stock universe (180+ symbols)
│   ├── nse_universe.json            # NSE universe definitions
│   ├── alerts_config.yaml           # Telegram alert templates & routing
│   └── workflow.yaml                # Agent workflow configuration
│
├── data/                            # Runtime data (gitignored)
│   ├── trade_system.db              # Primary SQLite database
│   ├── fo_historical/               # F&O OHLCV CSV archive
│   ├── market_data/                 # Index/stock market data cache
│   ├── option_chain_data/           # Option chain snapshots
│   ├── sector_performance/          # Sector rotation data
│   ├── top_gainers/                 # Daily top gainers archive
│   └── watchlists/                  # Symbol watchlists
│
├── logs/                            # Centralized log directory (gitignored)
│   ├── trade_system.log             # Main app log (rotating, 10MB x 5)
│   ├── live_bot.log                 # ST Flip live bot log
│   ├── streamlit.log                # Dashboard log
│   └── fyersApi.log                 # Fyers SDK logs
│
├── scripts/                         # Operational runners & utilities
│   ├── start_app.sh                 # Unified launcher (dashboard + bots)
│   ├── start_live_bot.sh            # Cron-ready live bot launcher
│   ├── deploy.sh                    # Deployment script
│   ├── run_live_trading.py          # Live bot entrypoint
│   ├── run_live_st_scanner.py       # ST Telegram scanner entrypoint
│   ├── run_post_market_swarm.py     # EOD batch analysis
│   ├── backfill_*.py                # Historical data backfill scripts
│   ├── backtest_*.py                # Backtesting scripts
│   ├── authenticate_fyers*.py       # Fyers auth utilities
│   └── ... (scanners, sync, migration utilities)
│
├── src/trade_system/
│   ├── core/                        # System orchestrator & lifecycle
│   │   └── orchestration/           # Agent startup sequencing
│   │
│   ├── domains/                     # ── Bounded Contexts (DDD) ──
│   │   ├── advisory/               # Intelligence: Agents, alerts, dispatch
│   │   │   ├── domain/agents/       # Agent protocols & interfaces
│   │   │   ├── application/agent/   # Live Alert, Chief, Risk, Sanity agents
│   │   │   └── infrastructure/      # Telegram & WhatsApp adapters
│   │   │
│   │   ├── analysis/               # Analytics: OI forensics, engines
│   │   │   ├── application/analysis/    # Smart OI, ISFRE, Midday Breakout
│   │   │   ├── application/backtesting/ # Backtesting framework
│   │   │   ├── application/evolution/   # Parameter optimization
│   │   │   └── application/research/    # Research lab services
│   │   │
│   │   ├── market_data/            # Data: Ingestion, storage, sync
│   │   │   ├── application/         # BarStore, data sync services
│   │   │   └── infrastructure/
│   │   │       ├── database/        # SQLAlchemy (NullPool + WAL mode)
│   │   │       └── data/            # Historical fetchers, FO universe
│   │   │
│   │   ├── strategy/               # Strategies: Indicators, signals
│   │   │   ├── application/indicators/  # SuperTrend, RSI, VWAP, Bollinger
│   │   │   ├── application/signals/     # Signal scoring & generation
│   │   │   └── application/strategies/  # SMC, ST Flip, Gamma, ORB
│   │   │
│   │   └── trading/                # Execution: Orders, risk, broker
│   │       ├── domain/models/       # Position, Order domain models
│   │       ├── domain/ports/        # BrokerPort protocol (hexagonal)
│   │       ├── application/execution/   # Autonomous router, trailing stops
│   │       └── infrastructure/brokers/  # Fyers, Dhan adapters
│   │
│   ├── interfaces/
│   │   ├── api/                     # REST/MCP API server
│   │   ├── cli/                     # CLI entrypoint (trade-system cmd)
│   │   ├── dashboard/               # Streamlit Mission Control (20+ pages)
│   │   └── live/                    # WebSocket engine & pipelines
│   │       ├── collector.py         # Main orchestrator
│   │       ├── websocket_adapter.py # Fyers WS lifecycle
│   │       ├── bar_aggregator.py    # Tick → OHLCV bars
│   │       └── pipelines/           # Index & F&O processing pipelines
│   │
│   ├── shared/                      # Cross-cutting concerns
│   │   ├── config/                  # Settings & feature flags
│   │   ├── events/                  # Domain event definitions
│   │   ├── notifications/           # Notification abstractions
│   │   ├── observability/           # Logging, metrics, tracing
│   │   └── utils/                   # Date, math, formatting helpers
│   │
│   └── modules/                     # Legacy bridge (being migrated)
│
├── tests/                           # Test suite
│   ├── conftest.py                  # Pytest fixtures & shared setup
│   ├── test_*.py                    # Unit tests (auto-discovered by pytest)
│   ├── integration/                 # Integration tests (DB, Telegram, etc.)
│   ├── backtests/                   # Backtest-specific test scripts
│   ├── checks/                      # DB & API sanity check scripts
│   ├── scratch/                     # Ad-hoc / exploratory test scripts
│   ├── scripts/                     # Test-related utility scripts
│   └── data/                        # Test fixtures & mock data
│
├── reports/                         # Generated reports (gitignored)
│   ├── backtest/                    # Backtest HTML reports
│   └── smc/, fvg/, gamma_blast/    # Strategy-specific reports
│
├── scratch/                         # Ad-hoc experiments (gitignored)
│
├── docs/                            # Documentation
│   ├── MULTI_BROKER_ARCHITECTURE.md # Multi-broker design reference
│   └── option_buyer_rulebook.md     # Trading rules documentation
│
├── Dockerfile                       # Multi-stage Python container
├── docker-compose.yml               # Production multi-service stack
├── .env.example                     # Environment variable template
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata & tooling config
└── ARCHITECTURE.md                  # Detailed architecture reference
```

---

## 🚀 Quick Start

### Option A: Local Setup (Development)

#### 1. Prerequisites
- Python 3.11+
- FYERS API v3 account with App ID and Secret
- Redis (optional — required only for multi-process event-bus mode)

#### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/trade_system_v2.git
cd trade_system_v2

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Environment Configuration
```bash
# Create your .env from the template
cp .env.example .env

# Edit with your credentials
nano .env   # or use your preferred editor
```

Key variables to populate:
```ini
FYERS_CLIENT_ID="YOUR_APP_ID-100"
FYERS_SECRET_KEY="YOUR_SECRET_KEY"
FYERS_USER_ID="YOUR_FYERS_USER_ID"
FYERS_PIN="YOUR_4_DIGIT_PIN"
FYERS_TOTP_SECRET="YOUR_TOTP_BASE32_SECRET"

TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
```

#### 4. Broker Authentication
```bash
python3 -m trade_system auth --mode totp
```

#### 5. Run the System
```bash
# Start everything (dashboard + live bots) in one command:
./scripts/start_app.sh

# Or in daemon mode (all services background):
./scripts/start_app.sh --daemon
```

#### Service Management:
```bash
# Check status and health:
./scripts/start_app.sh --status

# Stop all services:
./scripts/start_app.sh --stop
```

The unified launcher will:
1. Terminate any stale background processes on ports `8501`, `8502`, and `9090`.
2. Validate and renew the FYERS broker access token automatically.
3. Start the **ST Telegram Scanner Bot** (`scripts/run_live_st_scanner.py`) in the background.
4. Start the **ST Flip Live Bot** (`scripts/run_live_trading.py`) in the background.
5. Launch the **Streamlit Mission Control Dashboard** at `http://localhost:8501`.

---

### Option B: Docker Deployment (Production / Cloud)

For 24/7 cloud server deployment (Linux VPS / AWS EC2 / DigitalOcean):

#### 1. Setup
```bash
# Clone and configure
git clone https://github.com/your-username/trade_system_v2.git
cd trade_system_v2

# Create environment file from template
cp .env.example .env
nano .env   # populate with your credentials
```

#### 2. Build & Run
```bash
# Build images and start all services
docker compose up -d --build

# Verify all containers are healthy
docker compose ps

# Follow logs of specific services
docker compose logs -f live-bot
docker compose logs -f dashboard
docker compose logs -f st-touch-scanner
```

#### 3. Access
```bash
# Dashboard URL (configurable via DASHBOARD_PORT in .env)
open http://YOUR_SERVER_IP:8502
```

#### 4. Management
```bash
# Stop all services
docker compose down

# Rebuild after code changes
docker compose up -d --build

# View resource usage
docker stats

# Access a running container shell
docker exec -it trade_bot_live bash
```

#### Deployed Services:
| Service | Container | Purpose | Port / Health |
|---|---|---|---|
| `redis` | `trade_redis` | Inter-agent event bus | 6379 (Redis ping) |
| `live-bot` | `trade_bot_live` | Core real-time trading engine | 9090 `/health` |
| `sync-15m-db` | `trade_bot_sync_15m` | 15-minute historical sync | 9090 `/health` |
| `st-touch-scanner` | `trade_bot_st_scanner` | Supertrend options scanner | 9090 `/health` |
| `dashboard` | `trade_bot_dashboard` | Streamlit Mission Control | 8502 `/_stcore/health` |
| `cron-tasks` | `trade_bot_cron` | EOD batch (4:15 PM IST, Mon-Fri) | Crond process |

#### Named Volumes:
| Volume | Purpose |
|---|---|
| `redis_data` | Redis persistence |
| `trade_data` | SQLite databases, CSV candle storage, option chain data |
| `trade_logs` | Application logs across all services |

---

### Option C: Standalone Component Startup

If you prefer launching components independently:

```bash
# Streamlit Mission Control
streamlit run src/trade_system/interfaces/dashboard/main.py --server.port 8501

# ST Telegram Scanner Bot
python3 scripts/run_live_st_scanner.py

# ST Flip Live Trading Bot
python3 scripts/run_live_trading.py
# Or via CLI module:
python3 -m trade_system live
```

---

## 🛠️ CLI Commands & Scanners

The unified CLI provides direct access to analytical scans, backtests, and system tools:

```bash
# Smart Money Concepts (SMC) Daily Scanner
python3 -m trade_system smc-scan --direction both --min-score 60 --top-n 15 --telegram

# Wyckoff Method & VSA Scanner
python3 -m trade_system wyckoff-scan --direction both --min-score 60 --top-n 15 --telegram

# Fair Value Gap (FVG) Daily Scanner
python3 -m trade_system fvg-scan --direction both --min-score 65 --top-n 15 --telegram

# Weekly Squeeze & Breakout Scanner
python3 -m trade_system weekly-breakout-scan --min-consolidation-weeks 4 --direction both

# Fetch Historical Candles
python3 -m trade_system fetch-history --symbol NSE:NIFTY50-INDEX --resolution D --from-date 2025-01-01

# Run Local Strategy Backtest
python3 -m trade_system backtest --strategy supertrend --csv data/fo_historical/NSE_NIFTY50-INDEX_15m.csv

# Database Sanity Check & Healing
python3 -m trade_system sanity-check
```

---

## ⚙️ Configuration Reference

### Feature Flags (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ENABLE_INTRADAY_OPTION_ALERTS` | `false` | Enable intraday option buying alerts (disabled — historically unprofitable) |
| `TELEGRAM_ENABLE_INTRADAY_OPTION_ALERTS` | `false` | Telegram dispatch for intraday option setups |
| `FYERS_ENABLED` | `true` | Enable FYERS broker |
| `DHAN_ENABLED` | `false` | Enable Dhan backup broker |
| `WHATSAPP_ENABLED` | `false` | Enable WhatsApp fallback notifications |
| `DASHBOARD_PORT` | `8502` | Streamlit dashboard port (Docker only) |

### Alert Configuration (`config/alerts_config.yaml`)

Controls per-strategy alert routing, cooldowns, and Telegram message templates. See [alerts_config.yaml](config/alerts_config.yaml) for the full schema.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_midday_breakout_engine.py
pytest tests/test_intraday_flow_reversal.py

# Run with verbose output
pytest tests/ -v --tb=short
```

---

## 📚 Further Reading

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Detailed pipeline dataflows, domain boundaries, component responsibilities, and deployment diagrams.
- **[config/alerts_config.yaml](config/alerts_config.yaml)** — Full alert template schema and channel routing.
- **[config/fo_universe.json](config/fo_universe.json)** — F&O symbol universe with sector mappings.
