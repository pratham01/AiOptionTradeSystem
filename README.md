# Trade System v2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-Domain--Driven%20Design%20(DDD)-success.svg)](file:///Users/pratham/aitrade/trade_system_v2/ARCHITECTURE.md)
[![Broker](https://img.shields.io/badge/broker-FYERS%20API%20v3-orange.svg)](https://myapi.fyers.in/)
[![Dashboard](https://img.shields.io/badge/dashboard-Streamlit%20Mission%20Control-red.svg)](file:///Users/pratham/aitrade/trade_system_v2/src/trade_system/interfaces/dashboard/main.py)
[![Docker](https://img.shields.io/badge/deployment-Docker%20Compose-2496ED.svg)](file:///Users/pratham/aitrade/trade_system_v2/docker-compose.yml)

An enterprise-grade, institutional algorithmic trading platform and autonomous multi-agent swarm engineered for Indian derivatives (NSE F&O). Built on Domain-Driven Design (DDD), the system features real-time WebSocket ingestion, dynamic $\Delta\text{OI}$ option forensics, institutional market structure analysis, automated order routing with dynamic risk management, and a unified Streamlit Mission Control.

---

## 🏛️ Core Capabilities

### 1. ⚡ Autonomous Execution Cockpit & Router
- **Automated Signal Routing**: Wires institutional strategies directly to live execution via [autonomous_router.py](file:///Users/pratham/aitrade/trade_system_v2/src/trade_system/domains/trading/application/execution/autonomous_router.py).
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
- **Smart Money Concepts (SMC)**: Photon Market Structure, Order Blocks (OB), Fair Value Gaps (FVG), Change of Character (CHoCH), and Break of Structure (BOS) with confluence scoring via [smc_strategy.py](file:///Users/pratham/aitrade/trade_system_v2/src/trade_system/domains/strategy/application/strategies/smc_strategy.py).
- **Midday Breakout Engine**: Detects afternoon volatility expansion following morning consolidation coils (10:30–12:45 IST) via [midday_breakout_engine.py](file:///Users/pratham/aitrade/trade_system_v2/src/trade_system/domains/analysis/application/analysis/midday_breakout_engine.py).
- **Intraday Flow Reversal Engine (ISFRE)**: 4-quadrant derivative classification (Short Covering, Long Buildup, Long Unwinding, Short Buildup) via [intraday_flow_reversal.py](file:///Users/pratham/aitrade/trade_system_v2/src/trade_system/domains/analysis/application/analysis/intraday_flow_reversal.py).
- **Supertrend MTF (ST Flip Engine)**: Multi-timeframe trend confluence tracking real-time 1m/3m/5m/15m Supertrend flips on NIFTY, BANKNIFTY, SENSEX, and top F&O stocks.
- **0DTE Gamma Blast & Intraday Option Edge**: Quantitative Gamma Exposure (GEX) and Delta Exposure (DEX) scoring with strike selection.
- **Wyckoff Method & VSA**: Volume Spread Analysis detecting accumulation, spring tests, markup, and distribution phases.
- **Next Day Predictor & BTST Scanner**: Post-market scan of 180+ F&O stocks looking for >1.5% momentum surges, volume spikes, and trend confluence.

### 4. 🚀 Streamlit Mission Control (Multi-Page UI)
The interactive web suite in [main.py](file:///Users/pratham/aitrade/trade_system_v2/src/trade_system/interfaces/dashboard/main.py) provides 20+ purpose-built analytics surfaces:
- **Live Operations**:
  - `Autonomous Execution`: Live position matrix, order flow, 1-click test simulation, manual close, and emergency kill switches.
  - `Chief Trading Agent`: Daily market regime, macro briefing, and multi-agent coordination.
  - `Agent Dashboard`: Multi-agent swarm thought streams, status monitors, and Telegram dispatch log.
  - `Smart OI`: Real-time StockMojo terminal, strike heatmaps, and divergence alerts.
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
├── config/                      # Universe definitions (fo_universe.json) & settings
├── data/                        # Shared SQLite databases & historical candle storage
├── docker-compose.yml           # Production multi-container orchestration
├── Dockerfile                   # Python container specification
├── scripts/                     # Operational runners, backfillers, and schedulers
├── src/trade_system/
│   ├── core/                    # System orchestrator & agent lifecycle management
│   ├── domains/
│   │   ├── advisory/            # Agents (Chief, Live Alert, Risk, Sanity, SMC, Wyckoff)
│   │   ├── analysis/            # Analytical engines (Smart OI, Midday Breakout, ISFRE, RRG)
│   │   ├── market_data/         # WebSocket ingestion, bar store, and historical fetchers
│   │   ├── strategy/            # Strategy implementations & registry (SMC, ST, Gamma, ORB)
│   │   └── trading/             # Execution engine, position manager, autonomous router, broker
│   ├── interfaces/
│   │   ├── cli/                 # CLI entrypoint (trade-system command)
│   │   ├── dashboard/           # Streamlit Mission Control multi-page dashboard
│   │   └── live/                # Live WebSocket collectors and execution loops
│   └── shared/                  # Common domain models, events, logging, and configuration
└── tests/                       # Comprehensive unit and integration test suite
```

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.11+
- FYERS API v3 account with App ID and Secret
- Redis (optional for local standalone, required for event-bus multi-process setup)

### 2. Installation
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

### 3. Environment Configuration
Create your `.env` file from the template:
```bash
cp .env.example .env
```
Populate the essential variables:
```ini
FYERS_CLIENT_ID="YOUR_APP_ID-100"
FYERS_SECRET_KEY="YOUR_SECRET_KEY"
FYERS_USER_ID="YOUR_FYERS_USER_ID"
FYERS_PIN="YOUR_4_DIGIT_PIN"
FYERS_TOTP_SECRET="YOUR_TOTP_BASE32_SECRET"

TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
REDIS_URL="redis://localhost:6379/0"

# Telegram Alert Configuration (Intraday option buying is disabled by default)
ENABLE_INTRADAY_OPTION_ALERTS=false
```

### 4. Broker Authentication
Authenticate seamlessly with automated TOTP:
```bash
python3 -m trade_system auth --mode totp
```

---

## 🖥️ Running the System

### 1. Unified Application & ST Telegram Bot Launcher (Recommended)
Start the complete stack — Streamlit Mission Control dashboard, ST Telegram Touch Scanner Bot, and ST Flip Live Bot — with a single command:

```bash
# Start background ST bots + Streamlit dashboard in foreground:
./scripts/start_app.sh

# Or start all services in the background (daemon mode):
./scripts/start_app.sh --daemon
```

#### Manage Running Services:
```bash
# Check status and health endpoints:
./scripts/start_app.sh --status

# Stop all running instances (Streamlit, ST Scanner, Live Bot):
./scripts/start_app.sh --stop
```

The unified launcher will:
1. Gracefully terminate any stale background processes on ports `8501`, `8502`, and `9090`.
2. Validate and renew the FYERS broker access token automatically.
3. Start the **ST Telegram Scanner Bot** (`scripts/run_live_st_scanner.py`) in the background (logs to `logs/st_scanner.log`).
4. Start the **ST Flip Live Bot** (`scripts/run_live_trading.py`) in the background (logs to `logs/live_bot.log`).
5. Launch the **Streamlit Mission Control Dashboard** at `http://localhost:8501`.

---

### 2. Standalone Startup Commands

If you prefer launching components independently:

**Start Streamlit Mission Control Application:**
```bash
streamlit run src/trade_system/interfaces/dashboard/main.py --server.port 8501
```

**Start ST Telegram Scanner Bot:**
```bash
python3 scripts/run_live_st_scanner.py
```

**Start ST Flip Live Trading Bot:**
```bash
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

## ☁️ Cloud Deployment (Docker Compose)

For 24/7 cloud server deployment (Linux VPS / AWS EC2 / DigitalOcean), deploy the container stack via `docker-compose`:

```bash
# Build and run containers in background
docker compose up -d --build

# View container status
docker compose ps

# Follow logs of the live trading bot
docker compose logs -f live-bot
```

### Deployed Services:
| Service | Purpose | Port / Health |
|---|---|---|
| `redis` | Inter-agent event bus | Port 6379 (Redis ping) |
| `live-bot` | Core real-time trading engine & WebSocket client | HTTP 9090 health check |
| `sync-15m-db` | Automated 15-minute historical sync worker | HTTP 9090 health check |
| `st-touch-scanner`| Real-time Supertrend options scanner | HTTP 9090 health check |
| `dashboard` | Streamlit Mission Control web application | Port 8502 (`/_stcore/health`) |
| `cron-tasks` | Post-market EOD swarm scheduler (16:15 IST Mon-Fri) | Crond process |

---

## 🧪 Testing

Run unit and integration tests across all domains:
```bash
pytest tests/
```
Run specific module tests:
```bash
pytest tests/test_midday_breakout_engine.py
pytest tests/test_intraday_flow_reversal.py
```

---

## 📚 Architectural Reference
For in-depth diagrams, pipeline dataflows, and domain boundaries, see [ARCHITECTURE.md](file:///Users/pratham/aitrade/trade_system_v2/ARCHITECTURE.md).
