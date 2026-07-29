# Trade System v2

This repository is an enterprise-grade automated trading system for FYERS-based F&O workflows.

The system features:
- **Zero-Touch Authentication**: Automatic TOTP flow.
- **Next Day Predictive Engine**: Scans 180+ F&O stocks for >1.5% Momentum Surges, Supertrend MTF alignment, and Volume spikes.
- **Live Trading Bot**: Real-time 5m & 15m aggregated websocket monitoring.
- **Dockerized Architecture**: Built for 24/7 VPS Cloud deployment via `docker-compose`.
- **Streamlit Dashboards**: Beautiful, unified real-time tracking for Sectors, Breakouts, and Options flows.

## 🚀 Quick Start (Local Development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your `FYERS_CLIENT_ID`, `FYERS_SECRET_KEY`, `FYERS_TOTP_SECRET` and Telegram credentials to your `.env` file.

### Running Locally
**Start the Live Trading Bot:**
```bash
python3 -m trade_system live
```

**Start the Analytics Dashboard:**
```bash
streamlit run src/trade_system/interfaces/dashboard/main.py
```

## ☁️ Cloud Deployment (VPS)
We highly recommend deploying to a Linux VPS (DigitalOcean Droplet, AWS EC2, etc.) using Docker to keep your system running 24/7.

```bash
# Clone repository
git clone https://github.com/your-username/trade_system_v2.git
cd trade_system_v2

# Configure your secrets
nano .env

# Deploy using Docker Compose (sudo may be required depending on your user privileges)
docker compose up -d --build
```

This will automatically spin up three independent containers:
1. `live-bot`: Handles real-time monitoring and execution.
2. `dashboard`: Web UI available at `http://YOUR_SERVER_IP:8502`.
3. `cron-tasks`: Automatically runs post-market scanning and backfills every day at 16:15 IST.

*For more details, see the `deployment_guide.md` artifact.*

## 🧠 Core Strategy: Next Day Predictor
Our crown jewel is the `NextDayPredictorAgent`. It runs automatically after the market closes, analyzing the entire F&O universe.
It looks for:
- **Momentum Surges**: Stocks that moved >1.5% with high volume (dynamically bypassing broader Bear market regimes).
- **Trend Confluence**: Aligning Daily and 15m Supertrends.
- **Pattern Recognition**: Bullish Engulfing, Vol-Weighted Close, Monthly Breakouts.

It then pushes the highest conviction setups straight to your Telegram via the `TelegramNotifier`.

## 📁 Project Layout
- `src/trade_system/application/agent/`: AI and Predictive engines (NextDayPredictor, GammaBlast).
- `src/trade_system/interfaces/live/`: Live WebSocket collectors and Execution loops.
- `src/trade_system/interfaces/dashboard/`: Streamlit Web UIs.
- `src/trade_system/infrastructure/`: Databases (SQLite), Brokers (Fyers), Notifiers.
- `scratch/`: Research, one-off scripts, and backtests.
