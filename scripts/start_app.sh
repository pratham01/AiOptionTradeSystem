#!/bin/bash
# ==============================================================================
# AI Trade System v2 — Application & ST Telegram Bot Launcher
# ==============================================================================
# Usage:
#   ./scripts/start_app.sh          # Start ST bots in background + Streamlit foreground
#   ./scripts/start_app.sh --daemon # Start all services in background (daemon mode)
#   ./scripts/start_app.sh --stop   # Stop all running instances
#   ./scripts/start_app.sh --status # Check running status & health
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data"

ST_SCANNER_LOG="$PROJECT_DIR/logs/st_scanner.log"
LIVE_BOT_LOG="$PROJECT_DIR/logs/live_bot.log"
STREAMLIT_LOG="$PROJECT_DIR/logs/streamlit.log"

# Find python and streamlit executables
PYTHON_BIN="$(which python3 || which python)"
STREAMLIT_BIN="$(which streamlit)"

if [ -z "$STREAMLIT_BIN" ]; then
    if [ -x "/opt/anaconda3/bin/streamlit" ]; then
        STREAMLIT_BIN="/opt/anaconda3/bin/streamlit"
    fi
fi

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

stop_services() {
    echo -e "${YELLOW}Stopping existing Trade System processes...${NC}"
    
    # Kill running streamlit dashboard instances for this project
    PIDS_ST=$(pgrep -f "streamlit run src/trade_system/interfaces/dashboard/main.py" || true)
    if [ -n "$PIDS_ST" ]; then
        echo -e "Stopping Streamlit instances: $PIDS_ST"
        kill -9 $PIDS_ST 2>/dev/null || true
    fi

    # Kill live trading bot
    PIDS_BOT=$(pgrep -f "scripts/run_live_trading.py" || true)
    if [ -n "$PIDS_BOT" ]; then
        echo -e "Stopping Live Trading Bot instances: $PIDS_BOT"
        kill -9 $PIDS_BOT 2>/dev/null || true
    fi

    # Kill ST scanner bot
    PIDS_SCANNER=$(pgrep -f "scripts/run_live_st_scanner.py" || true)
    if [ -n "$PIDS_SCANNER" ]; then
        echo -e "Stopping ST Touch Scanner instances: $PIDS_SCANNER"
        kill -9 $PIDS_SCANNER 2>/dev/null || true
    fi
    echo -e "${GREEN}All processes stopped.${NC}"
}

check_status() {
    echo -e "${BLUE}=== Trade System Service Status ===${NC}"
    
    PIDS_ST=$(pgrep -f "streamlit run src/trade_system/interfaces/dashboard/main.py" || true)
    if [ -n "$PIDS_ST" ]; then
        echo -e "• Streamlit Dashboard:  ${GREEN}RUNNING${NC} (PID: $PIDS_ST)"
    else
        echo -e "• Streamlit Dashboard:  ${RED}STOPPED${NC}"
    fi

    PIDS_BOT=$(pgrep -f "scripts/run_live_trading.py" || true)
    if [ -n "$PIDS_BOT" ]; then
        echo -e "• ST Flip Live Bot:     ${GREEN}RUNNING${NC} (PID: $PIDS_BOT)"
    else
        echo -e "• ST Flip Live Bot:     ${RED}STOPPED${NC}"
    fi

    PIDS_SCANNER=$(pgrep -f "scripts/run_live_st_scanner.py" || true)
    if [ -n "$PIDS_SCANNER" ]; then
        echo -e "• ST Telegram Scanner:  ${GREEN}RUNNING${NC} (PID: $PIDS_SCANNER)"
    else
        echo -e "• ST Telegram Scanner:  ${RED}STOPPED${NC}"
    fi

    # Check health port 9090
    if curl -s -m 2 http://localhost:9090/health >/dev/null 2>&1; then
        echo -e "• Health API (9090):     ${GREEN}RESPONSIVE${NC}"
    fi

    # Check dashboard port 8501
    if curl -s -m 2 http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        echo -e "• Web Dashboard (8501): ${GREEN}RESPONSIVE (http://localhost:8501)${NC}"
    fi
}

case "$1" in
    --stop)
        stop_services
        exit 0
        ;;
    --status)
        check_status
        exit 0
        ;;
    --restart)
        stop_services
        sleep 2
        ;;
esac

# 1. Stop stale/hung instances before launching new ones
stop_services
sleep 1

echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}🚀 Starting Trade System v2 & ST Telegram Bot${NC}"
echo -e "${BLUE}======================================================${NC}"

# 2. Check / Renew Fyers Access Token
echo -e "${CYAN}[1/3] Verifying broker authentication...${NC}"
$PYTHON_BIN -c "from trade_system.shared.config import Settings; from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService; s = Settings.load(); a = FyersAuthService(s); token = a.get_valid_token(); print('✓ Fyers Token Active' if token else '✗ Fyers Token Missing')"

# 3. Start ST Telegram Scanner Bot in background
echo -e "${CYAN}[2/3] Starting ST Telegram Scanner Bot (scripts/run_live_st_scanner.py)...${NC}"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
nohup $PYTHON_BIN scripts/run_live_st_scanner.py >> "$ST_SCANNER_LOG" 2>&1 &
SCANNER_PID=$!
echo -e "${GREEN}✓ ST Telegram Scanner Bot running (PID: $SCANNER_PID, Log: logs/st_scanner.log)${NC}"

# 4. Start Live Trading Bot (ST Flip real-time collector) in background
echo -e "${CYAN}[3/3] Starting ST Flip Live Bot (scripts/run_live_trading.py)...${NC}"
nohup $PYTHON_BIN scripts/run_live_trading.py >> "$LIVE_BOT_LOG" 2>&1 &
BOT_PID=$!
echo -e "${GREEN}✓ ST Flip Live Bot running (PID: $BOT_PID, Log: logs/live_bot.log)${NC}"

# 5. Start Streamlit Application
if [ "$1" == "--daemon" ]; then
    echo -e "${CYAN}Starting Streamlit Application in background (daemon mode)...${NC}"
    nohup $STREAMLIT_BIN run src/trade_system/interfaces/dashboard/main.py --server.port 8501 --server.headless true >> "$STREAMLIT_LOG" 2>&1 &
    ST_PID=$!
    echo -e "${GREEN}✓ Streamlit Application running (PID: $ST_PID, Log: logs/streamlit.log)${NC}"
    echo -e "${GREEN}======================================================${NC}"
    echo -e "${GREEN}Mission Control URL: ${CYAN}http://localhost:8501${NC}"
    echo -e "${GREEN}======================================================${NC}"
else
    echo -e "${CYAN}Starting Streamlit Application in foreground (http://localhost:8501)...${NC}"
    echo -e "${YELLOW}(Press Ctrl+C to stop dashboard. Background bots will continue unless stopped with ./scripts/start_app.sh --stop)${NC}"
    $STREAMLIT_BIN run src/trade_system/interfaces/dashboard/main.py --server.port 8501
fi
