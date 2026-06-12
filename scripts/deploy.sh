#!/bin/bash
# ==============================================================================
# Automated Deployment Script for AI Trading System v2
# ==============================================================================
# This script builds and deploys the trading system containers.
# Run this on your cloud VPS.
# ==============================================================================

set -e

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Starting Deployment Sequence ===${NC}"

# 1. Verify Docker installation
if ! [ -x "$(command -v docker)" ]; then
    echo -e "${RED}Error: docker is not installed. Please install Docker first.${NC}" >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}Error: docker compose (V2) is not installed. Please install docker-compose-v2.${NC}" >&2
    exit 1
fi

# 2. Check for .env file
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env configuration file not found!${NC}"
    echo -e "${YELLOW}Please create a .env file containing your Fyers/Dhan keys and Telegram tokens in this directory.${NC}"
    exit 1
fi

# 3. Create persistent directories
echo -e "${GREEN}[1/4] Ensuring persistent data directory exists...${NC}"
mkdir -p data
mkdir -p logs

# 4. Stop existing containers (if any)
echo -e "${GREEN}[2/4] Stopping existing services...${NC}"
docker compose down || true

# 5. Build and launch containers
echo -e "${GREEN}[3/4] Building and launching containers in detached mode...${NC}"
docker compose up -d --build

# 6. Status check
echo -e "${GREEN}[4/4] Verification of running services...${NC}"
sleep 3

docker compose ps

echo -e "\n========================================================"
echo -e "${GREEN}🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
echo -e "========================================================"
echo -e "• Streamlit Dashboard:  ${BLUE}http://<your-server-ip>:8502${NC}"
echo -e "• To view live-bot logs: ${YELLOW}docker compose logs -f live-bot${NC}"
echo -e "• To view cron logs:     ${YELLOW}docker compose logs -f cron-tasks${NC}"
echo -e "========================================================"
