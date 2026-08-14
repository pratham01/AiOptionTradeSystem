#!/bin/bash
# start_live_bot.sh
# This script starts the trade-system live bot.
# It is designed to be run via cron at 9:10 AM every weekday.

PROJECT_DIR="/Users/pratham/aitrade/trade_system_v2"
LOG_FILE="$PROJECT_DIR/live_bot.log"

cd "$PROJECT_DIR" || exit 1

echo "----------------------------------------" >> "$LOG_FILE"
echo "Starting live bot at $(date)" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

# Start the live bot process
/opt/anaconda3/bin/trade-system live >> "$LOG_FILE" 2>&1
