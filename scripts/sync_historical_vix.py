#!/usr/bin/env python3
"""
Sync Historical India VIX Data (NSE:INDIAVIX-INDEX) into SQLite Database (trade_system.db).
Fetches 15m and Daily candles from Fyers broker API and stores them in ohlcv_15m and ohlcv_daily.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import sqlite3
import pandas as pd

from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.shared.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def setup_broker() -> FyersBroker:
    settings = Settings.load()
    token_file = Path(".secrets/fyers_token.json")
    if not token_file.exists():
        raise FileNotFoundError("Token file .secrets/fyers_token.json not found.")
    token_data = json.loads(token_file.read_text())
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token_data.get("access_token", ""),
        user_id=settings.fyers.user_id,
    )
    if not broker.authenticate():
        raise RuntimeError("Fyers broker authentication failed.")
    return broker


def sync_vix_data(db_path: Path = Path("data/trade_system.db")):
    broker = setup_broker()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Sync 15m VIX candles in chunks of 90 days (Fyers limit is 100 days for intraday)
    vix_symbol = "NSE:INDIAVIX-INDEX"
    today = date.today()
    start_date = today - timedelta(days=365)  # Fetch past 1 year of 15m data

    total_15m_inserted = 0
    curr_start = start_date
    while curr_start < today:
        curr_end = min(curr_start + timedelta(days=90), today)
        LOGGER.info("Fetching 15m VIX from %s to %s...", curr_start, curr_end)
        try:
            res = broker.fyers.history({
                "symbol": vix_symbol,
                "resolution": "15",
                "date_format": "1",
                "range_from": str(curr_start),
                "range_to": str(curr_end),
                "cont_flag": "1"
            })
            candles = res.get("candles", [])
            if candles:
                rows_to_insert = []
                for c in candles:
                    # c: [epoch_sec, open, high, low, close, volume]
                    ts = datetime.fromtimestamp(c[0]).strftime("%Y-%m-%d %H:%M:%S.000000")
                    rows_to_insert.append((
                        vix_symbol,
                        ts,
                        float(c[1]),
                        float(c[2]),
                        float(c[3]),
                        float(c[4]),
                        float(c[5]) if len(c) > 5 else 0.0
                    ))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO ohlcv_15m (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows_to_insert)
                conn.commit()
                total_15m_inserted += len(rows_to_insert)
                LOGGER.info("Inserted %d 15m VIX candles.", len(rows_to_insert))
        except Exception as e:
            LOGGER.error("Error fetching 15m VIX for %s to %s: %s", curr_start, curr_end, e)
        
        curr_start = curr_end + timedelta(days=1)

    LOGGER.info("Total 15m VIX candles inserted: %d", total_15m_inserted)

    # 2. Sync Daily VIX candles (Past 3 years in 360-day chunks)
    total_daily_inserted = 0
    d_start = today - timedelta(days=365 * 3)
    curr_d_start = d_start
    while curr_d_start < today:
        curr_d_end = min(curr_d_start + timedelta(days=360), today)
        LOGGER.info("Fetching Daily VIX from %s to %s...", curr_d_start, curr_d_end)
        try:
            res = broker.fyers.history({
                "symbol": vix_symbol,
                "resolution": "D",
                "date_format": "1",
                "range_from": str(curr_d_start),
                "range_to": str(curr_d_end),
                "cont_flag": "1"
            })
            candles = res.get("candles", [])
            if candles:
                rows_to_insert = []
                for c in candles:
                    ts = datetime.fromtimestamp(c[0]).strftime("%Y-%m-%d 00:00:00.000000")
                    rows_to_insert.append((
                        vix_symbol,
                        ts,
                        float(c[1]),
                        float(c[2]),
                        float(c[3]),
                        float(c[4]),
                        float(c[5]) if len(c) > 5 else 0.0
                    ))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO ohlcv_daily (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows_to_insert)
                conn.commit()
                total_daily_inserted += len(rows_to_insert)
                LOGGER.info("Inserted %d Daily VIX candles.", len(rows_to_insert))
        except Exception as e:
            LOGGER.error("Error fetching Daily VIX for %s to %s: %s", curr_d_start, curr_d_end, e)

        curr_d_start = curr_d_end + timedelta(days=1)

    LOGGER.info("Total Daily VIX candles inserted: %d", total_daily_inserted)
    conn.close()


if __name__ == "__main__":
    sync_vix_data()
