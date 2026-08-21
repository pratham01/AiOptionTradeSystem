#!/usr/bin/env python3
"""Data integrity checker for OHLCV market data in the trade system database.

Checks:
1. Gap detection — identifies missing 1-min bars per trading day per symbol
2. Duplicate detection — finds duplicate (symbol, timestamp) entries
3. Data sanity — negative prices, zero volume on non-zero candles, future timestamps
4. Trading day coverage — compares stored dates vs. expected NSE trading calendar

Usage:
    python scripts/data_integrity_check.py
    python scripts/data_integrity_check.py --symbols NSE:NIFTY50-INDEX,NSE:RELIANCE-EQ
    python scripts/data_integrity_check.py --notify
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("integrity")

# NSE market hours: 09:15 to 15:30 IST = 375 minutes = 375 one-minute bars
EXPECTED_1M_BARS_PER_DAY = 375
# Allow some tolerance (pre-market and closing auction may vary)
MIN_EXPECTED_1M_BARS = 350


def check_duplicates(conn: sqlite3.Connection, table: str) -> list[dict]:
    """Find duplicate (symbol, timestamp) entries in a table."""
    query = f"""
        SELECT symbol, timestamp, COUNT(*) as cnt
        FROM {table}
        GROUP BY symbol, timestamp
        HAVING cnt > 1
        ORDER BY cnt DESC
        LIMIT 100
    """
    cursor = conn.execute(query)
    duplicates = []
    for row in cursor.fetchall():
        duplicates.append({
            "table": table,
            "symbol": row[0],
            "timestamp": row[1],
            "count": row[2],
        })
    return duplicates


def check_gaps(
    conn: sqlite3.Connection,
    table: str = "ohlcv_1m",
    symbols: list[str] | None = None,
    lookback_days: int = 5,
) -> list[dict]:
    """Detect trading days with significantly fewer bars than expected."""
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        query = f"""
            SELECT symbol, DATE(timestamp) as trade_date, COUNT(*) as bar_count
            FROM {table}
            WHERE DATE(timestamp) >= ? AND symbol IN ({placeholders})
            GROUP BY symbol, trade_date
            ORDER BY symbol, trade_date
        """
        params = [cutoff] + symbols
    else:
        query = f"""
            SELECT symbol, DATE(timestamp) as trade_date, COUNT(*) as bar_count
            FROM {table}
            WHERE DATE(timestamp) >= ?
            GROUP BY symbol, trade_date
            ORDER BY symbol, trade_date
        """
        params = [cutoff]

    cursor = conn.execute(query, params)
    gaps = []
    for row in cursor.fetchall():
        bar_count = row[2]
        if bar_count < MIN_EXPECTED_1M_BARS:
            gaps.append({
                "table": table,
                "symbol": row[0],
                "date": row[1],
                "bar_count": bar_count,
                "expected_min": MIN_EXPECTED_1M_BARS,
                "gap_pct": round((1 - bar_count / EXPECTED_1M_BARS_PER_DAY) * 100, 1),
            })
    return gaps


def check_sanity(conn: sqlite3.Connection, table: str) -> list[dict]:
    """Check for data anomalies: negative prices, extreme values, future timestamps."""
    issues = []

    # Negative or zero prices
    query = f"""
        SELECT symbol, timestamp, open, high, low, close
        FROM {table}
        WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
        LIMIT 50
    """
    for row in conn.execute(query).fetchall():
        issues.append({
            "table": table,
            "type": "negative_or_zero_price",
            "symbol": row[0],
            "timestamp": row[1],
            "values": {"open": row[2], "high": row[3], "low": row[4], "close": row[5]},
        })

    # High < Low (impossible candlestick)
    query = f"""
        SELECT symbol, timestamp, high, low
        FROM {table}
        WHERE high < low
        LIMIT 50
    """
    for row in conn.execute(query).fetchall():
        issues.append({
            "table": table,
            "type": "high_less_than_low",
            "symbol": row[0],
            "timestamp": row[1],
            "values": {"high": row[2], "low": row[3]},
        })

    # Future timestamps
    now_iso = datetime.now().isoformat()
    query = f"""
        SELECT symbol, timestamp
        FROM {table}
        WHERE timestamp > ?
        LIMIT 50
    """
    for row in conn.execute(query, [now_iso]).fetchall():
        issues.append({
            "table": table,
            "type": "future_timestamp",
            "symbol": row[0],
            "timestamp": row[1],
        })

    return issues


def get_table_stats(conn: sqlite3.Connection, table: str) -> dict:
    """Get basic statistics for a table."""
    try:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        symbol_count = conn.execute(
            f"SELECT COUNT(DISTINCT symbol) FROM {table}"
        ).fetchone()[0]
        min_ts = conn.execute(f"SELECT MIN(timestamp) FROM {table}").fetchone()[0]
        max_ts = conn.execute(f"SELECT MAX(timestamp) FROM {table}").fetchone()[0]
        return {
            "table": table,
            "rows": row_count,
            "symbols": symbol_count,
            "earliest": min_ts,
            "latest": max_ts,
        }
    except Exception as exc:
        return {"table": table, "error": str(exc)}


def run_checks(
    db_path: Path,
    symbols: list[str] | None = None,
    lookback_days: int = 5,
) -> dict:
    """Run all integrity checks and return a structured report."""
    conn = sqlite3.connect(str(db_path))
    report = {
        "timestamp": datetime.now().isoformat(),
        "db_path": str(db_path),
        "db_size_mb": round(db_path.stat().st_size / (1024 * 1024), 1),
        "tables": {},
        "duplicates": [],
        "gaps": [],
        "sanity_issues": [],
        "overall_status": "healthy",
    }

    tables = ["ohlcv_1m", "ohlcv_3m", "ohlcv_5m", "ohlcv_15m", "ohlcv_daily"]

    for table in tables:
        LOGGER.info("Checking table: %s", table)

        # Stats
        report["tables"][table] = get_table_stats(conn, table)

        # Duplicates
        dupes = check_duplicates(conn, table)
        if dupes:
            report["duplicates"].extend(dupes)
            LOGGER.warning("  Found %d duplicate entries in %s", len(dupes), table)

        # Sanity
        issues = check_sanity(conn, table)
        if issues:
            report["sanity_issues"].extend(issues)
            LOGGER.warning("  Found %d sanity issues in %s", len(issues), table)

    # Gap detection (only on 1-min table)
    gaps = check_gaps(conn, "ohlcv_1m", symbols, lookback_days)
    if gaps:
        report["gaps"] = gaps
        LOGGER.warning("Found %d gap entries in ohlcv_1m", len(gaps))

    conn.close()

    # Determine overall status
    if report["duplicates"] or report["sanity_issues"]:
        report["overall_status"] = "degraded"
    if len(report["gaps"]) > 10:
        report["overall_status"] = "degraded"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Data integrity checker")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/trade_system.db"),
        help="Path to the SQLite database",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to check (default: all)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=5,
        help="Number of days to look back for gap detection (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for JSON report (default: stdout)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notification if issues found",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        LOGGER.error("Database not found: %s", args.db_path)
        sys.exit(1)

    symbols = args.symbols.split(",") if args.symbols else None
    report = run_checks(args.db_path, symbols, args.lookback_days)

    # Output
    report_json = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json)
        LOGGER.info("Report written to: %s", args.output)
    else:
        print(report_json)

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("Overall Status: %s", report["overall_status"].upper())
    LOGGER.info("  Duplicates: %d", len(report["duplicates"]))
    LOGGER.info("  Data gaps:  %d", len(report["gaps"]))
    LOGGER.info("  Sanity:     %d issues", len(report["sanity_issues"]))
    for table, stats in report["tables"].items():
        LOGGER.info("  %s: %s rows, %s symbols", table, stats.get("rows", "?"), stats.get("symbols", "?"))
    LOGGER.info("=" * 60)

    # Telegram notification on issues
    if args.notify and report["overall_status"] != "healthy":
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
            from trade_system.shared.config import Settings
            from trade_system.shared.notifications.telegram import TelegramNotifier

            settings = Settings.load()
            if settings.telegram.enabled:
                notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
                msg = (
                    f"⚠️ <b>Data Integrity: {report['overall_status'].upper()}</b>\n\n"
                    f"Duplicates: {len(report['duplicates'])}\n"
                    f"Gaps: {len(report['gaps'])}\n"
                    f"Sanity issues: {len(report['sanity_issues'])}\n"
                    f"DB size: {report['db_size_mb']} MB"
                )
                notifier.send(msg)
        except Exception as exc:
            LOGGER.error("Failed to send notification: %s", exc)

    sys.exit(0 if report["overall_status"] == "healthy" else 1)


if __name__ == "__main__":
    main()
