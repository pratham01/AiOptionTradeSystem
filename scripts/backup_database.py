#!/usr/bin/env python3
"""Automated SQLite database backup with rotation and integrity validation.

Features:
- Online backup using sqlite3.backup() (zero-downtime, no locks)
- Rotated backups: keeps last 7 daily + last 4 weekly
- Post-backup integrity validation (PRAGMA integrity_check)
- Optional Telegram notification on failure
- Can be run via cron, Docker cron-tasks, or manually

Usage:
    python scripts/backup_database.py
    python scripts/backup_database.py --db-path data/trade_system.db --backup-dir backups/
    python scripts/backup_database.py --notify-on-failure
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("backup")


def backup_database(
    db_path: Path,
    backup_dir: Path,
    pages_per_step: int = 500,
    sleep_between_steps: float = 0.05,
) -> Path:
    """Perform an online SQLite backup using the C API backup interface.

    This copies the database page-by-page without holding a write lock,
    allowing the live bot to continue writing during the backup.

    Args:
        db_path: Path to the source database.
        backup_dir: Directory to store the backup file.
        pages_per_step: Number of pages to copy per iteration (controls lock granularity).
        sleep_between_steps: Seconds to sleep between page-copy iterations.

    Returns:
        Path to the completed backup file.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weekday = datetime.now().strftime("%A").lower()
    backup_name = f"trade_system_{timestamp}_{weekday}.db"
    backup_path = backup_dir / backup_name

    LOGGER.info("Starting backup: %s -> %s", db_path, backup_path)
    start = time.monotonic()

    # Open source in read-only mode
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    dest = sqlite3.connect(str(backup_path))

    try:
        with dest:
            backup = dest.backup(source, pages=pages_per_step, sleep=sleep_between_steps)
    except AttributeError:
        # Python < 3.7 fallback (unlikely but safe)
        LOGGER.warning("sqlite3.backup() not available. Using file copy fallback.")
        source.close()
        dest.close()
        shutil.copy2(str(db_path), str(backup_path))
    else:
        source.close()
        dest.close()

    elapsed = time.monotonic() - start
    size_mb = backup_path.stat().st_size / (1024 * 1024)
    LOGGER.info(
        "Backup complete: %.1f MB in %.1fs -> %s",
        size_mb,
        elapsed,
        backup_path.name,
    )
    return backup_path


def validate_backup(backup_path: Path) -> bool:
    """Run PRAGMA integrity_check on the backup to verify it's not corrupted."""
    LOGGER.info("Validating backup integrity: %s", backup_path.name)
    try:
        conn = sqlite3.connect(str(backup_path))
        cursor = conn.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()

        if result and result[0] == "ok":
            LOGGER.info("Integrity check PASSED.")
            return True
        else:
            LOGGER.error("Integrity check FAILED: %s", result)
            return False
    except Exception as exc:
        LOGGER.error("Integrity check error: %s", exc)
        return False


def rotate_backups(
    backup_dir: Path,
    keep_daily: int = 7,
    keep_weekly: int = 4,
) -> int:
    """Remove old backups, keeping the most recent daily and weekly copies.

    Args:
        backup_dir: Directory containing backup files.
        keep_daily: Number of most recent daily backups to keep.
        keep_weekly: Number of most recent weekly (Sunday) backups to keep.

    Returns:
        Number of backup files removed.
    """
    all_backups = sorted(
        backup_dir.glob("trade_system_*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not all_backups:
        return 0

    # Separate Sunday backups (weekly) from daily backups
    weekly = [b for b in all_backups if "sunday" in b.name]
    daily = [b for b in all_backups if "sunday" not in b.name]

    # Determine which to keep
    keep_set = set()
    keep_set.update(daily[:keep_daily])
    keep_set.update(weekly[:keep_weekly])

    # Always keep the most recent backup regardless
    keep_set.add(all_backups[0])

    removed = 0
    for backup_file in all_backups:
        if backup_file not in keep_set:
            LOGGER.info("Removing old backup: %s", backup_file.name)
            backup_file.unlink()
            removed += 1

    LOGGER.info(
        "Rotation complete: %d kept, %d removed (policy: %d daily + %d weekly).",
        len(keep_set),
        removed,
        keep_daily,
        keep_weekly,
    )
    return removed


def send_failure_notification(error_msg: str) -> None:
    """Send a Telegram notification about backup failure."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from trade_system.shared.config import Settings
        from trade_system.shared.notifications.telegram import TelegramNotifier

        settings = Settings.load()
        if settings.telegram.enabled:
            notifier = TelegramNotifier(
                settings.telegram.bot_token, settings.telegram.chat_id
            )
            notifier.send(
                f"🔴 <b>Database Backup FAILED</b>\n\n"
                f"<code>{error_msg}</code>\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
    except Exception as exc:
        LOGGER.error("Failed to send failure notification: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite database backup with rotation")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/trade_system.db"),
        help="Path to the SQLite database (default: data/trade_system.db)",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("data/backups"),
        help="Directory for backup files (default: data/backups/)",
    )
    parser.add_argument(
        "--keep-daily",
        type=int,
        default=7,
        help="Number of daily backups to retain (default: 7)",
    )
    parser.add_argument(
        "--keep-weekly",
        type=int,
        default=4,
        help="Number of weekly (Sunday) backups to retain (default: 4)",
    )
    parser.add_argument(
        "--notify-on-failure",
        action="store_true",
        help="Send Telegram notification on backup failure",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        LOGGER.error("Database file not found: %s", args.db_path)
        sys.exit(1)

    try:
        # 1. Perform backup
        backup_path = backup_database(args.db_path, args.backup_dir)

        # 2. Validate integrity
        if not validate_backup(backup_path):
            error_msg = f"Backup integrity check failed for {backup_path.name}"
            LOGGER.error(error_msg)
            backup_path.unlink()  # Remove corrupted backup
            if args.notify_on_failure:
                send_failure_notification(error_msg)
            sys.exit(1)

        # 3. Rotate old backups
        rotate_backups(
            args.backup_dir,
            keep_daily=args.keep_daily,
            keep_weekly=args.keep_weekly,
        )

        LOGGER.info("✅ Backup pipeline complete.")

    except Exception as exc:
        error_msg = f"Backup failed: {exc}"
        LOGGER.exception(error_msg)
        if args.notify_on_failure:
            send_failure_notification(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()

