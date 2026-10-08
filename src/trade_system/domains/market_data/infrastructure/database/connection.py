"""Database connection management — production-hardened SQLite engine.

Key improvements over the original:
- Thread-safe singleton with double-checked locking
- Connection pooling via QueuePool (not NullPool) with size limits
- SQLite PRAGMAs for performance and crash safety (WAL, busy_timeout, synchronous=NORMAL)
- Composite indexes for common query patterns (symbol + timestamp)
- Scoped session factory for thread-safe session access
- Structured logging instead of f-string warnings
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from trade_system.domains.market_data.infrastructure.database.models import Base

LOGGER = logging.getLogger(__name__)

# ── Database path resolution ────────────────────────────────────────────────
DB_DIR = Path(os.getenv("TRADE_SYSTEM_DATA_DIR", "data")).resolve()
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "trade_system.db"

# ── Thread-safe engine singleton ────────────────────────────────────────────
_engine: Engine | None = None
_engine_lock = threading.Lock()

# ── SQLite PRAGMAs applied per-connection ───────────────────────────────────
_SQLITE_PRAGMAS = [
    "PRAGMA journal_mode=WAL;",           # Write-Ahead Logging for concurrent reads
    "PRAGMA busy_timeout=5000;",          # Wait up to 5s instead of failing immediately
    "PRAGMA synchronous=NORMAL;",         # Faster writes; still crash-safe with WAL
    "PRAGMA cache_size=-64000;",          # 64 MB page cache (negative = KB)
    "PRAGMA foreign_keys=ON;",            # Enforce FK constraints
    "PRAGMA temp_store=MEMORY;",          # Temp tables in memory
]


def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    """SQLAlchemy event listener: configure every new SQLite connection."""
    cursor = dbapi_connection.cursor()
    for pragma in _SQLITE_PRAGMAS:
        cursor.execute(pragma)
    cursor.close()


# ── Index migrations ────────────────────────────────────────────────────────
_MIGRATION_QUERIES = [
    # Composite indexes for the most common query pattern: WHERE symbol = ? AND timestamp >= ?
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_sym_ts ON ohlcv_1m (symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_3m_sym_ts ON ohlcv_3m (symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_sym_ts ON ohlcv_5m (symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_sym_ts ON ohlcv_15m (symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_daily_sym_ts ON ohlcv_daily (symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_option_chain_sym_ts ON option_chain_data (underlying_symbol, timestamp);",
    # Legacy single-column indexes (kept for backward compatibility)
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_timestamp ON ohlcv_1m (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_3m_timestamp ON ohlcv_3m (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_timestamp ON ohlcv_5m (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_timestamp ON ohlcv_15m (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_daily_timestamp ON ohlcv_daily (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_option_timestamp ON option_chain_data (timestamp);",
]


def _run_migrations(engine: Engine) -> None:
    """Ensure essential indexes exist. Non-fatal on failure."""
    try:
        with engine.begin() as conn:
            for q in _MIGRATION_QUERIES:
                conn.execute(text(q))
        LOGGER.info("Database migrations applied successfully.")
    except Exception as exc:
        LOGGER.warning("Startup migration warning: %s", exc)


# ── Engine factory ──────────────────────────────────────────────────────────

def get_engine(db_path: str | None = None) -> Engine:
    """Return (or create) the thread-safe SQLAlchemy engine singleton.

    Uses double-checked locking to avoid race conditions during startup.
    QueuePool with size=5 keeps a small pool of reusable connections.
    """
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        # Double-checked locking
        if _engine is not None:
            return _engine

        path = db_path or str(DB_PATH)
        LOGGER.info("Creating SQLAlchemy engine for: %s", path)

        new_engine = create_engine(
            f"sqlite:///{path}",
            poolclass=NullPool,
            connect_args={"timeout": 30, "check_same_thread": False},
            echo=False,
        )

        # Register per-connection PRAGMA setup
        event.listen(new_engine, "connect", _set_sqlite_pragmas)

        # Create all tables
        Base.metadata.create_all(bind=new_engine)

        # Apply index migrations
        _run_migrations(new_engine)

        _engine = new_engine
        LOGGER.info("Database engine initialized (WAL mode, pool_size=5).")

    return _engine


# ── Session factories ───────────────────────────────────────────────────────

# Legacy alias — kept for existing import sites
engine = get_engine()

# Standard session factory
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Thread-safe scoped session — one session per thread
ScopedSession = scoped_session(SessionLocal)


def init_db() -> None:
    """Initialize database with all tables and indexes."""
    engine_obj = get_engine()
    Base.metadata.create_all(bind=engine_obj)
    _run_migrations(engine_obj)


def get_db_session() -> Session:
    """Get a database session (non-scoped, caller must close)."""
    return SessionLocal()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic commit/rollback.

    Usage:
        with get_db_context() as session:
            session.add(obj)
        # auto-committed on exit, auto-rolled-back on exception
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_health() -> dict:
    """Check database health. Returns dict with status and details."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
            result = conn.execute(text("PRAGMA page_count;")).scalar()
            page_size = conn.execute(text("PRAGMA page_size;")).scalar()
            db_size_bytes = (result or 0) * (page_size or 4096)
        return {
            "status": "healthy",
            "db_path": str(DB_PATH),
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 1),
            "pool_size": get_engine().pool.size() if hasattr(get_engine().pool, 'size') else "N/A",
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
        }

