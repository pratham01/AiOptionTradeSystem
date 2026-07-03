"""Database connection management."""

import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from trade_system.infrastructure.database.models import Base

# Database path
DB_DIR = Path(os.getenv("TRADE_SYSTEM_DATA_DIR", "data")).resolve()
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "trade_system.db"

# Create engine (singleton)
_engine: Engine | None = None


def run_migrations(engine: Engine) -> None:
    """Ensure essential indexes are created on existing SQLite databases."""
    from sqlalchemy import text
    queries = [
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_timestamp ON ohlcv_1m (timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_3m_timestamp ON ohlcv_3m (timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_timestamp ON ohlcv_5m (timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_timestamp ON ohlcv_15m (timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_daily_timestamp ON ohlcv_daily (timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_option_timestamp ON option_chain_data (timestamp);"
    ]
    try:
        with engine.begin() as conn:
            for q in queries:
                conn.execute(text(q))
    except Exception as exc:
        # Don't fail the startup if migration fails (e.g. database locked)
        import logging
        logging.getLogger(__name__).warning(f"Startup migration warning: {exc}")


def get_engine(db_path: str | None = None) -> Engine:
    """Return (or create) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        path = db_path or str(DB_PATH)
        _engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"timeout": 30},
            echo=False
        )
        # Enable Write-Ahead Logging (WAL) mode for concurrency support
        from sqlalchemy import text
        try:
            with _engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to enable WAL mode: {e}")
            
        Base.metadata.create_all(bind=_engine)
        run_migrations(_engine)
    return _engine


# Legacy alias kept for existing code
engine = get_engine()

# Create session factory
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Initialize database with all tables and indexes."""
    engine_obj = get_engine()
    Base.metadata.create_all(bind=engine_obj)
    run_migrations(engine_obj)


def get_db_session() -> Session:
    """Get a database session."""
    return SessionLocal()


@contextmanager
def get_db_context():
    """Context manager for database sessions."""
    session = get_db_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

