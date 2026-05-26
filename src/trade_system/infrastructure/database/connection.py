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


def get_engine(db_path: str | None = None) -> Engine:
    """Return (or create) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        path = db_path or str(DB_PATH)
        _engine = create_engine(f"sqlite:///{path}", echo=False)
        Base.metadata.create_all(bind=_engine)
    return _engine


# Legacy alias kept for existing code
engine = get_engine()

# Create session factory
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Initialize database with all tables."""
    Base.metadata.create_all(bind=get_engine())


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

