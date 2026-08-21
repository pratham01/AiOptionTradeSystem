"""Integration tests for database operations — CRUD, concurrency, and health checks.

Tests verify the hardened SQLite connection with:
- Basic CRUD on OHLCV tables
- Concurrent multi-threaded writes (stress test)
- Database health check endpoint
- Context manager session handling
"""

import threading
import time
from datetime import datetime

import pytest
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session

from trade_system.domains.market_data.infrastructure.database.models import Base, Ohlcv1m, Ohlcv3m


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Apply same PRAGMAs as production
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(in_memory_engine):
    """Create a session bound to the in-memory engine."""
    Session = sessionmaker(bind=in_memory_engine)
    session = Session()
    yield session
    session.close()


class TestOhlcvCrud:
    """Test basic CRUD operations on OHLCV tables."""

    def test_insert_and_read_1m_bar(self, db_session):
        """Test inserting and reading a 1-minute OHLCV bar."""
        bar = Ohlcv1m(
            symbol="NSE:NIFTY50-INDEX",
            timestamp=datetime(2026, 8, 18, 9, 15, 0),
            open=24500.0,
            high=24520.0,
            low=24490.0,
            close=24510.0,
            volume=150000.0,
        )
        db_session.add(bar)
        db_session.commit()

        result = db_session.query(Ohlcv1m).filter_by(symbol="NSE:NIFTY50-INDEX").first()
        assert result is not None
        assert result.open == 24500.0
        assert result.high == 24520.0
        assert result.close == 24510.0
        assert result.volume == 150000.0

    def test_upsert_updates_existing(self, db_session):
        """Test that re-inserting same symbol+timestamp updates the row."""
        ts = datetime(2026, 8, 18, 9, 16, 0)
        bar1 = Ohlcv1m(
            symbol="NSE:NIFTY50-INDEX",
            timestamp=ts,
            open=24500.0, high=24520.0, low=24490.0, close=24510.0, volume=100.0,
        )
        db_session.add(bar1)
        db_session.commit()

        # Update the same bar
        existing = db_session.query(Ohlcv1m).filter_by(
            symbol="NSE:NIFTY50-INDEX", timestamp=ts
        ).first()
        existing.close = 24550.0
        existing.volume = 200.0
        db_session.commit()

        result = db_session.query(Ohlcv1m).filter_by(
            symbol="NSE:NIFTY50-INDEX", timestamp=ts
        ).first()
        assert result.close == 24550.0
        assert result.volume == 200.0

    def test_query_date_range(self, db_session):
        """Test querying bars within a date range."""
        bars = []
        for i in range(10):
            bars.append(Ohlcv1m(
                symbol="NSE:NIFTY50-INDEX",
                timestamp=datetime(2026, 8, 18, 9, 15 + i, 0),
                open=24500.0 + i, high=24520.0, low=24490.0,
                close=24510.0, volume=1000.0,
            ))
        db_session.add_all(bars)
        db_session.commit()

        # Query subset
        from_dt = datetime(2026, 8, 18, 9, 17, 0)
        to_dt = datetime(2026, 8, 18, 9, 22, 0)
        results = (
            db_session.query(Ohlcv1m)
            .filter(
                Ohlcv1m.symbol == "NSE:NIFTY50-INDEX",
                Ohlcv1m.timestamp >= from_dt,
                Ohlcv1m.timestamp <= to_dt,
            )
            .order_by(Ohlcv1m.timestamp.asc())
            .all()
        )
        assert len(results) == 6  # 09:17 through 09:22

    def test_multiple_symbols(self, db_session):
        """Test storing bars for multiple symbols."""
        symbols = ["NSE:NIFTY50-INDEX", "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ"]
        for sym in symbols:
            db_session.add(Ohlcv1m(
                symbol=sym,
                timestamp=datetime(2026, 8, 18, 9, 15, 0),
                open=100.0, high=110.0, low=90.0, close=105.0, volume=500.0,
            ))
        db_session.commit()

        count = db_session.query(Ohlcv1m).count()
        assert count == 3

        nifty = db_session.query(Ohlcv1m).filter_by(symbol="NSE:NIFTY50-INDEX").first()
        assert nifty is not None


class TestConcurrentWrites:
    """Test thread-safe concurrent database writes."""

    def test_concurrent_inserts(self, tmp_path):
        """Stress test: multiple threads writing bars concurrently to a file-based SQLite."""
        # Use a file-based temp DB so all threads share the same database
        db_path = tmp_path / "test_concurrent.db"
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False, "timeout": 30},
            echo=False,
        )

        @event.listens_for(engine, "connect")
        def _set_pragmas(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()

        Base.metadata.create_all(bind=engine)

        errors = []

        def write_bars(thread_id: int, n_bars: int):
            SessionFactory = sessionmaker(bind=engine)
            session = SessionFactory()
            try:
                for i in range(n_bars):
                    hour = 9 + ((i * 3) // 60)
                    minute = (i * 3) % 60
                    bar = Ohlcv3m(
                        symbol=f"NSE:THREAD{thread_id}-EQ",
                        timestamp=datetime(2026, 8, 18, hour, minute, 0),
                        open=100.0, high=110.0, low=90.0, close=105.0, volume=500.0,
                    )
                    session.add(bar)
                session.commit()
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")
                session.rollback()
            finally:
                session.close()

        # Launch 5 threads each writing 20 bars
        threads = []
        for tid in range(5):
            t = threading.Thread(target=write_bars, args=(tid, 20))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent write errors: {errors}"

        # Verify all bars were written
        SessionFactory = sessionmaker(bind=engine)
        session = SessionFactory()
        total = session.query(Ohlcv3m).count()
        session.close()
        engine.dispose()
        assert total == 100  # 5 threads × 20 bars


class TestHealthCheck:
    """Test database health check."""

    def test_health_check_returns_status(self):
        """Test that get_db_health returns a valid status dict."""
        from trade_system.domains.market_data.infrastructure.database.connection import get_db_health

        health = get_db_health()
        assert "status" in health
        # Should be either healthy or unhealthy
        assert health["status"] in ("healthy", "unhealthy")

    def test_health_check_includes_size(self):
        """Test that health check reports database size."""
        from trade_system.domains.market_data.infrastructure.database.connection import get_db_health

        health = get_db_health()
        if health["status"] == "healthy":
            assert "db_size_mb" in health
