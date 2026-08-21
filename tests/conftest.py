"""Shared test fixtures for the trade_system test suite."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from trade_system.domains.market_data.infrastructure.database.models import Base


# ── Database Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with production-equivalent PRAGMAs."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )

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
    """Create a database session bound to the in-memory engine."""
    SessionFactory = sessionmaker(bind=in_memory_engine)
    session = SessionFactory()
    yield session
    session.close()


# ── Mock Broker Fixture ─────────────────────────────────────────────────────

@pytest.fixture
def mock_broker():
    """Create a mock broker client for testing without real API calls."""
    broker = MagicMock()
    broker.get_quotes.return_value = {
        "NSE:NIFTY50-INDEX": {"ltp": 24500.0, "volume": 1000000}
    }
    broker.websocket_access_token.return_value = "mock-ws-token"
    broker.get_positions.return_value = []
    broker.get_orders.return_value = []
    broker.is_authenticated.return_value = True
    return broker


# ── Mock Telegram Fixture ───────────────────────────────────────────────────

@pytest.fixture
def mock_telegram(monkeypatch):
    """Mock Telegram API calls to prevent real messages during tests."""
    mock_post = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"ok": True}
    mock_post.return_value = mock_response

    monkeypatch.setattr("trade_system.shared.notifications.telegram.requests.post", mock_post)
    return mock_post


# ── Settings Fixture ────────────────────────────────────────────────────────

@pytest.fixture
def test_settings():
    """Create a test Settings object with safe defaults."""
    from trade_system.shared.config.settings import Settings, FyersConfig, TelegramConfig, LlmConfig
    return Settings(
        fyers=FyersConfig(client_id="test", access_token="test-token", enabled=True),
        telegram=TelegramConfig(bot_token="test-bot-token", chat_id="test-chat-id", enabled=False),
        llm=LlmConfig(provider="openai", model="gpt-4o-mini", api_key="test-key"),
        log_level="DEBUG",
    )
