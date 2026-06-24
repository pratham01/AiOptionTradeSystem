"""
Application settings with multi-broker support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time as dt_time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    """Split comma-separated values into list."""
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse boolean from string."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_path(value: str | None, default: str) -> Path:
    """Parse path from string."""
    return Path(value) if value else Path(default)


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    """Technical indicator configuration."""

    supertrend_period: int = 7
    supertrend_multiplier: int = 3
    trend_timeframe_minutes: int = 3
    confirmed_timeframe_minutes: int = 15
    confirmed_entry_cutoff: dt_time = field(default_factory=lambda: dt_time(14, 45))

    # Alert Thresholds
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_surge_multiplier: float = 2.0
    retracement_buffer_pct: float = 0.2
    alert_debounce_seconds: int = 3600 # 1 hour

    @classmethod
    def from_env(cls) -> "IndicatorConfig":
        """Load from environment variables."""

        def _parse_time(value: str | None, default: str) -> dt_time:
            if value:
                try:
                    return dt_time.fromisoformat(value)
                except ValueError:
                    pass
            return dt_time.fromisoformat(default)

        return cls(
            supertrend_period=int(os.getenv("SUPER_TREND_PERIOD", "7")),
            supertrend_multiplier=int(os.getenv("SUPER_TREND_MULTIPLIER", "3")),
            trend_timeframe_minutes=int(os.getenv("SUPER_TREND_TIMEFRAME_MINUTES", "3")),
            confirmed_timeframe_minutes=int(os.getenv("CONFIRMED_TIMEFRAME_MINUTES", "15")),
            confirmed_entry_cutoff=_parse_time(os.getenv("CONFIRMED_ENTRY_CUTOFF"), "14:45"),
            rsi_overbought=float(os.getenv("ALERT_RSI_OVERBOUGHT", "70.0")),
            rsi_oversold=float(os.getenv("ALERT_RSI_OVERSOLD", "30.0")),
            volume_surge_multiplier=float(os.getenv("ALERT_VOLUME_SURGE_MULTIPLIER", "2.0")),
            retracement_buffer_pct=float(os.getenv("ALERT_RETRACEMENT_BUFFER_PCT", "0.2")),
            alert_debounce_seconds=int(os.getenv("ALERT_DEBOUNCE_SECONDS", "3600")),
        )


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    """Generic broker configuration."""

    client_id: str = ""
    access_token: str = ""
    enabled: bool = False
    priority: int = 1  # Lower = higher priority for primary broker


@dataclass(frozen=True, slots=True)
class FyersConfig(BrokerConfig):
    """Fyers-specific configuration."""

    user_id: str = ""
    secret_key: str = ""
    redirect_uri: str = "https://trade.fyers.in/api-login/redirect-uri/index.html"
    pin: str = ""
    totp_secret: str = ""
    token_path: Path = field(default_factory=lambda: Path(".secrets/fyers_token.json"))


@dataclass(frozen=True, slots=True)
class DhanConfig(BrokerConfig):
    """Dhan-specific configuration."""

    api_key: str = ""
    token_path: Path = field(default_factory=lambda: Path(".secrets/dhan_token.json"))


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Telegram notification configuration."""

    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False


@dataclass(frozen=True, slots=True)
class WhatsappConfig:
    """WhatsApp notification configuration (Meta Cloud API)."""

    phone_number_id: str = ""
    access_token: str = ""
    to_number: str = ""
    enabled: bool = False



@dataclass(frozen=True, slots=True)
class LlmConfig:
    """LLM configuration for the agentic brain."""

    provider: str = "openai"  # "openai", "gemini", "anthropic"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str | None = None
    timeout: int = 120


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings with multi-broker support."""

    # Primary data broker (for real-time quotes)
    primary_broker: str = "fyers"  # "fyers" or "dhan"

    # Backup broker (for failover)
    backup_broker: str = "dhan"

    # Broker configurations
    fyers: FyersConfig = field(default_factory=FyersConfig)
    dhan: DhanConfig = field(default_factory=DhanConfig)

    # LLM Configuration
    llm: LlmConfig = field(default_factory=LlmConfig)

    # Notification settings
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    whatsapp: WhatsappConfig = field(default_factory=WhatsappConfig)
    st_confirmed_telegram: TelegramConfig = field(default_factory=TelegramConfig)
    top_gainer_telegram: TelegramConfig = field(default_factory=TelegramConfig)
    top_sectors_telegram: TelegramConfig = field(default_factory=TelegramConfig)

    # Data directories
    data_dir: Path = field(default_factory=lambda: Path("data"))
    secrets_dir: Path = field(default_factory=lambda: Path(".secrets"))
    option_chain_dir: Path = field(default_factory=lambda: Path("data/option_chain_data"))
    market_data_dir: Path = field(default_factory=lambda: Path("data/market_data"))

    # Trading settings
    live_symbols: list[str] = field(default_factory=lambda: ["NSE:NIFTY50-INDEX"])
    index_symbols: list[str] = field(default_factory=lambda: ["NSE:NIFTY50-INDEX"])
    live_timeframe_minutes: int = 1
    option_chain_interval_seconds: int = 180
    option_chain_adjacent_strikes: int = 7

    # Market hours (IST)
    market_premarket_check: str = "09:09"
    market_start: str = "09:15"
    market_end: str = "15:30"

    # Analysis settings
    premarket_daily_lookback_days: int = 3
    premarket_intraday_history_days: int = 3
    major_gap_threshold_pct: float = 0.5

    # Feature flags
    enable_experimental_ict_stream: bool = False

    # Indicator settings
    indicator_config: IndicatorConfig = field(default_factory=IndicatorConfig)

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @property
    def fyers_client_id(self) -> str:
        return self.fyers.client_id

    @property
    def fyers_access_token(self) -> str:
        return self.fyers.access_token

    @property
    def fyers_user_id(self) -> str:
        return self.fyers.user_id

    @property
    def telegram_bot_token(self) -> str:
        return self.telegram.bot_token

    @property
    def telegram_chat_id(self) -> str:
        return self.telegram.chat_id

    @property
    def st_confirmed_telegram_bot_token(self) -> str:
        return self.st_confirmed_telegram.bot_token

    @property
    def st_confirmed_telegram_chat_id(self) -> str:
        return self.st_confirmed_telegram.chat_id

    @classmethod
    def load(cls, env_file: str = ".env") -> "Settings":
        """Load settings from environment file."""
        load_dotenv(env_file, override=True)

        # Parse broker settings
        fyers_config = FyersConfig(
            client_id=os.getenv("FYERS_CLIENT_ID", ""),
            user_id=os.getenv("FYERS_USER_ID", ""),
            secret_key=os.getenv("FYERS_SECRET_KEY", ""),
            redirect_uri=os.getenv(
                "FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html"
            ),
            pin=os.getenv("FYERS_PIN", ""),
            totp_secret=os.getenv("FYERS_TOTP_SECRET", ""),
            access_token=os.getenv("FYERS_ACCESS_TOKEN", ""),
            token_path=_parse_path(os.getenv("FYERS_TOKEN_PATH"), ".secrets/fyers_token.json"),
            enabled=_parse_bool(os.getenv("FYERS_ENABLED"), True),
            priority=int(os.getenv("FYERS_PRIORITY", "1")),
        )

        dhan_config = DhanConfig(
            client_id=os.getenv("DHAN_CLIENT_ID", ""),
            api_key=os.getenv("DHAN_API_KEY", ""),
            access_token=os.getenv("DHAN_ACCESS_TOKEN", ""),
            token_path=_parse_path(os.getenv("DHAN_TOKEN_PATH"), ".secrets/dhan_token.json"),
            enabled=_parse_bool(os.getenv("DHAN_ENABLED"), False),
            priority=int(os.getenv("DHAN_PRIORITY", "2")),
        )

        # Parse telegram settings
        telegram_config = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")),
        )

        st_telegram_config = TelegramConfig(
            bot_token=os.getenv("ST_CONFIRMED_TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("ST_CONFIRMED_TELEGRAM_CHAT_ID", ""),
            enabled=bool(
                os.getenv("ST_CONFIRMED_TELEGRAM_BOT_TOKEN")
                and os.getenv("ST_CONFIRMED_TELEGRAM_CHAT_ID")
            ),
        )

        top_gainer_telegram_config = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_TOP_GAINER_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_TOP_GAINER_CHAT_ID", ""),
            enabled=bool(
                os.getenv("TELEGRAM_TOP_GAINER_TOKEN")
                and os.getenv("TELEGRAM_TOP_GAINER_CHAT_ID")
            ),
        )

        top_sectors_telegram_config = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_TOP_SECTOR_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_TOP_SECTORS_CHAT_ID", ""),
            enabled=bool(
                os.getenv("TELEGRAM_TOP_SECTOR_TOKEN")
                and os.getenv("TELEGRAM_TOP_SECTORS_CHAT_ID")
            ),
        )

        whatsapp_config = WhatsappConfig(
            phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID", ""),
            access_token=os.getenv("WHATSAPP_ACCESS_TOKEN", ""),
            to_number=os.getenv("WHATSAPP_TO_NUMBER", ""),
            enabled=_parse_bool(os.getenv("WHATSAPP_ENABLED"), False),
        )

        # Parse LLM settings
        # Preference: Gemini 1.5 Pro (The requested "Brain"), but respect LLM_PROVIDER if explicitly set
        llm_provider = os.getenv("LLM_PROVIDER")
        if not llm_provider:
             llm_provider = "gemini" if os.getenv("GOOGLE_API_KEY") else "openai"
             
        llm_model = os.getenv("LLM_MODEL")
        if not llm_model:
             llm_model = "gemini-1.5-pro" if llm_provider == "gemini" else "gpt-4o-mini"
             
        llm_config = LlmConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL"),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        )

        return cls(
            primary_broker=os.getenv("PRIMARY_BROKER", "fyers"),
            backup_broker=os.getenv("BACKUP_BROKER", "dhan"),
            fyers=fyers_config,
            dhan=dhan_config,
            llm=llm_config,
            telegram=telegram_config,
            whatsapp=whatsapp_config,
            st_confirmed_telegram=st_telegram_config,
            top_gainer_telegram=top_gainer_telegram_config,
            top_sectors_telegram=top_sectors_telegram_config,
            data_dir=_parse_path(os.getenv("TRADE_SYSTEM_DATA_DIR"), "data"),
            secrets_dir=_parse_path(os.getenv("TRADE_SYSTEM_SECRETS_DIR"), ".secrets"),
            option_chain_dir=_parse_path(os.getenv("OPTION_CHAIN_DATA_DIR"), "data/option_chain_data"),
            market_data_dir=_parse_path(os.getenv("MARKET_DATA_DIR"), "data/market_data"),
            live_symbols=_split_csv(os.getenv("LIVE_SYMBOLS"), ["NSE:NIFTY50-INDEX"]),
            index_symbols=_split_csv(os.getenv("INDEX_SYMBOLS"), ["NSE:NIFTY50-INDEX"]),
            live_timeframe_minutes=int(os.getenv("LIVE_TIMEFRAME_MINUTES", "1")),
            option_chain_interval_seconds=int(os.getenv("OPTION_CHAIN_INTERVAL_SECONDS", "180")),
            option_chain_adjacent_strikes=int(os.getenv("OPTION_CHAIN_ADJACENT_STRIKES", "7")),
            market_premarket_check=os.getenv("MARKET_PREMARKET_CHECK", "09:09"),
            market_start=os.getenv("MARKET_START", "09:15"),
            market_end=os.getenv("MARKET_END", "15:30"),
            premarket_daily_lookback_days=int(os.getenv("PREMARKET_DAILY_LOOKBACK_DAYS", "3")),
            premarket_intraday_history_days=int(os.getenv("PREMARKET_INTRADAY_HISTORY_DAYS", "3")),
            major_gap_threshold_pct=float(os.getenv("MAJOR_GAP_THRESHOLD_PCT", "0.5")),
            enable_experimental_ict_stream=_parse_bool(
                os.getenv("ENABLE_EXPERIMENTAL_ICT_STREAM"), False
            ),
            indicator_config=IndicatorConfig.from_env(),
            log_level=os.getenv("TRADE_SYSTEM_LOG_LEVEL", "INFO"),
        )

    def ensure_directories(self) -> None:
        """Create necessary data directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        self.option_chain_dir.mkdir(parents=True, exist_ok=True)
        self.market_data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "fo_historical").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "top_gainers").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "sector_performance").mkdir(parents=True, exist_ok=True)

        self.fyers.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.dhan.token_path.parent.mkdir(parents=True, exist_ok=True)

    def get_primary_broker_config(self) -> BrokerConfig:
        """Get configuration for primary broker."""
        if self.primary_broker.lower() == "fyers":
            return self.fyers
        elif self.primary_broker.lower() == "dhan":
            return self.dhan
        raise ValueError(f"Unknown primary broker: {self.primary_broker}")

    def get_backup_broker_config(self) -> BrokerConfig | None:
        """Get configuration for backup broker if enabled."""
        if not self.backup_broker:
            return None

        if self.backup_broker.lower() == "fyers" and self.fyers.enabled:
            return self.fyers
        elif self.backup_broker.lower() == "dhan" and self.dhan.enabled:
            return self.dhan
        return None
