"""
Configuration package for Trade System.
Provides both new hierarchical settings and a backward-compatible flat interface.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from .settings import (
    Settings as NewSettings,
    IndicatorConfig,
    BrokerConfig,
    FyersConfig,
    DhanConfig,
    TelegramConfig,
    WhatsappConfig
)

@dataclasses.dataclass(frozen=True, slots=True)
class Settings(NewSettings):
    """
    Backward-compatible Settings class.
    
    This class wraps the new hierarchical settings and provides flat properties
    matching the old configuration format.
    """
    
    @classmethod
    def load(cls, env_file: str = ".env") -> "Settings":
        new_settings = NewSettings.load(env_file)
        # Create a Settings instance from the NewSettings instance
        return cls(**{f.name: getattr(new_settings, f.name) for f in dataclasses.fields(NewSettings)})

    @property
    def fyers_user_id(self) -> str:
        return self.fyers.user_id
        
    @property
    def fyers_client_id(self) -> str:
        return self.fyers.client_id
        
    @property
    def fyers_secret_key(self) -> str:
        return self.fyers.secret_key
        
    @property
    def fyers_redirect_uri(self) -> str:
        return self.fyers.redirect_uri
        
    @property
    def fyers_pin(self) -> str:
        return self.fyers.pin
        
    @property
    def fyers_totp_secret(self) -> str:
        return self.fyers.totp_secret
        
    @property
    def fyers_access_token(self) -> str:
        return self.fyers.access_token
        
    @property
    def fyers_token_path(self) -> Path:
        return self.fyers.token_path
        
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
        
    @property
    def option_chain_data_dir(self) -> Path:
        return self.option_chain_dir

__all__ = [
    "Settings",
    "IndicatorConfig",
    "BrokerConfig",
    "FyersConfig",
    "DhanConfig",
    "TelegramConfig",
    "WhatsappConfig"
]
