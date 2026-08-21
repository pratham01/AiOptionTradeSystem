"""Secrets provider abstraction for secure credential management.

Supports multiple backends:
- EnvSecretsProvider: reads from .env file (local development)
- EncryptedFileSecretsProvider: reads from AES-encrypted file (production-local)
- CloudSecretsProvider: stub for AWS Secrets Manager / GCP Secret Manager

Usage:
    provider = get_secrets_provider()  # auto-selects based on SECRETS_PROVIDER env var
    api_key = provider.get_secret("FYERS_SECRET_KEY")
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SecretsProvider(Protocol):
    """Protocol for secrets providers."""

    def get_secret(self, key: str, default: str = "") -> str:
        """Retrieve a secret value by key."""
        ...

    def has_secret(self, key: str) -> bool:
        """Check if a secret exists."""
        ...

    def list_keys(self) -> list[str]:
        """List all available secret keys."""
        ...


class EnvSecretsProvider:
    """Reads secrets from environment variables / .env file.

    This is the default provider for local development.
    """

    def __init__(self, env_file: str = ".env") -> None:
        self._env_file = env_file
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
        except ImportError:
            pass

    def get_secret(self, key: str, default: str = "") -> str:
        return os.getenv(key, default)

    def has_secret(self, key: str) -> bool:
        return key in os.environ

    def list_keys(self) -> list[str]:
        # Return keys that look like secrets (common prefixes)
        secret_prefixes = (
            "FYERS_", "DHAN_", "TELEGRAM_", "ST_CONFIRMED_",
            "LLM_", "GOOGLE_", "OPENAI_", "WHATSAPP_",
        )
        return [k for k in os.environ if k.startswith(secret_prefixes)]


class EncryptedFileSecretsProvider:
    """Reads secrets from an AES-encrypted JSON file.

    The file is decrypted at runtime using a master key from the
    MASTER_KEY environment variable.

    File format (encrypted): AES-CBC encrypted JSON blob, base64-encoded.
    """

    def __init__(self, encrypted_path: str = ".secrets.enc", master_key: str | None = None) -> None:
        self._path = Path(encrypted_path)
        self._master_key = master_key or os.getenv("MASTER_KEY", "")
        self._secrets: dict[str, str] = {}

        if self._path.exists() and self._master_key:
            self._load()
        elif self._path.exists() and not self._master_key:
            LOGGER.warning(
                "Encrypted secrets file found at %s but MASTER_KEY not set. "
                "Falling back to environment variables.",
                self._path,
            )

    def _derive_key(self, master_key: str) -> bytes:
        """Derive a 32-byte AES key from the master password using SHA-256."""
        return hashlib.sha256(master_key.encode("utf-8")).digest()

    def _load(self) -> None:
        """Decrypt and load the secrets file."""
        try:
            from cryptography.fernet import Fernet

            # Derive Fernet key from master key (Fernet needs base64-encoded 32 bytes)
            key_bytes = self._derive_key(self._master_key)
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)

            encrypted_data = self._path.read_bytes()
            decrypted = fernet.decrypt(encrypted_data)
            self._secrets = json.loads(decrypted.decode("utf-8"))
            LOGGER.info("Loaded %d secrets from encrypted file.", len(self._secrets))

        except ImportError:
            LOGGER.error(
                "cryptography package not installed. "
                "Install with: pip install cryptography"
            )
        except Exception as exc:
            LOGGER.error("Failed to decrypt secrets file: %s", exc)

    def get_secret(self, key: str, default: str = "") -> str:
        return self._secrets.get(key, os.getenv(key, default))

    def has_secret(self, key: str) -> bool:
        return key in self._secrets or key in os.environ

    def list_keys(self) -> list[str]:
        return list(self._secrets.keys())

    @classmethod
    def encrypt_env_file(cls, env_file: str, output_path: str, master_key: str) -> None:
        """Encrypt a .env file into the encrypted secrets format.

        Args:
            env_file: Path to the plaintext .env file.
            output_path: Path for the encrypted output file.
            master_key: Master password for encryption.
        """
        from cryptography.fernet import Fernet

        # Parse .env file
        secrets = {}
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if value:  # Only store non-empty values
                        secrets[key] = value

        # Encrypt
        key_bytes = hashlib.sha256(master_key.encode("utf-8")).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        fernet = Fernet(fernet_key)

        plaintext = json.dumps(secrets, indent=2).encode("utf-8")
        encrypted = fernet.encrypt(plaintext)

        Path(output_path).write_bytes(encrypted)
        LOGGER.info(
            "Encrypted %d secrets from %s -> %s",
            len(secrets), env_file, output_path,
        )


class CloudSecretsProvider:
    """Stub for cloud-based secrets managers (AWS, GCP, Azure).

    To be implemented when deploying to cloud. Currently falls back
    to environment variables.
    """

    def __init__(self, provider: str = "aws", region: str = "ap-south-1") -> None:
        self._provider = provider
        self._region = region
        LOGGER.info("CloudSecretsProvider initialized (provider=%s, region=%s)", provider, region)

    def get_secret(self, key: str, default: str = "") -> str:
        # TODO: Implement AWS Secrets Manager / GCP Secret Manager lookup
        return os.getenv(key, default)

    def has_secret(self, key: str) -> bool:
        return key in os.environ

    def list_keys(self) -> list[str]:
        return []


def get_secrets_provider() -> SecretsProvider:
    """Factory function to create the appropriate secrets provider.

    Selection is based on the SECRETS_PROVIDER environment variable:
    - "env" (default): EnvSecretsProvider
    - "encrypted": EncryptedFileSecretsProvider
    - "aws" or "gcp": CloudSecretsProvider
    """
    provider_type = os.getenv("SECRETS_PROVIDER", "env").lower()

    if provider_type == "encrypted":
        return EncryptedFileSecretsProvider()
    elif provider_type in ("aws", "gcp", "azure"):
        return CloudSecretsProvider(provider=provider_type)
    else:
        return EnvSecretsProvider()
