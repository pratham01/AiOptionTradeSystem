"""API authentication middleware — API key-based auth with rate limiting.

Provides:
- API key validation (bcrypt-hashed keys stored in config)
- Rate limiting (sliding window, per-key)
- FastAPI middleware integration

Usage:
    from trade_system.interfaces.api.auth import require_api_key, RateLimiter

    # In FastAPI:
    @app.get("/signals", dependencies=[Depends(require_api_key)])
    async def get_signals(): ...
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import threading
from collections import defaultdict
from typing import Any

LOGGER = logging.getLogger(__name__)


class APIKeyValidator:
    """Validates API keys against a set of allowed hashed keys.

    Keys are stored as SHA-256 hashes for security. The plaintext
    key is never stored.
    """

    def __init__(self, allowed_keys: list[str] | None = None) -> None:
        """Initialize with allowed plaintext keys (will be hashed internally).

        Args:
            allowed_keys: List of plaintext API keys. If None, reads from
                API_KEYS env var (comma-separated).
        """
        if allowed_keys is None:
            raw = os.getenv("API_KEYS", "")
            allowed_keys = [k.strip() for k in raw.split(",") if k.strip()]

        self._hashed_keys: set[str] = set()
        for key in allowed_keys:
            self._hashed_keys.add(self._hash_key(key))

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key with SHA-256."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def validate(self, key: str) -> bool:
        """Check if a plaintext API key is valid."""
        if not self._hashed_keys:
            # No keys configured = auth disabled (development mode)
            return True
        return self._hash_key(key) in self._hashed_keys


class RateLimiter:
    """In-memory sliding window rate limiter.

    Thread-safe. Tracks request timestamps per key and rejects
    requests that exceed the configured rate.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if a request from this key is allowed."""
        now = time.monotonic()
        cutoff = now - self._window_seconds

        with self._lock:
            timestamps = self._requests[key]
            # Remove expired timestamps
            self._requests[key] = [t for t in timestamps if t > cutoff]
            timestamps = self._requests[key]

            if len(timestamps) >= self._max_requests:
                return False

            timestamps.append(now)
            return True

    def remaining(self, key: str) -> int:
        """Get remaining requests for this key in the current window."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            active = [t for t in self._requests.get(key, []) if t > cutoff]
            return max(0, self._max_requests - len(active))


# Global instances
_api_key_validator = APIKeyValidator()
_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


def validate_request(api_key: str | None) -> tuple[bool, str]:
    """Validate an API request (key + rate limit).

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not api_key:
        if not _api_key_validator._hashed_keys:
            return True, ""  # No auth configured
        return False, "Missing X-API-Key header"

    if not _api_key_validator.validate(api_key):
        return False, "Invalid API key"

    if not _rate_limiter.is_allowed(api_key):
        return False, "Rate limit exceeded (100 req/min)"

    return True, ""


# FastAPI dependency (optional — only used if FastAPI is installed)
def require_api_key():
    """FastAPI dependency for API key authentication."""
    try:
        from fastapi import Request, HTTPException, Depends

        async def _verify(request: Request):
            api_key = request.headers.get("X-API-Key")
            valid, error = validate_request(api_key)
            if not valid:
                raise HTTPException(status_code=401 if "Invalid" in error or "Missing" in error else 429, detail=error)

        return Depends(_verify)
    except ImportError:
        pass
