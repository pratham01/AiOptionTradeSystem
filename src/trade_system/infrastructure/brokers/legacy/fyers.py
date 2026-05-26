from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Semaphore, Lock

import pandas as pd
from fyers_apiv3 import fyersModel

from .base import BaseBroker
from .fyers_auth import FyersAuthenticator

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_nse_market_status(market_status: list[dict[str, Any]]) -> tuple[bool, str]:
    for market in market_status:
        exchange = _as_int(market.get("exchange"))
        segment = _as_int(market.get("segment"))
        status = str(market.get("status", "")).upper()
        if exchange == 10 and segment == 10:
            if status in {"OPEN", "PREOPEN", "PREOPEN_CLOSED"}:
                return True, status
            return False, status or "CLOSED"
    return False, "NSE market status not present"


class FyersBroker(BaseBroker):
    def __init__(
        self,
        client_id: str,
        access_token: str,
        user_id: str | None = None,
        log_path: str | None = None,
        authenticator: FyersAuthenticator | None = None,
    ):
        self.client_id = client_id
        self.access_token = access_token
        self.user_id = user_id
        self.log_path = Path(log_path or "logs/")
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.fyers = None
        self.client = None
        self.authenticator = authenticator
        self.settings = getattr(authenticator, "settings", None)
        # Fyers API Limit: ~10 requests per second.
        self._last_request_time = 0.0
        self._rate_limit_lock = Lock()

    def _apply_rate_limit(self):
        """Ensures at least 0.11s between requests across all threads."""
        with self._rate_limit_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < 0.11: 
                time.sleep(0.11 - elapsed)
            self._last_request_time = time.time()

    def _token_for_sdk(self) -> str:
        """
        Returns the JWT part of the token. Fyers REST SDK expects just the JWT 
        as it prepends the client_id internally.
        """
        token = self.access_token or ""
        if isinstance(token, str) and ":" in token:
            return token.split(":", 1)[1]
        return token

    def authenticate(self, retry_with_totp: bool = True) -> bool:
        logger.info("Authenticating with Fyers...")
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                is_async=False,
                token=self._token_for_sdk(),
                log_path=str(self.log_path),
            )
            self.client = self.fyers
            profile = self.fyers.get_profile()

            if profile.get("s") == "ok":
                logger.info("Fyers authentication successful.")
                return True

            if profile.get("code") == -8 and retry_with_totp and self.authenticator:
                logger.warning("Fyers token expired. Attempting automated refresh via TOTP...")
                try:
                    new_token = self.authenticator.generate_access_token()
                    self.access_token = new_token
                    self.authenticator.update_env_file(new_token)
                    return self.authenticate(retry_with_totp=False)
                except Exception as exc:
                    logger.error("Failed to automatically refresh Fyers token: %s", exc)
                    return False

            logger.error("Fyers authentication failed: %s (code: %s)", 
                         profile.get("errmsg") or profile.get("message"), 
                         profile.get("code"))
            return False
        except Exception as exc:
            logger.error("Error during Fyers authentication: %s", exc)
            return False

    def verify_session(self) -> None:
        if not self.fyers and not self.authenticate():
            raise RuntimeError("Unable to authenticate with FYERS.")
        profile = self.fyers.get_profile()
        if profile.get("s") != "ok":
            if profile.get("code") == -8 and self.authenticator:
                logger.warning("FYERS session expired during verification. Refreshing token.")
                new_token = self.authenticator.generate_access_token()
                self.access_token = new_token
                self.authenticator.update_env_file(new_token)
                if not self.authenticate(retry_with_totp=False):
                    raise RuntimeError(profile.get("errmsg") or "Unable to re-authenticate with FYERS.")
                return
            raise RuntimeError(profile.get("errmsg") or profile.get("message") or "FYERS session verification failed.")

    def websocket_access_token(self) -> str:
        """
        Returns the full token (client_id:access_token) required for WebSockets.
        """
        return f"{self.client_id}:{self._token_for_sdk()}"

    def get_historical_data(self, symbol: str, resolution: str, range_from: str, range_to: str) -> Optional[pd.DataFrame]:
        return self.fetch_history(symbol=symbol, resolution=resolution, range_from=range_from, range_to=range_to)

    def fetch_history(
        self,
        *,
        symbol: str,
        resolution: str,
        range_from: str,
        range_to: str,
        date_format: str = "1",
        cont_flag: str = "1",
    ) -> pd.DataFrame:
        self.verify_session()
        self._apply_rate_limit()
        logger.info("Fetching historical data for %s (%s) from %s to %s", symbol, resolution, range_from, range_to)
        payload = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": date_format,
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": cont_flag,
        }
        response = self.fyers.history(data=payload)
        if response.get("s") != "ok":
            logger.error("Failed to fetch FYERS history: %s", response.get("errmsg") or response.get("message"))
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(response.get("candles", []), columns=columns)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df

    def fetch_history_by_epoch(self, symbol: str, resolution: str, start_epoch: int, end_epoch: int) -> pd.DataFrame:
        self.verify_session()
        self._apply_rate_limit()
        payload = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",
            "range_from": str(start_epoch),
            "range_to": str(end_epoch),
            "cont_flag": "1",
        }
        response = self.fyers.history(data=payload)
        if response.get("s") != "ok":
            logger.error("Failed to fetch FYERS history by epoch: %s", response.get("errmsg") or response.get("message"))
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(response.get("candles", []), columns=columns)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df

    def subscribe_to_realtime_data(self, symbols: List[str], callback: Any):
        logger.warning("Realtime subscription is handled via the websocket-based live collector.")

    def get_option_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        self.verify_session()
        self._apply_rate_limit()
        logger.info("Fetching option chain for %s...", symbol)
        data = {"symbol": symbol, "strikecount": 10, "greeks": "1"}
        try:
            response = self.fyers.optionchain(data=data)
            if response.get("s") == "ok":
                return response.get("data", {})
            logger.error("Failed to fetch Fyers option chain: %s", response.get("errmsg"))
        except Exception as exc:
            logger.error("Error fetching Fyers option chain: %s", exc)
        return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        self.verify_session()
        self._apply_rate_limit()
        response = self.fyers.quotes(data={"symbols": ",".join(symbols)})
        if response.get("s") != "ok":
            logger.error("Failed to fetch FYERS quotes: %s", response.get("errmsg") or response.get("message"))
            return {}
        parsed: Dict[str, Dict[str, Any]] = {}
        for item in response.get("d", []):
            symbol = str(item.get("n", "")).strip()
            if symbol:
                parsed[symbol] = item.get("v") or {}
        return parsed

    def is_market_open_today(self) -> tuple[bool, str]:
        self.verify_session()
        try:
            response = self.fyers.market_status()
        except Exception as exc:
            raise RuntimeError(f"FYERS market status check failed: {exc}") from exc
        if response.get("s") != "ok":
            return False, response.get("errmsg") or response.get("message") or "Unknown market status failure"
        return _parse_nse_market_status(response.get("marketStatus", []))


FyersBrokerClient = FyersBroker
