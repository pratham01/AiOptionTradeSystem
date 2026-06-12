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
    # Prioritize exchange 10, segment 10, NORMAL market type
    for market in market_status:
        exchange = _as_int(market.get("exchange"))
        segment = _as_int(market.get("segment"))
        market_type = str(market.get("market_type", "")).upper()
        status = str(market.get("status", "")).upper()
        if exchange == 10 and segment == 10 and market_type == "NORMAL":
            if status in {"OPEN", "PREOPEN", "PREOPEN_CLOSED"}:
                return True, status
            return False, status or "CLOSED"
            
    # Fallback to the first matching exchange/segment if NORMAL is not found
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
        self._last_session_check = None

    def _check_auth_failure(self, response: dict[str, Any]) -> None:
        """Helper to mark authenticated status as False by clearing self.fyers if token expired."""
        if isinstance(response, dict) and response.get("s") != "ok":
            code = response.get("code")
            msg = str(response.get("errmsg") or response.get("message") or "").lower()
            if code in [-8, -17] or "token" in msg or "auth" in msg:
                self.fyers = None

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

            code = profile.get("code")

            # Rate-limited on /profile — token is still valid (WebSocket + history work).
            _RATE_LIMIT_CODES = {-353, -209}
            err_msg = str(profile.get("errmsg") or profile.get("message") or "")
            if code in _RATE_LIMIT_CODES or "limit" in err_msg.lower() or "429" in err_msg:
                logger.warning(
                    "Fyers /profile rate-limited (code %s). Token accepted as valid; proceeding.", code
                )
                return True

            if code in [-8, -17] and retry_with_totp:
                # Prior to triggering TOTP, check if another process wrote a valid token to fyers_token.json today
                from datetime import date
                import json
                cached_token = None
                token_path = Path(".secrets/fyers_token.json")
                if token_path.exists():
                    try:
                        payload = json.loads(token_path.read_text())
                        token_val = payload.get("access_token")
                        mtime = datetime.fromtimestamp(token_path.stat().st_mtime)
                        if token_val and mtime.date() == date.today() and token_val != self.access_token:
                            cached_token = token_val
                            logger.info("Found a newer cached token on disk. Reloading it.")
                    except Exception:
                        pass
                
                if cached_token:
                    self.access_token = cached_token
                    return self.authenticate(retry_with_totp=False)

                if self.authenticator:
                    logger.warning("Fyers token expired (code: %s). Attempting automated refresh via TOTP...", code)
                    try:
                        new_token = self.authenticator.generate_access_token()
                        self.access_token = new_token
                        self.authenticator.update_env_file(new_token)
                        return self.authenticate(retry_with_totp=False)
                    except Exception as exc:
                        logger.error("Failed to automatically refresh Fyers token: %s", exc)
                        return False

            logger.error(
                "Fyers authentication failed: %s (code: %s)",
                profile.get("errmsg") or profile.get("message"),
                code,
            )
            return False
        except Exception as exc:
            logger.error("Error during Fyers authentication: %s", exc)
            return False

    def verify_session(self) -> None:
        # If the SDK object isn't initialized yet, try to authenticate first.
        if not self.fyers:
            if not self.authenticate():
                if self.access_token:
                    # Force SDK init even if profile check was rate-limited
                    self.fyers = fyersModel.FyersModel(
                        client_id=self.client_id,
                        is_async=False,
                        token=self._token_for_sdk(),
                        log_path=str(self.log_path),
                    )
                    self.client = self.fyers
                    logger.warning("authenticate() failed but token present. Proceeding with SDK object.")
                else:
                    raise RuntimeError("Unable to authenticate with FYERS.")
            self._last_session_check = datetime.now()
            return
        
        # Check cache
        from datetime import timedelta
        now = datetime.now()
        if getattr(self, "_last_session_check", None) and now - self._last_session_check < timedelta(seconds=300):
            return

        # SDK object already exists — re-check profile only to detect token expiry.
        profile = self.fyers.get_profile()
        self._last_session_check = now
        if profile.get("s") == "ok":
            return
        code = profile.get("code")
        err_msg = str(profile.get("errmsg") or profile.get("message") or "")
        # Rate-limited — token still valid.
        if code in {-353, -209} or "limit" in err_msg.lower() or "429" in err_msg:
            logger.warning("Fyers /profile rate-limited during verify_session (code %s). Proceeding.", code)
            return
        # Token expired — refresh.
        if code in [-8, -17]:
            # Prior to triggering TOTP, check if another process wrote a valid token to fyers_token.json today
            from datetime import date
            import json
            cached_token = None
            token_path = Path(".secrets/fyers_token.json")
            if token_path.exists():
                try:
                    payload = json.loads(token_path.read_text())
                    token_val = payload.get("access_token")
                    mtime = datetime.fromtimestamp(token_path.stat().st_mtime)
                    if token_val and mtime.date() == date.today() and token_val != self.access_token:
                        cached_token = token_val
                        logger.info("Found a newer cached token on disk. Reloading it.")
                except Exception:
                    pass
            
            if cached_token:
                self.access_token = cached_token
                if self.authenticate(retry_with_totp=False):
                    return
            
            if self.authenticator:
                logger.warning("FYERS session expired during verification (code: %s). Refreshing token.", code)
                try:
                    new_token = self.authenticator.generate_access_token()
                    self.access_token = new_token
                    self.authenticator.update_env_file(new_token)
                    if not self.authenticate(retry_with_totp=False):
                        raise RuntimeError(profile.get("errmsg") or "Unable to re-authenticate with FYERS.")
                    return
                except Exception as exc:
                    logger.error("Failed to automatically refresh Fyers token: %s", exc)
                    raise RuntimeError("Unable to authenticate with Fyers") from exc
            return
        raise RuntimeError(profile.get("errmsg") or err_msg or "FYERS session verification failed.")

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
            self._check_auth_failure(response)
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

    def get_quotes(self, symbols: List[str], _retry_count: int = 0) -> Dict[str, Dict[str, Any]]:
        if not symbols:
            return {}
            
        # Batch symbols (Fyers allows 50 symbols per call)
        if len(symbols) > 50:
            all_quotes = {}
            for i in range(0, len(symbols), 50):
                batch = symbols[i:i+50]
                all_quotes.update(self.get_quotes(batch))
                time.sleep(0.8) # Prevent 429 rate limits
            return all_quotes

        self.verify_session()
        self._apply_rate_limit()
        
        try:
            response = self.fyers.quotes(data={"symbols": ",".join(symbols)})

            # 429 Handling
            if response.get("code") == 429 or "Limit Exceeded" in str(response.get("errmsg", "")):
                if _retry_count < 3:
                    logger.warning("Fyers Rate Limit (429) hit in legacy client. Retry %s/3 in 3s...", _retry_count+1)
                    time.sleep(3.0)
                    return self.get_quotes(symbols, _retry_count=_retry_count+1)
                else:
                    logger.error("Max retries hit for 429 in legacy client. Returning empty quotes.")
                    return {}

            if response.get("s") != "ok":
                self._check_auth_failure(response)
                logger.error("Failed to fetch FYERS quotes: %s", response.get("errmsg") or response.get("message"))
                return {}

            parsed: Dict[str, Dict[str, Any]] = {}
            for item in response.get("d", []):
                symbol = str(item.get("n", "")).strip()
                if symbol:
                    parsed[symbol] = item.get("v") or {}
            return parsed

        except Exception as exc:
            if _retry_count < 3:
                logger.warning("Fyers quotes request failed: %s. Retry %s/3 in 3.0s...", exc, _retry_count+1)
                time.sleep(3.0)
                return self.get_quotes(symbols, _retry_count=_retry_count+1)
            else:
                logger.error("Failed to fetch quotes in legacy client: %s", exc)
                return {}

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
