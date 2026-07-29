from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
from fyers_apiv3 import fyersModel

from .base import BaseBroker
from .fyers_auth import FyersAuthenticator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fyers API error code sets
# ---------------------------------------------------------------------------
_TOKEN_EXPIRED_CODES: frozenset[int] = frozenset([-8, -16, -17])
_RATE_LIMIT_CODES: frozenset[int] = frozenset([-353, -209, 429])

_OHLCV_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_nse_market_status(market_status: list[dict[str, Any]]) -> tuple[bool, str]:
    """Parse Fyers market status response and return (is_open, label) for NSE."""
    for market in market_status:
        exchange = _as_int(market.get("exchange"))
        segment = _as_int(market.get("segment"))
        market_type = str(market.get("market_type", "")).upper()
        status = str(market.get("status", "")).upper()
        if exchange == 10 and segment == 10 and market_type == "NORMAL":
            return (True, status) if status in {"OPEN", "PREOPEN", "PREOPEN_CLOSED"} else (False, status or "CLOSED")

    for market in market_status:
        exchange = _as_int(market.get("exchange"))
        segment = _as_int(market.get("segment"))
        status = str(market.get("status", "")).upper()
        if exchange == 10 and segment == 10:
            return (True, status) if status in {"OPEN", "PREOPEN", "PREOPEN_CLOSED"} else (False, status or "CLOSED")

    return False, "NSE market status not present"


def _parse_candle_df(candles: list[list]) -> pd.DataFrame:
    """Convert raw Fyers candle list to a timezone-naive IST DataFrame."""
    df = pd.DataFrame(candles, columns=_OHLCV_COLUMNS)
    if not df.empty:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="s", utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)
        )
    return df


# ---------------------------------------------------------------------------
# Broker class
# ---------------------------------------------------------------------------

class FyersBroker(BaseBroker):
    """
    Fyers REST + WebSocket broker client.

    Design principles
    -----------------
    * **Idempotent token refresh**: ``_refresh_token()`` is the single entry-point
      for TOTP re-auth.  It is protected by ``_token_refresh_lock`` so concurrent
      threads cannot trigger multiple simultaneous TOTP calls.
    * **Session check caching**: REST profile checks are rate-limited to once every
      5 minutes (``_SESSION_CHECK_TTL_SECS``) to avoid API 429 errors.
    * **Fail-safe**: If refresh fails but a token is present, callers proceed rather
      than crash, mirroring the existing graceful degradation for rate-limited calls.
    """

    _SESSION_CHECK_TTL_SECS: int = 300  # 5 minutes

    def __init__(
        self,
        client_id: str,
        access_token: str,
        user_id: str | None = None,
        log_path: str | None = None,
        authenticator: FyersAuthenticator | None = None,
    ) -> None:
        self.client_id = client_id
        self.access_token = access_token
        self.user_id = user_id
        self.log_path = Path(log_path or "logs/")
        self.log_path.mkdir(parents=True, exist_ok=True)

        if authenticator is None:
            from trade_system.shared.config import Settings
            from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthenticator
            self.settings = getattr(self, "settings", None) or Settings.load()
            self.authenticator = FyersAuthenticator(self.settings)
        else:
            self.authenticator = authenticator
            self.settings = getattr(authenticator, "settings", None)

        # Fyers SDK model object — None until first authenticate()
        self.fyers: fyersModel.FyersModel | None = None
        self.client = self.fyers  # legacy alias

        # Fyers API limit: ~10 req/s — enforce >= 110 ms between REST calls
        self._last_request_time: float = 0.0
        self._rate_limit_lock = Lock()

        # Thread-safe token refresh: serialise concurrent callers
        self._token_refresh_lock = Lock()

        # Session check cache
        self._last_session_check: datetime | None = None

        # Websocket token cache — invalidated whenever access_token changes
        self._ws_token_cache: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_for_sdk(self) -> str:
        """
        Return the bare JWT expected by the Fyers REST SDK.
        The SDK prepends ``client_id:`` internally — do NOT include it here.
        """
        token = self.access_token or ""
        return token.split(":", 1)[1] if ":" in token else token

    def _apply_rate_limit(self) -> None:
        """Ensure at least 110 ms between REST requests across all threads."""
        with self._rate_limit_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < 0.11:
                time.sleep(0.11 - elapsed)
            self._last_request_time = time.time()

    def _check_auth_failure(self, response: dict[str, Any]) -> None:
        """Invalidate the SDK session if the API reports token expiry."""
        if not isinstance(response, dict) or response.get("s") == "ok":
            return
        code = response.get("code")
        msg = str(response.get("errmsg") or response.get("message") or "").lower()
        if code in _TOKEN_EXPIRED_CODES or "token" in msg or "auth" in msg:
            self.fyers = None
            self.client = None
            self._ws_token_cache = None

    def _is_rate_limited(self, response: dict[str, Any]) -> bool:
        """Return True if the API response indicates rate-limiting (not auth error)."""
        code = response.get("code")
        errmsg = str(response.get("errmsg") or response.get("message") or "")
        return code in _RATE_LIMIT_CODES or "limit" in errmsg.lower() or "429" in errmsg

    def _read_disk_token(self) -> str | None:
        """
        Read a token written to ``.secrets/fyers_token.json`` today by another process.
        Returns the token string, or None if not present / stale / same as current.
        """
        token_path = Path(".secrets/fyers_token.json")
        if not token_path.exists():
            return None
        try:
            payload = json.loads(token_path.read_text())
            token_val: str | None = payload.get("access_token")
            mtime = datetime.fromtimestamp(token_path.stat().st_mtime)
            if token_val and mtime.date() == date.today() and token_val != self.access_token:
                return token_val
        except Exception:
            pass
        return None

    def _init_sdk(self) -> None:
        """(Re)initialise the Fyers SDK model from the current access_token."""
        self.fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            is_async=False,
            token=self._token_for_sdk(),
            log_path=str(self.log_path),
        )
        self.client = self.fyers
        self._ws_token_cache = None  # invalidate cached WS token

    def _refresh_token(self, *, retry_with_totp: bool = True) -> bool:
        """
        Single entry-point for token refresh, protected by ``_token_refresh_lock``.

        Refresh sequence:
        1. Check if another thread already refreshed (session check TTL guard).
        2. Look for a fresher token on disk written by a sibling process today.
        3. Run TOTP re-authentication via the configured authenticator.

        Returns True if a usable token is now in place.
        """
        with self._token_refresh_lock:
            # Guard: another thread may have already refreshed while we waited
            if self._last_session_check and (
                datetime.now() - self._last_session_check
            ).total_seconds() < self._SESSION_CHECK_TTL_SECS:
                return self.fyers is not None

            # Check disk for a fresher token from a sibling process
            disk_token = self._read_disk_token()
            if disk_token:
                logger.info("Loaded fresher Fyers token from disk.")
                self.access_token = disk_token
                self._init_sdk()
                return True

            # TOTP re-authentication
            if retry_with_totp and self.authenticator:
                logger.warning("Fyers token expired — attempting automated TOTP refresh.")
                try:
                    new_token = self.authenticator.generate_access_token()
                    self.access_token = new_token
                    self.authenticator.update_env_file(new_token)
                    self._init_sdk()
                    logger.info("TOTP refresh successful.")
                    return True
                except Exception as exc:
                    logger.error("TOTP refresh failed: %s", exc)

            return False

    # ------------------------------------------------------------------
    # Public session management
    # ------------------------------------------------------------------

    def authenticate(self, retry_with_totp: bool = True) -> bool:
        """
        Initialise (or reinitialise) the Fyers SDK and validate the token.

        Returns True if the session is usable, including when the /profile
        endpoint is rate-limited (token is still valid for WS + history).
        """
        logger.info("Authenticating with Fyers.")
        try:
            self._init_sdk()
            profile = self.fyers.get_profile()

            if profile.get("s") == "ok":
                logger.info("Fyers authentication successful.")
                self._last_session_check = datetime.now()
                return True

            code = profile.get("code")

            if self._is_rate_limited(profile):
                logger.warning(
                    "Fyers /profile rate-limited (code %s). Token accepted as valid.", code
                )
                self._last_session_check = datetime.now()
                return True

            if code in _TOKEN_EXPIRED_CODES:
                if self._refresh_token(retry_with_totp=retry_with_totp):
                    return self.authenticate(retry_with_totp=False)

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
        """
        Ensure the Fyers SDK is initialised and the token is valid.

        * SDK uninitialised → calls ``authenticate()``.
        * Within session cache TTL → skip REST round-trip.
        * Token expired → delegates to ``_refresh_token()`` then re-authenticates.

        Raises ``RuntimeError`` only if no token is available at all.
        """
        # SDK not initialised
        if not self.fyers:
            if not self.authenticate():
                if self.access_token:
                    self._init_sdk()
                    logger.warning("authenticate() failed but token present — proceeding.")
                else:
                    raise RuntimeError("Unable to authenticate with FYERS — no token available.")
            self._last_session_check = datetime.now()
            return

        # Within cache TTL — skip REST round-trip
        if self._last_session_check and (
            datetime.now() - self._last_session_check
        ).total_seconds() < self._SESSION_CHECK_TTL_SECS:
            return

        # Re-check profile
        profile = self.fyers.get_profile()
        self._last_session_check = datetime.now()

        if profile.get("s") == "ok":
            return

        code = profile.get("code")

        if self._is_rate_limited(profile):
            logger.warning("Fyers /profile rate-limited in verify_session (code %s). Proceeding.", code)
            return

        if code in _TOKEN_EXPIRED_CODES:
            if self._refresh_token(retry_with_totp=True):
                if not self.authenticate(retry_with_totp=False):
                    raise RuntimeError("Unable to re-authenticate with FYERS after token refresh.")
                return
            if self.access_token:
                logger.warning("Token refresh failed — proceeding with existing token.")
                return
            raise RuntimeError(profile.get("errmsg") or "FYERS token expired and refresh failed.")

        raise RuntimeError(
            profile.get("errmsg")
            or str(profile.get("message") or "")
            or "FYERS session verification failed."
        )

    def websocket_access_token(self) -> str:
        """
        Return the full ``client_id:jwt`` token required by the Fyers WebSocket SDK.
        Cached and invalidated whenever the underlying access_token changes.
        """
        if self._ws_token_cache is None:
            self._ws_token_cache = f"{self.client_id}:{self._token_for_sdk()}"
        return self._ws_token_cache

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_historical_data(
        self, symbol: str, resolution: str, range_from: str, range_to: str
    ) -> Optional[pd.DataFrame]:
        return self.fetch_history(
            symbol=symbol, resolution=resolution, range_from=range_from, range_to=range_to
        )

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
        logger.info(
            "Fetching historical data for %s (%s) from %s to %s",
            symbol, resolution, range_from, range_to,
        )
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
            logger.error(
                "Failed to fetch FYERS history: %s",
                response.get("errmsg") or response.get("message"),
            )
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        return _parse_candle_df(response.get("candles", []))

    def fetch_history_by_epoch(
        self, symbol: str, resolution: str, start_epoch: int, end_epoch: int
    ) -> pd.DataFrame:
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
            logger.error(
                "Failed to fetch FYERS history by epoch: %s",
                response.get("errmsg") or response.get("message"),
            )
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        return _parse_candle_df(response.get("candles", []))

    def subscribe_to_realtime_data(self, symbols: List[str], callback: Any) -> None:
        logger.warning("Realtime subscription is handled via the websocket-based live collector.")

    def get_option_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        self.verify_session()
        self._apply_rate_limit()
        logger.info("Fetching option chain for %s.", symbol)
        try:
            response = self.fyers.optionchain(
                data={"symbol": symbol, "strikecount": 10, "greeks": "1"}
            )
            if response.get("s") == "ok":
                return response.get("data", {})
            logger.error("Failed to fetch option chain: %s", response.get("errmsg"))
        except Exception as exc:
            logger.error("Error fetching option chain: %s", exc)
        return None

    def get_quotes(self, symbols: List[str], _retry_count: int = 0) -> Dict[str, Dict[str, Any]]:
        if not symbols:
            return {}

        # Fyers max: 50 symbols per call — batch larger requests
        if len(symbols) > 50:
            all_quotes: Dict[str, Dict[str, Any]] = {}
            for i in range(0, len(symbols), 50):
                all_quotes.update(self.get_quotes(symbols[i : i + 50]))
                time.sleep(0.8)  # avoid 429 between batches
            return all_quotes

        self.verify_session()
        self._apply_rate_limit()

        try:
            response = self.fyers.quotes(data={"symbols": ",".join(symbols)})

            if response.get("code") == 429 or "Limit Exceeded" in str(response.get("errmsg", "")):
                if _retry_count < 3:
                    logger.warning("Rate limit hit in get_quotes. Retry %d/3 in 3 s.", _retry_count + 1)
                    time.sleep(3.0)
                    return self.get_quotes(symbols, _retry_count=_retry_count + 1)
                logger.error("Max retries exhausted for 429 in get_quotes.")
                return {}

            if response.get("s") != "ok":
                self._check_auth_failure(response)
                logger.error(
                    "Failed to fetch quotes: %s",
                    response.get("errmsg") or response.get("message"),
                )
                return {}

            return {
                str(item.get("n", "")).strip(): (item.get("v") or {})
                for item in response.get("d", [])
                if str(item.get("n", "")).strip()
            }

        except Exception as exc:
            if _retry_count < 3:
                logger.warning("Quotes request failed: %s. Retry %d/3.", exc, _retry_count + 1)
                time.sleep(3.0)
                return self.get_quotes(symbols, _retry_count=_retry_count + 1)
            logger.error("get_quotes failed after retries: %s", exc)
            return {}

    def is_market_open_today(self) -> tuple[bool, str]:
        self.verify_session()
        try:
            response = self.fyers.market_status()
        except Exception as exc:
            raise RuntimeError(f"FYERS market status check failed: {exc}") from exc
        if response.get("s") != "ok":
            return False, (
                response.get("errmsg") or response.get("message") or "Unknown market status failure"
            )
        return _parse_nse_market_status(response.get("marketStatus", []))

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: int,
        quantity: int,
        order_type: int,
        product_type: str,
        price: float = 0.0,
        stoploss: float = 0.0,
    ) -> Dict[str, Any]:
        """Place a new order via the Fyers API."""
        self.verify_session()
        self._apply_rate_limit()
        logger.info(
            "Placing order — symbol=%s side=%s qty=%d type=%d",
            symbol, "BUY" if side == 1 else "SELL", quantity, order_type,
        )
        data = {
            "symbol": symbol,
            "qty": quantity,
            "type": order_type,
            "side": side,
            "productType": product_type,
            "limitPrice": price,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": stoploss,
            "takeProfit": 0,
        }
        try:
            response = self.fyers.place_order(data=data)
            if response.get("s") != "ok":
                logger.error("Failed to place order: %s", response.get("errmsg") or response.get("message"))
            return response
        except Exception as exc:
            logger.error("Error placing order: %s", exc)
            return {"s": "error", "message": str(exc)}

    def modify_order(
        self, order_id: str, quantity: int, order_type: int, price: float = 0.0
    ) -> Dict[str, Any]:
        self.verify_session()
        self._apply_rate_limit()
        data = {"id": order_id, "type": order_type, "limitPrice": price, "qty": quantity}
        try:
            response = self.fyers.modify_order(data=data)
            if response.get("s") != "ok":
                logger.error("Failed to modify order %s: %s", order_id, response.get("errmsg"))
            return response
        except Exception as exc:
            logger.error("Error modifying order: %s", exc)
            return {"s": "error", "message": str(exc)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        self.verify_session()
        self._apply_rate_limit()
        try:
            response = self.fyers.cancel_order(data={"id": order_id})
            if response.get("s") != "ok":
                logger.error("Failed to cancel order %s: %s", order_id, response.get("errmsg"))
            return response
        except Exception as exc:
            logger.error("Error canceling order: %s", exc)
            return {"s": "error", "message": str(exc)}

    def get_positions(self) -> List[Dict[str, Any]]:
        self.verify_session()
        self._apply_rate_limit()
        try:
            response = self.fyers.positions()
            if response.get("s") == "ok":
                return response.get("netPositions", [])
            logger.error("Failed to fetch positions: %s", response.get("errmsg"))
        except Exception as exc:
            logger.error("Error fetching positions: %s", exc)
        return []

    def get_funds(self) -> Dict[str, Any]:
        self.verify_session()
        self._apply_rate_limit()
        try:
            response = self.fyers.funds()
            if response.get("s") == "ok":
                return {f.get("title"): f.get("equityAmount") for f in response.get("fund_limit", [])}
            logger.error("Failed to fetch funds: %s", response.get("errmsg"))
        except Exception as exc:
            logger.error("Error fetching funds: %s", exc)
        return {}


# Legacy alias — maintained for backward compatibility with existing callers
FyersBrokerClient = FyersBroker
