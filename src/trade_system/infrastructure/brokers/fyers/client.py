"""
Fyers Broker Client Implementation - Refactored to new interface.

This is a refactored version of the original FyersBroker that implements
the new Broker interface for consistency with Dhan broker.
"""

from __future__ import annotations
import time

import logging
from datetime import date, datetime
from typing import Any
from threading import Lock

import pandas as pd
from fyers_apiv3 import fyersModel

from ....core.ports.broker import (
    AuthenticationError,
    Broker,
    DataBroker,
    DataFetchError,
    HistoricalData,
    MarketQuote,
    OrderError,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
)

LOGGER = logging.getLogger(__name__)


class FyersBrokerV2(DataBroker):
    """
    Fyers Broker implementation for market data.

    This is the new interface-compatible version.
    The original FyersBroker is kept for backward compatibility.
    """

    def __init__(
        self,
        client_id: str,
        access_token: str,
        user_id: str | None = None,
        authenticator: Any | None = None,
    ) -> None:
        super().__init__("Fyers", client_id, access_token)
        self.user_id = user_id
        self.authenticator = authenticator
        self.fyers = None
        self.client = None
        self._last_request_time = 0.0
        self._rate_limit_lock = Lock()

    def _apply_rate_limit(self):
        """Ensures at least 0.25s between requests across all threads."""
        with self._rate_limit_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < 0.25:
                time.sleep(0.25 - elapsed)
            self._last_request_time = time.time()

    def _token_for_sdk(self) -> str:
        """
        Returns the JWT part of the token. Fyers REST SDK expects just the JWT 
        as it prepends the client_id internally.
        """
        if ":" in self.access_token:
            return self.access_token.split(":", 1)[1]
        return self.access_token

    def authenticate(self, retry_with_totp: bool = True) -> bool:
        """Authenticate with Fyers API."""
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                is_async=False,
                token=self._token_for_sdk(),
            )
            self.client = self.fyers
            profile = self.fyers.get_profile()

            if profile.get("s") == "ok":
                self._authenticated = True
                LOGGER.info("Fyers authentication successful")
                return True

            code = profile.get("code")
            _RATE_LIMIT_CODES = {-353, -209}
            err_msg = str(profile.get("errmsg") or profile.get("message") or "")
            if code in _RATE_LIMIT_CODES or "limit" in err_msg.lower() or "429" in err_msg:
                self._authenticated = True
                LOGGER.warning(f"Fyers /profile rate-limited (code {code}). Token accepted as valid.")
                return True

            if code in [-8, -17] and retry_with_totp and self.authenticator:
                LOGGER.warning(f"Fyers authentication failed (code: {code}). Attempting automated refresh via TOTP...")
                try:
                    new_token = self.authenticator.generate_access_token()
                    self.access_token = new_token
                    if hasattr(self.authenticator, "update_env_file"):
                        self.authenticator.update_env_file(new_token)
                    return self.authenticate(retry_with_totp=False)
                except Exception as exc:
                    LOGGER.error(f"Failed to automatically refresh Fyers token: {exc}")
                    return False

            LOGGER.error(
                f"Fyers authentication failed: {err_msg} "
                f"(code: {code})"
            )
            return False

        except Exception as e:
            LOGGER.error(f"Error during Fyers authentication: {e}")
            return False

    def verify_session(self) -> None:
        """Verify the current session is active, refresh if needed."""
        if not self.fyers:
            if not self.authenticate():
                if self.access_token:
                    # Force SDK init even if profile check was rate-limited
                    self.fyers = fyersModel.FyersModel(
                        client_id=self.client_id,
                        is_async=False,
                        token=self._token_for_sdk(),
                    )
                    self.client = self.fyers
                    self._authenticated = True
                    LOGGER.warning("authenticate() failed but token present. Proceeding with SDK object.")
                else:
                    raise AuthenticationError("Unable to authenticate with FYERS.")
            return

        profile = self.fyers.get_profile()
        if profile.get("s") == "ok":
            return
            
        code = profile.get("code")
        err_msg = str(profile.get("errmsg") or profile.get("message") or "")
        
        if code in {-353, -209} or "limit" in err_msg.lower() or "429" in err_msg:
            LOGGER.warning(f"Fyers /profile rate-limited during verify_session (code {code}). Proceeding.")
            return

        if code in [-8, -17] and self.authenticator:
            LOGGER.warning(f"FYERS session invalid (code: {code}). Refreshing token.")
            new_token = self.authenticator.generate_access_token()
            self.access_token = new_token
            if hasattr(self.authenticator, "update_env_file"):
                self.authenticator.update_env_file(new_token)
            if not self.authenticate(retry_with_totp=False):
                raise AuthenticationError(profile.get("errmsg") or "Unable to re-authenticate with FYERS.")
            return
            
        raise DataFetchError(profile.get("errmsg") or err_msg or "FYERS session verification failed.")

    def get_quotes(self, symbols: list[str], _retry_count: int = 0) -> dict[str, MarketQuote]:
        """Get real-time quotes with 429 recovery and batching."""
        if not symbols: return {}
        
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
            data = self.fyers.quotes(data={"symbols": ",".join(symbols)})

            # 429 Handling
            if data.get("code") == 429 or "Limit Exceeded" in str(data.get("errmsg", "")):
                if _retry_count < 3:
                    LOGGER.warning(f"Fyers Rate Limit (429) hit. Retry {_retry_count+1}/3 in 3s...")
                    time.sleep(3.0)
                    return self.get_quotes(symbols, _retry_count=_retry_count+1)
                else:
                    LOGGER.error("Max retries hit for 429. Returning empty quotes to prevent crash.")
                    return {}

            if data.get("s") != "ok":
                # Return empty instead of raising to keep the UI alive
                LOGGER.error(f"Fyers API error: {data.get('errmsg') or data.get('message')}")
                return {}

            quotes = {}
            for item in data.get("d", []):
                symbol = item.get("n", "")
                try:
                    v = item.get("v", {})
                    quotes[symbol] = MarketQuote(
                        symbol=symbol,
                        exchange=symbol.split(":")[0] if ":" in symbol else "NSE",
                        last_price=float(v.get("lp", 0)),
                        open=float(v.get("open", 0)),
                        high=float(v.get("high", 0)),
                        low=float(v.get("low", 0)),
                        close=float(v.get("close", 0)),
                        previous_close=float(v.get("prev_close_price", 0)),
                        volume=int(v.get("volume", 0)),
                        change=float(v.get("change", 0)),
                        change_percent=float(v.get("chp", 0)),
                        timestamp=datetime.now(),
                        bid=float(v.get("bid_price", 0)),
                        ask=float(v.get("ask_price", 0)),
                        bid_qty=int(v.get("bid_qty", 0)),
                        ask_qty=int(v.get("ask_qty", 0)),
                        ltp=float(v.get("lp", 0)),
                    )
                except: continue
            return quotes

        except Exception as e:
            # If quotes call raised JSONDecodeError/ConnectionError (e.g. HTML 429/502), retry
            if _retry_count < 3:
                LOGGER.warning(f"Fyers quotes request failed: {e}. Retry {_retry_count+1}/3 in 3.0s...")
                time.sleep(3.0)
                return self.get_quotes(symbols, _retry_count=_retry_count+1)
            else:
                LOGGER.error(f"Failed to fetch quotes: {e}")
                return {} # Safe fallback

    def get_historical_data(
        self,
        symbol: str,
        start_date_or_res: str | date,
        end_date_or_from: str | date,
        timeframe_or_to: str = "DAY",
    ) -> list[HistoricalData] | pd.DataFrame:
        """
        Polymorphic method supporting both new and legacy interfaces.
        
        New: (symbol, start_date: date, end_date: date, timeframe: str) -> list[HistoricalData]
        Legacy: (symbol, resolution: str, range_from: str, range_to: str) -> pd.DataFrame
        """
        if isinstance(start_date_or_res, date):
            # New interface
            return self._get_historical_data_impl(
                symbol=symbol,
                start_date=start_date_or_res,
                end_date=end_date_or_from,
                timeframe=timeframe_or_to
            )
        else:
            # Legacy interface
            return self.fetch_history(
                symbol=symbol,
                resolution=start_date_or_res,
                range_from=end_date_or_from,
                range_to=timeframe_or_to
            )

    def _get_historical_data_impl(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "DAY",
    ) -> list[HistoricalData]:
        """Implementation of the new DataBroker historical data fetch."""
        self.verify_session()
        self._apply_rate_limit()

        try:
            # Convert timeframe to Fyers format
            fyers_timeframe = self._map_timeframe(timeframe)

            # Convert dates to epoch timestamps
            start_epoch = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_epoch = int(datetime.combine(end_date, datetime.max.time()).timestamp())

            payload = {
                "symbol": symbol,
                "resolution": fyers_timeframe,
                "date_format": "0", # 0 for epoch, 1 for YYYY-MM-DD
                "range_from": start_epoch,
                "range_to": end_epoch,
                "cont_flag": "1",
            }

            response = self.fyers.history(data=payload)

            if response.get("s") != "ok":
                err = response.get("errmsg") or response.get("message") or "Unknown Fyers API error"
                raise DataFetchError(f"Fyers history API error: {err}")

            candles = response.get("candles", [])
            historical_data = []

            for candle in candles:
                try:
                    # Candle format: [timestamp, open, high, low, close, volume]
                    if len(candle) >= 6:
                        ts = datetime.fromtimestamp(candle[0])
                        historical_data.append(
                            HistoricalData(
                                timestamp=ts,
                                open=float(candle[1]),
                                high=float(candle[2]),
                                low=float(candle[3]),
                                close=float(candle[4]),
                                volume=int(candle[5]),
                            )
                        )
                except (ValueError, IndexError) as e:
                    LOGGER.warning(f"Failed to parse candle: {candle}, error: {e}")
                    continue

            LOGGER.info(
                f"Fetched {len(historical_data)} candles for {symbol} "
                f"from {start_date} to {end_date}"
            )
            return historical_data

        except Exception as e:
            LOGGER.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise DataFetchError(f"Historical data fetch failed: {e}")

    def get_market_status(self) -> dict[str, Any]:
        """Get market status."""
        try:
            data = self.fyers.market_status()

            if data.get("s") != "ok":
                return {"status": "unknown", "market_open": False}

            # Parse NSE market status
            for market in data.get("marketStatus", []):
                if market.get("exchange") == 10 and market.get("segment") == 10:  # NSE Equity
                    status = market.get("status", "").upper()
                    return {
                        "status": status,
                        "exchange": "NSE",
                        "market_open": status in {"OPEN", "PREOPEN", "PREOPEN_CLOSED"},
                    }

            return {"status": "unknown", "market_open": False}

        except Exception as e:
            LOGGER.warning(f"Failed to get market status: {e}")
            return {"status": "unknown", "market_open": False}

    @staticmethod
    def _map_timeframe(timeframe: str) -> str:
        """Map generic timeframe to Fyers format."""
        mapping = {
            "DAY": "1D",
            "1H": "60",
            "30MIN": "30",
            "15MIN": "15",
            "5MIN": "5",
            "1MIN": "1",
        }
        return mapping.get(timeframe.upper(), "1D")

    def __repr__(self) -> str:
        return f"FyersBrokerV2(client_id={self.client_id}, authenticated={self._authenticated})"


# Backward compatibility wrapper
class FyersBroker(FyersBrokerV2):
    """
    Backward compatible wrapper for original FyersBroker.

    This maintains the old interface while internally using the new implementation.
    """

    def __init__(
        self,
        client_id: str,
        access_token: str,
        user_id: str | None = None,
        log_path: str | None = None,
        authenticator: Any = None,
    ) -> None:
        super().__init__(client_id, access_token, user_id)
        # Store legacy attributes for compatibility
        self.log_path = log_path
        self.authenticator = authenticator
        self.settings = getattr(authenticator, "settings", None) if authenticator else None

    def verify_session(self) -> None:
        """Legacy method - authenticate if not already."""
        if not self._authenticated:
            if not self.authenticate():
                raise RuntimeError("Unable to authenticate with Fyers")

    def websocket_access_token(self) -> str:
        """Get websocket access token (appid:token)."""
        return f"{self.client_id}:{self._token_for_sdk()}"

    def fetch_history(
        self,
        *,
        symbol: str,
        resolution: str,
        range_from: str | date,
        range_to: str | date,
        date_format: str = "1",
        cont_flag: str = "1",
    ) -> pd.DataFrame:
        """
        Legacy method - convert to new interface and return DataFrame.
        """
        # Parse dates safely
        if isinstance(range_from, str):
            start_date = datetime.fromisoformat(range_from).date()
        else:
            start_date = range_from

        if isinstance(range_to, str):
            end_date = datetime.fromisoformat(range_to).date()
        else:
            end_date = range_to

        # Get historical data using implementation method
        historical_data = self._get_historical_data_impl(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=resolution,
        )

        # Convert to DataFrame (legacy format)
        if not historical_data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        data = []
        for item in historical_data:
            data.append([
                item.timestamp.timestamp(),
                item.open,
                item.high,
                item.low,
                item.close,
                item.volume,
            ])

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(
            "Asia/Kolkata"
        ).dt.tz_localize(None)
        return df

    def fetch_history_by_epoch(
        self,
        symbol: str,
        resolution: str,
        start_epoch: int,
        end_epoch: int,
    ) -> pd.DataFrame:
        """Legacy method - convert epoch to ISO dates."""
        start_date = datetime.fromtimestamp(start_epoch).date()
        end_date = datetime.fromtimestamp(end_epoch).date()
        return self.fetch_history(
            symbol=symbol,
            resolution=resolution,
            range_from=start_date.isoformat(),
            range_to=end_date.isoformat(),
        )
