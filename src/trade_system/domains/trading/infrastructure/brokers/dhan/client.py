"""
Dhan Broker Client Implementation (v2) - SDK Powered.

Reference: https://dhanhq.co/docs/v2/
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from dhanhq import dhanhq

from trade_system.domains.trading.domain.ports.broker import (
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


class DhanBroker(Broker):
    """
    Dhan Broker implementation using official dhanhq SDK.
    """

    def __init__(
        self,
        client_id: str,
        access_token: str,
        api_key: str | None = None,
    ) -> None:
        super().__init__("Dhan", client_id, access_token)
        self.dhan = dhanhq(client_id, access_token)
        self._security_master: dict[str, str] = {}
        self._master_loaded = False

    def _load_security_master(self) -> None:
        """Load security master for symbol resolution."""
        if self._master_loaded:
            return

        cache_path = Path("data/dhan_master.csv")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import csv
            import requests
            
            # Use public scrip master URL
            url = "https://images.dhan.co/api-data/api-scrip-master.csv"
            
            should_download = True
            if cache_path.exists():
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=24):
                    should_download = False

            if should_download:
                LOGGER.info("Downloading Dhan scrip master...")
                res = requests.get(url, timeout=60)
                res.raise_for_status()
                cache_path.write_text(res.text)
            
            with open(cache_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get("SEM_TRADING_SYMBOL", "").strip()
                    sec_id = row.get("SEM_SMST_SECURITY_ID", "").strip()
                    exch = row.get("SEM_EXM_EXCH_ID", "").strip()
                    
                    if symbol and sec_id and exch in ["NSE", "IDX_I"]:
                        self._security_master[symbol] = sec_id
            
            self._master_loaded = True
            LOGGER.info(f"Loaded {len(self._security_master)} symbols from Dhan master")
        except Exception as e:
            LOGGER.error(f"Failed to load scrip master: {e}")

    def _get_security_id(self, symbol: str) -> str:
        if not self._master_loaded: self._load_security_master()
        return self._security_master.get(symbol, symbol)

    def authenticate(self) -> bool:
        """Authenticate with Dhan."""
        try:
            res = self.dhan.get_fund_limits()
            if res.get("status") == "success":
                self._authenticated = True
                LOGGER.info("Dhan authentication successful")
                return True
            else:
                remarks = res.get("remarks", {})
                err = remarks.get("error_message") or "Unknown error"
                LOGGER.error(f"Dhan authentication failed: {err}")
                self._authenticated = False
                return False
        except Exception as e:
            LOGGER.error(f"Dhan connection error: {e}")
            return False

    def get_quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """Get real-time quotes."""
        if not self._authenticated: self.authenticate()
        
        quotes = {}
        for symbol in symbols:
            try:
                clean_sym = self._convert_symbol(symbol)
                sec_id = self._get_security_id(clean_sym)
                
                # Using ticker_data (LTP only) for speed and broad scope
                payload = {
                    "InstrumentInfoList": [{
                        "exchangeSegment": "NSE_EQ",
                        "instrumentId": sec_id
                    }]
                }
                res = self.dhan.ticker_data(payload)
                
                if res.get("status") == "success":
                    data_list = res.get("data", {}).get("data", [])
                    if data_list:
                        data = data_list[0]
                        quote = MarketQuote(
                            symbol=symbol,
                            exchange="NSE",
                            last_price=float(data.get("lp", 0)),
                            open=float(data.get("o", 0)),
                            high=float(data.get("h", 0)),
                            low=float(data.get("l", 0)),
                            close=float(data.get("lp", 0)),
                            previous_close=float(data.get("pc", 0)),
                            volume=int(data.get("v", 0)),
                            change=float(data.get("ch", 0)),
                            change_percent=float(data.get("cp", 0)),
                            timestamp=datetime.now(),
                            ltp=float(data.get("lp", 0)),
                        )
                        quotes[symbol] = quote
            except Exception as e:
                LOGGER.warning(f"Dhan quote error for {symbol}: {e}")
        return quotes

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "DAY",
    ) -> list[HistoricalData]:
        """Get historical OHLCV data."""
        if not self._authenticated: self.authenticate()
        
        try:
            clean_symbol = self._convert_symbol(symbol)
            dhan_timeframe = self._map_timeframe(timeframe)
            
            res = self.dhan.historical_daily_data(
                symbol=clean_symbol,
                exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
                expiry_code=0,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            if res.get("status") == "success":
                candles = res.get("data", {}).get("candles", [])
                historical_data = []
                for c in candles:
                    historical_data.append(
                        HistoricalData(
                            timestamp=datetime.fromtimestamp(c[0]),
                            open=float(c[1]),
                            high=float(c[2]),
                            low=float(c[3]),
                            close=float(c[4]),
                            volume=int(c[5]),
                        )
                    )
                return historical_data
            else:
                LOGGER.error(f"Dhan history failed for {symbol}: {res}")
                return []
        except Exception as e:
            LOGGER.error(f"Dhan history exception for {symbol}: {e}")
            return []

    def get_market_status(self) -> dict[str, Any]:
        """Get market status."""
        return {"status": "open" if self._authenticated else "closed", "market_open": self._authenticated}

    def place_order(self, order: OrderRequest) -> OrderResponse:
        raise NotImplementedError("Dhan trading not fully implemented")

    def get_order_status(self, order_id: str) -> OrderResponse | None:
        return None

    def get_positions(self) -> list[dict[str, Any]]:
        try:
            res = self.dhan.get_positions()
            return res.get("data", [])
        except:
            return []

    def get_holdings(self) -> list[dict[str, Any]]:
        try:
            res = self.dhan.get_holdings()
            return res.get("data", [])
        except:
            return []

    def get_funds(self) -> dict[str, float]:
        try:
            res = self.dhan.get_fund_limits()
            data = res.get("data", {})
            return {
                "available": float(data.get("availabelBalance") or data.get("available_balance", 0)),
                "used": float(data.get("utilizedAmount") or data.get("utilized_amount", 0)),
                "total": float(data.get("sodLimit", 0)),
            }
        except:
            return {"available": 0.0, "used": 0.0, "total": 0.0}

    @staticmethod
    def _convert_symbol(symbol: str) -> str:
        if symbol.startswith("NSE:"): symbol = symbol[4:]
        if symbol.endswith("-EQ"): symbol = symbol[:-3]
        if symbol.endswith("-INDEX"): symbol = symbol[:-6]
        return symbol

    @staticmethod
    def _map_timeframe(timeframe: str) -> str:
        mapping = {"DAY": "1d", "1H": "60m", "30MIN": "30m", "15MIN": "15m", "5MIN": "5m", "1MIN": "1m"}
        return mapping.get(timeframe.upper(), "1d")

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {OrderType.MARKET: "MARKET", OrderType.LIMIT: "LIMIT", OrderType.SL: "STOP_LOSS", OrderType.SL_MARKET: "STOP_LOSS_MARKET"}
        return mapping.get(order_type, "MARKET")

    def __repr__(self) -> str:
        return f"DhanBroker(client_id={self.client_id}, authenticated={self._authenticated})"
