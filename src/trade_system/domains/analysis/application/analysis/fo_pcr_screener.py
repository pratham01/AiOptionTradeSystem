"""
F&O Stock Put-Call Ratio (PCR) & Overbought / Oversold Screener.

Fetches live Option Chain data across the entire F&O stock universe to compute:
1. Put-Call Ratio (PCR) by Open Interest (OI)
2. Put-Call Ratio (PCR) by Volume
3. Max Pain Strike & ATM Implied Volatility (IV)
4. Institutional Positioning & Sentiment:
   - OVERBOUGHT: PCR >= 1.30 (Heavy Put Writing / Bullish Congestion)
   - OVERSOLD: PCR <= 0.55 (Heavy Call Writing / Bearish Congestion / Short Squeeze Candidate)
   - NEUTRAL: 0.55 < PCR < 1.30
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import pandas as pd

from trade_system.domains.market_data.infrastructure.data.fo_universe import (
    get_fo_universe,
    get_sector_mapping,
)
from trade_system.domains.trading.infrastructure.brokers.legacy import (
    FyersBrokerClient,
    FyersAuthService,
)
from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass
class StockPCRInfo:
    """Detailed PCR and Option Chain metrics for an individual F&O stock."""
    symbol: str
    clean_symbol: str
    sector: str
    spot_price: float
    pcr_oi: float
    pcr_volume: float
    total_call_oi: int
    total_put_oi: int
    total_call_volume: int
    total_put_volume: int
    max_pain_strike: float
    sentiment_state: str        # "EXTREME_OVERBOUGHT", "OVERBOUGHT", "NEUTRAL", "OVERSOLD", "EXTREME_OVERSOLD"
    contrarian_bias: str        # "BEARISH_REVERSAL_RISK" (if overbought), "SHORT_SQUEEZE_POTENTIAL" (if oversold), "NEUTRAL"
    nearest_expiry: str
    atm_strike: float = 0.0
    highest_ce_oi_strike: float = 0.0
    highest_pe_oi_strike: float = 0.0
    analysis_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "clean_symbol": self.clean_symbol,
            "sector": self.sector,
            "spot_price": round(self.spot_price, 2),
            "pcr_oi": round(self.pcr_oi, 2),
            "pcr_volume": round(self.pcr_volume, 2),
            "total_call_oi": self.total_call_oi,
            "total_put_oi": self.total_put_oi,
            "total_call_volume": self.total_call_volume,
            "total_put_volume": self.total_put_volume,
            "max_pain_strike": round(self.max_pain_strike, 2),
            "sentiment_state": self.sentiment_state,
            "contrarian_bias": self.contrarian_bias,
            "nearest_expiry": self.nearest_expiry,
            "atm_strike": round(self.atm_strike, 2),
            "highest_ce_oi_strike": round(self.highest_ce_oi_strike, 2),
            "highest_pe_oi_strike": round(self.highest_pe_oi_strike, 2),
            "analysis_narrative": self.analysis_narrative,
        }


class FOPCRScreener:
    """
    Screener to fetch Option Chains and analyze PCR across F&O Universe.
    Note: For individual equity stock options, institutional Call writing is structurally
    higher than Index options. Thus:
    - OVERBOUGHT: PCR >= 0.85 (Extreme >= 1.00)
    - OVERSOLD: PCR <= 0.55 (Extreme <= 0.45)
    """

    def __init__(
        self,
        broker: Optional[FyersBrokerClient] = None,
        overbought_threshold: float = 0.85,
        oversold_threshold: float = 0.55,
        extreme_overbought_threshold: float = 1.00,
        extreme_oversold_threshold: float = 0.45,
    ) -> None:
        self.broker = broker
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.extreme_overbought_threshold = extreme_overbought_threshold
        self.extreme_oversold_threshold = extreme_oversold_threshold
        self.sector_map = get_sector_mapping()

    def _ensure_broker(self) -> FyersBrokerClient:
        if self.broker is None or self.broker.fyers is None:
            settings = Settings.load()
            auth = FyersAuthService(settings)
            token = auth.get_valid_token()
            self.broker = FyersBrokerClient(
                client_id=settings.fyers.client_id,
                access_token=token,
                user_id=settings.fyers.user_id,
                authenticator=auth.authenticator,
            )
            self.broker.authenticate()
        return self.broker

    def fetch_stock_pcr(
        self,
        symbol: str,
        strikecount: int = 15,
        spot_price: Optional[float] = None,
    ) -> Optional[StockPCRInfo]:
        """
        Fetch live Option Chain for a stock symbol and calculate comprehensive PCR metrics.
        """
        try:
            broker = self._ensure_broker()
            clean_sym = symbol.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-INDEX", "")
            sector = self.sector_map.get(symbol, "GENERAL")

            data = {
                "symbol": symbol,
                "strikecount": strikecount,
            }
            res = broker.fyers.optionchain(data=data)
            if not isinstance(res, dict) or res.get("s") != "ok":
                LOGGER.debug("Option chain query returned non-ok for %s: %s", symbol, res)
                return None

            oc_data = res.get("data", {})
            options_chain = oc_data.get("optionsChain", [])
            if not options_chain:
                return None

            # Calculate Call and Put aggregates
            ce_oi = 0
            pe_oi = 0
            ce_vol = 0
            pe_vol = 0
            detected_spot = spot_price or 0.0
            expiry_str = ""

            ce_strikes_oi: Dict[float, int] = {}
            pe_strikes_oi: Dict[float, int] = {}

            for item in options_chain:
                opt_type = str(item.get("option_type", "")).upper()
                strike = float(item.get("strike_price", 0.0))
                oi = int(item.get("oi", 0))
                vol = int(item.get("volume", 0))

                # Capture spot price or underlying value
                if detected_spot == 0.0 and item.get("underlying_value"):
                    detected_spot = float(item.get("underlying_value"))

                if not expiry_str and item.get("expiry_date"):
                    expiry_str = str(item.get("expiry_date"))

                if opt_type == "CE":
                    ce_oi += oi
                    ce_vol += vol
                    ce_strikes_oi[strike] = ce_strikes_oi.get(strike, 0) + oi
                elif opt_type == "PE":
                    pe_oi += oi
                    pe_vol += vol
                    pe_strikes_oi[strike] = pe_strikes_oi.get(strike, 0) + oi

            if ce_oi == 0 and pe_oi == 0:
                return None

            # Compute PCR
            pcr_oi = pe_oi / ce_oi if ce_oi > 0 else 1.0
            pcr_vol = pe_vol / ce_vol if ce_vol > 0 else 1.0

            # Find Highest OI Strike (Walls)
            highest_ce_strike = max(ce_strikes_oi, key=ce_strikes_oi.get) if ce_strikes_oi else 0.0
            highest_pe_strike = max(pe_strikes_oi, key=pe_strikes_oi.get) if pe_strikes_oi else 0.0

            # Calculate Max Pain
            all_strikes = sorted(set(list(ce_strikes_oi.keys()) + list(pe_strikes_oi.keys())))
            max_pain_strike = 0.0
            min_loss = float("inf")

            for target_strike in all_strikes:
                total_loss = 0.0
                for s, oi_val in ce_strikes_oi.items():
                    if target_strike > s:
                        total_loss += (target_strike - s) * oi_val
                for s, oi_val in pe_strikes_oi.items():
                    if target_strike < s:
                        total_loss += (s - target_strike) * oi_val

                if total_loss < min_loss:
                    min_loss = total_loss
                    max_pain_strike = target_strike

            # Closest ATM strike & Spot fallback
            if detected_spot == 0.0 and all_strikes:
                detected_spot = all_strikes[len(all_strikes) // 2]
            atm_strike = min(all_strikes, key=lambda s: abs(s - detected_spot)) if (all_strikes and detected_spot > 0) else 0.0

            # Sentiment Classification
            if pcr_oi >= self.extreme_overbought_threshold:
                sentiment = "EXTREME_OVERBOUGHT"
                bias = "BEARISH_REVERSAL_RISK"
                narrative = f"PCR at extreme high ({pcr_oi:.2f}). Heavy Put writing indicates euphoric crowding; vulnerable to profit-booking."
            elif pcr_oi >= self.overbought_threshold:
                sentiment = "OVERBOUGHT"
                bias = "BEARISH_REVERSAL_RISK"
                narrative = f"PCR elevated ({pcr_oi:.2f}). Strong bullish put writing cushion; watch for resistance rejection."
            elif pcr_oi <= self.extreme_oversold_threshold:
                sentiment = "EXTREME_OVERSOLD"
                bias = "SHORT_SQUEEZE_POTENTIAL"
                narrative = f"PCR at extreme low ({pcr_oi:.2f}). Heavy Call writing indicates peak bearish consensus; prime short squeeze candidate."
            elif pcr_oi <= self.oversold_threshold:
                sentiment = "OVERSOLD"
                bias = "SHORT_SQUEEZE_POTENTIAL"
                narrative = f"PCR oversold ({pcr_oi:.2f}). Aggressive call writing ceiling; any upward spark can trigger short covering."
            else:
                sentiment = "NEUTRAL"
                bias = "NEUTRAL"
                narrative = f"PCR is balanced ({pcr_oi:.2f}) with normal positioning."

            return StockPCRInfo(
                symbol=symbol,
                clean_symbol=clean_sym,
                sector=sector,
                spot_price=detected_spot,
                pcr_oi=pcr_oi,
                pcr_volume=pcr_vol,
                total_call_oi=ce_oi,
                total_put_oi=pe_oi,
                total_call_volume=ce_vol,
                total_put_volume=pe_vol,
                max_pain_strike=max_pain_strike,
                sentiment_state=sentiment,
                contrarian_bias=bias,
                nearest_expiry=expiry_str,
                atm_strike=atm_strike,
                highest_ce_oi_strike=highest_ce_strike,
                highest_pe_oi_strike=highest_pe_strike,
                analysis_narrative=narrative,
            )
        except Exception as exc:
            LOGGER.error("FOPCRScreener error for %s: %s", symbol, exc)
            return None

    def scan_universe_pcr(
        self,
        symbols: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Scan full F&O universe or custom list to identify Overbought and Oversold stocks.
        """
        target_symbols = symbols or get_fo_universe()
        if max_symbols:
            target_symbols = target_symbols[:max_symbols]

        LOGGER.info("Scanning PCR for %d F&O stocks...", len(target_symbols))

        broker = self._ensure_broker()
        # Pre-fetch quotes in batch (50 at a time)
        quotes_map: Dict[str, float] = {}
        try:
            for chunk_start in range(0, len(target_symbols), 50):
                chunk = target_symbols[chunk_start:chunk_start + 50]
                q_res = broker.get_quotes(chunk)
                for s, q_item in q_res.items():
                    if isinstance(q_item, dict):
                        quotes_map[s] = float(q_item.get("lp", 0.0) or q_item.get("ltp", 0.0))
                    elif hasattr(q_item, "ltp"):
                        quotes_map[s] = float(q_item.ltp)
        except Exception as q_err:
            LOGGER.debug("Failed batch quote fetch: %s", q_err)

        results: List[StockPCRInfo] = []
        for i, sym in enumerate(target_symbols, 1):
            spot = quotes_map.get(sym, 0.0)
            info = self.fetch_stock_pcr(sym, spot_price=spot)
            if info:
                results.append(info)
            if i % 30 == 0 or i == len(target_symbols):
                LOGGER.info("PCR scan progress: %d/%d processed...", i, len(target_symbols))

        overbought = [s for s in results if "OVERBOUGHT" in s.sentiment_state]
        oversold = [s for s in results if "OVERSOLD" in s.sentiment_state]
        neutral = [s for s in results if s.sentiment_state == "NEUTRAL"]

        # Sort Overbought by highest PCR descending
        overbought.sort(key=lambda s: s.pcr_oi, reverse=True)
        # Sort Oversold by lowest PCR ascending
        oversold.sort(key=lambda s: s.pcr_oi, reverse=False)
        # Sort All by PCR descending
        results.sort(key=lambda s: s.pcr_oi, reverse=True)

        return {
            "overbought": overbought,
            "oversold": oversold,
            "neutral": neutral,
            "all": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
