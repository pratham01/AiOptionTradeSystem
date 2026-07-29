"""
SmartOIAnalyzer - Analysis and filtering engine for Nifty 50 Smart OI.
Filters out noise to show signal strikes, detects institutional buildup,
and compares with price action to find confluences or divergences.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from trade_system.domains.market_data.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)

class SmartOIAnalyzer:
    """
    Analyzes Option Chain data, filters noise to reveal institutional activity,
    computes buildup signals, and flags confluences/divergences with price action.
    """

    def __init__(self, symbol: str = "NSE:NIFTY50-INDEX"):
        self.symbol = symbol
        # Map simple symbol name to database underlying symbol
        if symbol in ["NIFTY", "NIFTY50", "NSE:NIFTY50-INDEX"]:
            self.db_symbol = "NSE:NIFTY50-INDEX"
            self.strike_step = 50
        elif symbol in ["BANKNIFTY", "NIFTYBANK", "NSE:NIFTYBANK-INDEX"]:
            self.db_symbol = "NSE:NIFTYBANK-INDEX"
            self.strike_step = 100
        elif symbol in ["SENSEX", "BSE:SENSEX-INDEX"]:
            self.db_symbol = "BSE:SENSEX-INDEX"
            self.strike_step = 100
        else:
            self.db_symbol = symbol
            self.strike_step = 50  # Default fallback

    def fetch_latest_snapshots(self, date_str: str, limit: int = 2) -> List[Tuple[datetime, pd.DataFrame]]:
        """
        Fetch the latest N snapshots from the database for a specific date.
        Returns a list of tuples containing (timestamp, DataFrame).
        """
        engine = get_engine()
        snapshots = []
        try:
            with engine.connect() as conn:
                # Get unique timestamps for this symbol and date
                ts_query = text("""
                    SELECT DISTINCT timestamp 
                    FROM option_chain_data 
                    WHERE underlying_symbol = :symbol 
                      AND date(timestamp) = :date_str
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                ts_df = pd.read_sql(ts_query, conn, params={"symbol": self.db_symbol, "date_str": date_str, "limit": limit})
                if ts_df.empty:
                    return []

                for ts in ts_df["timestamp"]:
                    data_query = text("""
                        SELECT * 
                        FROM option_chain_data 
                        WHERE underlying_symbol = :symbol 
                          AND timestamp = :ts
                    """)
                    df = pd.read_sql(data_query, conn, params={"symbol": self.db_symbol, "ts": ts})
                    # Parse timestamp if string
                    ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                    snapshots.append((ts_dt, df))
        except Exception as e:
            LOGGER.error(f"Failed to fetch snapshots for {self.symbol} on {date_str}: {e}")
        return snapshots

    def analyze_smart_oi(
        self, 
        current_df: pd.DataFrame, 
        prev_df: pd.DataFrame | None = None, 
        spot_price: float | None = None
    ) -> Dict[str, Any]:
        """
        Run the Smart OI analysis on a current option chain DataFrame.
        Filters noise and computes signal strikes and overall positioning.
        """
        if current_df.empty:
            return {"signal": "NEUTRAL", "signal_strikes": [], "summary": {}}

        # Estimate spot price if not provided
        if spot_price is None:
            # Try to get from ltp of deep in-the-money options or default to strike mean
            pe_strikes = current_df[current_df["option_type"] == "PE"]
            ce_strikes = current_df[current_df["option_type"] == "CE"]
            if not pe_strikes.empty and not ce_strikes.empty:
                # Find strike where PE LTP is close to CE LTP, or approximate from greeks if delta is near 0.5
                spot_price = current_df["strike"].mean()  # fallback

        # Round ATM strike
        atm_strike = round(spot_price / self.strike_step) * self.strike_step if spot_price else current_df["strike"].mean()

        # Merge current with previous to compute difference in OI, LTP, and volume
        merged_df = current_df.copy()
        if prev_df is not None and not prev_df.empty:
            # We index by strike and option_type to align
            prev_aligned = prev_df.set_index(["strike", "option_type"])
            curr_aligned = merged_df.set_index(["strike", "option_type"])
            
            # Compute difference
            curr_aligned["oi_change"] = curr_aligned["oi"] - prev_aligned["oi"]
            curr_aligned["volume_change"] = curr_aligned["volume"] - prev_aligned["volume"]
            curr_aligned["ltp_change"] = curr_aligned["ltp"] - prev_aligned["ltp"]
            curr_aligned["oi_change_pct"] = (curr_aligned["oi_change"] / prev_aligned["oi"].replace(0, np.nan)) * 100
            curr_aligned["ltp_change_pct"] = (curr_aligned["ltp_change"] / prev_aligned["ltp"].replace(0, np.nan)) * 100
            
            # Reset index
            merged_df = curr_aligned.reset_index()
        else:
            # Default mock changes
            merged_df["oi_change"] = 0
            merged_df["volume_change"] = merged_df["volume"]
            merged_df["ltp_change"] = 0.0
            merged_df["oi_change_pct"] = 0.0
            merged_df["ltp_change_pct"] = 0.0

        # Replace NaNs and ensure types
        merged_df["oi_change"] = merged_df["oi_change"].fillna(0).astype(int)
        merged_df["volume_change"] = merged_df["volume_change"].fillna(0).astype(int)
        merged_df["ltp_change"] = merged_df["ltp_change"].fillna(0.0)
        merged_df["oi_change_pct"] = merged_df["oi_change_pct"].fillna(0.0)
        merged_df["ltp_change_pct"] = merged_df["ltp_change_pct"].fillna(0.0)

        # ── FILTERING RULES ──────────────────────────────────────────────────
        max_oi = merged_df["oi"].max() if not merged_df.empty else 1
        
        # 1. Deep OTM filter: within ±7 strikes of ATM
        merged_df["is_near_atm"] = merged_df["strike"].apply(
            lambda s: abs(s - atm_strike) <= 7 * self.strike_step
        )
        
        # 2. Small OI / Volume filter (retail lottery tickets)
        merged_df["has_liquidity"] = (merged_df["oi"] >= 0.015 * max_oi) & (merged_df["volume"] > 0)
        
        # 3. Stale positions filter: must have change in OI or LTP
        # (If prev_df is provided, we check if there's any intraday update)
        merged_df["is_stale"] = False
        if prev_df is not None and not prev_df.empty:
            merged_df["is_stale"] = (merged_df["oi_change"] == 0) & (merged_df["ltp_change"] == 0.0)
            
        # 4. One-off Spike Filter: high interval volume but negligible interval OI change
        # This implies pure scalping/churning rather than institutional build-up
        merged_df["is_one_off"] = False
        if prev_df is not None and not prev_df.empty:
            merged_df["is_one_off"] = (merged_df["volume_change"] > 500) & (merged_df["oi_change"].abs() / merged_df["volume_change"] < 0.01)

        # Final keep mask
        merged_df["keep"] = (
            merged_df["is_near_atm"] & 
            merged_df["has_liquidity"] & 
            (~merged_df["is_stale"]) & 
            (~merged_df["is_one_off"])
        )

        signal_df = merged_df[merged_df["keep"]].copy()

        # ── BUILDUP CLASSIFICATION ───────────────────────────────────────────
        buildup_types = []
        buildup_actions = []
        
        for _, row in signal_df.iterrows():
            opt_type = row["option_type"]
            oi_chg = row["oi_change"]
            ltp_chg = row["ltp_change"]
            
            # Simple threshold for direction
            is_oi_up = oi_chg >= 0
            is_ltp_up = ltp_chg > 0
            
            if is_oi_up and is_ltp_up:
                buildup = "Long Buildup"
                action = "Call Buying" if opt_type == "CE" else "Put Buying"
            elif is_oi_up and not is_ltp_up:
                buildup = "Short Buildup"
                action = "Call Writing" if opt_type == "CE" else "Put Writing"
            elif not is_oi_up and not is_ltp_up:
                buildup = "Long Unwinding"
                action = "Call Unwinding" if opt_type == "CE" else "Put Unwinding"
            else: # not is_oi_up and is_ltp_up:
                buildup = "Short Covering"
                action = "Call Covering" if opt_type == "CE" else "Put Covering"
                
            buildup_types.append(buildup)
            buildup_actions.append(action)

        signal_df["buildup"] = buildup_types
        signal_df["action"] = buildup_actions

        # ── OVERALL STATE SIGNALS ────────────────────────────────────────────
        # Put writing (PE Short Buildup) is Bullish. Call writing (CE Short Buildup) is Bearish.
        # Call buying (CE Long Buildup) is Bullish. Put buying (PE Long Buildup) is Bearish.
        bullish_oi = 0
        bearish_oi = 0
        unwind_oi = 0

        for _, row in signal_df.iterrows():
            act = row["action"]
            chg = abs(row["oi_change"])
            
            if act in ["Put Writing", "Call Buying", "Call Covering"]:
                bullish_oi += chg
            elif act in ["Call Writing", "Put Buying", "Put Covering"]:
                bearish_oi += chg
            elif act in ["Call Unwinding", "Put Unwinding"]:
                unwind_oi += chg

        total_chg = bullish_oi + bearish_oi + unwind_oi
        
        if total_chg == 0:
            overall_signal = "neutral"
        elif unwind_oi > 0.6 * total_chg:
            overall_signal = "unwind"
        elif bullish_oi > bearish_oi * 1.25:
            overall_signal = "long buildup" # Bullish positioning
        elif bearish_oi > bullish_oi * 1.25:
            overall_signal = "short buildup" # Bearish positioning
        else:
            overall_signal = "neutral"

        # Prepare summary dict
        summary = {
            "spot_price": spot_price,
            "atm_strike": atm_strike,
            "bullish_oi_flow": int(bullish_oi),
            "bearish_oi_flow": int(bearish_oi),
            "unwind_oi_flow": int(unwind_oi),
            "total_signal_strikes": len(signal_df),
            "total_raw_strikes": len(current_df),
            "filtered_out_count": len(current_df) - len(signal_df),
        }

        # Keep only key columns for clean representation
        signal_strikes = []
        if not signal_df.empty:
            signal_df_sorted = signal_df.sort_values("strike")
            for _, row in signal_df_sorted.iterrows():
                signal_strikes.append({
                    "strike": float(row["strike"]),
                    "option_type": row["option_type"],
                    "symbol": row["symbol"],
                    "ltp": float(row["ltp"]),
                    "oi": int(row["oi"]),
                    "oi_change": int(row["oi_change"]),
                    "oi_change_pct": float(row["oi_change_pct"]),
                    "volume": int(row["volume"]),
                    "buildup": row["buildup"],
                    "action": row["action"],
                })

        return {
            "signal": overall_signal,
            "signal_strikes": signal_strikes,
            "summary": summary,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def detect_confluence_divergence(
        self, 
        oi_signal: str, 
        ohlcv_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compares the Smart OI signal with the price action trend and volume expansion.
        Classifies current setup as BULLISH CONFLUENCE, BEARISH CONFLUENCE, DIVERGENCE, or NEUTRAL.
        """
        if ohlcv_df.empty or len(ohlcv_df) < 5:
            return {
                "status": "NEUTRAL",
                "narrative": "Insufficient price data to perform confluence analysis.",
                "price_trend": "SIDEWAYS",
                "volume_expansion": False,
                "key_level": None
            }

        # Sort ohlcv by timestamp ascending
        ohlcv = ohlcv_df.sort_values("timestamp").copy()
        
        # Calculate moving averages/technical levels
        ohlcv["close_sma20"] = ohlcv["close"].rolling(window=min(20, len(ohlcv))).mean()
        ohlcv["vol_sma20"] = ohlcv["volume"].rolling(window=min(20, len(ohlcv))).mean()
        
        # VWAP calculation
        ohlcv["typical_price"] = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        ohlcv["tp_vol"] = ohlcv["typical_price"] * ohlcv["volume"]
        ohlcv["cum_vol"] = ohlcv["volume"].cumsum()
        ohlcv["cum_tp_vol"] = ohlcv["tp_vol"].cumsum()
        ohlcv["vwap"] = ohlcv["cum_tp_vol"] / ohlcv["cum_vol"].replace(0, 1)

        latest = ohlcv.iloc[-1]
        prev = ohlcv.iloc[-2]
        
        # Key Technical Levels
        recent_bars = ohlcv.tail(15)
        highest_high = recent_bars["high"].max()
        lowest_low = recent_bars["low"].min()
        
        current_price = latest["close"]
        vwap = latest["vwap"]
        
        # Trend classification
        # Simple definition: price breakout is above highest high or VWAP with volume
        is_above_vwap = current_price > vwap
        is_above_prev_high = current_price >= prev["high"]
        
        price_trend = "UP" if is_above_vwap else ("DOWN" if current_price < vwap else "SIDEWAYS")
        
        # Volume expansion: volume is 1.5x the SMA
        current_vol = latest["volume"]
        vol_sma = latest["vol_sma20"]
        # Fallback if vol_sma is 0 (like indices in raw OHLCV)
        volume_expansion = current_vol > vol_sma * 1.5 if vol_sma > 0 else False
        
        # We also look at index options volume from the option chain if index volume itself is 0
        # Let's classify confluence & divergence
        status = "NEUTRAL"
        narrative = "No clear confluence or divergence between Smart OI and Price Action."
        key_level = None

        if oi_signal == "long buildup": # Bullish Smart OI
            if current_price > vwap and is_above_prev_high:
                status = "BULLISH CONFLUENCE"
                key_level = vwap
                narrative = (
                    "🔥 HIGH-CONVICTION LONG SETUP: Smart OI shows put writing accumulation (bullish positioning) "
                    f"concurring with a price breakout above key level ₹{key_level:.2f}. Enter with normal size, "
                    f"stops below ₹{key_level:.2f}."
                )
            elif current_price < vwap:
                status = "BULLISH DIVERGENCE (FALLING PRICE / BULLISH OI)"
                narrative = (
                    "⚠️ BULLISH DIVERGENCE: Price is falling below VWAP, but Smart OI shows bullish put writing accumulation. "
                    "This suggests a potential false decline or bear trap. Watch for a reversal pattern."
                )
        elif oi_signal == "short buildup": # Bearish Smart OI
            if current_price < vwap and current_price <= prev["low"]:
                status = "BEARISH CONFLUENCE"
                key_level = vwap
                narrative = (
                    "📉 STRONG SHORT SETUP: Smart OI flags call writing (bearish positioning) concurring with a price breakdown "
                    f"below support at ₹{key_level:.2f}. Set stops above the resistance, target lower Smart OI levels."
                )
            elif current_price > vwap:
                status = "BEARISH DIVERGENCE (RISING PRICE / BEARISH OI)"
                narrative = (
                    "⚠️ BEARISH DIVERGENCE: Price is rising above VWAP, but Smart OI flags bearish call writing positioning. "
                    "This suggests a false rally. Exercise caution and look for signs of reversal."
                )
        elif oi_signal == "unwind":
            status = "UNWIND"
            narrative = "Smart OI shows institutional position unwinding. Volume is declining; expect sideways or range-bound behavior."
            
        return {
            "status": status,
            "narrative": narrative,
            "price_trend": price_trend,
            "volume_expansion": volume_expansion,
            "key_level": float(key_level) if key_level is not None else None,
            "vwap": float(vwap),
            "highest_high": float(highest_high),
            "lowest_low": float(lowest_low)
        }
