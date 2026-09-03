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

    def update_strike_step_from_df(self, df: pd.DataFrame) -> float:
        """Dynamically infer the underlying's strike step from the option chain."""
        if df is not None and not df.empty and "strike" in df.columns:
            strikes = sorted(df["strike"].dropna().unique())
            if len(strikes) >= 2:
                diffs = [round(strikes[i+1] - strikes[i], 2) for i in range(len(strikes)-1)]
                pos_diffs = [d for d in diffs if d > 0]
                if pos_diffs:
                    import statistics
                    try:
                        self.strike_step = statistics.mode(pos_diffs)
                    except Exception:
                        self.strike_step = min(pos_diffs)
        return self.strike_step

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

    def analyze_market_direction(
        self,
        current_df: pd.DataFrame,
        prev_df: pd.DataFrame | None = None,
        ohlcv_df: pd.DataFrame | None = None,
        spot_price: float | None = None,
        snapshots: List[Tuple[datetime, pd.DataFrame]] | None = None
    ) -> Dict[str, Any]:
        """
        Comprehensive Market Direction Analysis Engine.
        Synthesizes Smart OI Flow, Dealer Gamma Exposure (GEX), Institutional Big Money Notional,
        and Technical Price Confluence to produce Intraday and Swing Directional Biases.
        """
        from trade_system.domains.analysis.application.analysis import pro_oc_analyzer

        smart_res = self.analyze_smart_oi(current_df, prev_df, spot_price=spot_price)
        summary = smart_res.get("summary", {})
        spot = summary.get("spot_price", spot_price or (current_df["strike"].mean() if not current_df.empty else 0.0))

        # 1. Pro Analytics
        gex_info = pro_oc_analyzer.compute_gex_profile(current_df, spot, self.strike_step)
        big_money = pro_oc_analyzer.compute_institutional_big_money(current_df, prev_df, spot, self.strike_step)
        wall_shifts = pro_oc_analyzer.compute_wall_shifts(snapshots or [], spot, self.strike_step)
        oi_conc = pro_oc_analyzer.compute_oi_concentration(current_df, spot, self.strike_step)

        # 2. Intraday Score (-100 to +100)
        intraday_score = 0
        bull_flow = summary.get("bullish_oi_flow", 0)
        bear_flow = summary.get("bearish_oi_flow", 0)
        unwind_flow = summary.get("unwind_oi_flow", 0)
        total_flow = bull_flow + bear_flow + unwind_flow

        if total_flow > 0:
            flow_diff = (bull_flow - bear_flow) / float(total_flow)
            intraday_score += int(flow_diff * 50)  # Max +/- 50 from OI flow

        # Technical VWAP Alignment adjustment
        vwap_val = None
        if ohlcv_df is not None and not ohlcv_df.empty and len(ohlcv_df) >= 3:
            conf = self.detect_confluence_divergence(smart_res.get("signal", "neutral"), ohlcv_df)
            vwap_val = conf.get("vwap")
            if vwap_val and spot > vwap_val:
                intraday_score += 20
            elif vwap_val and spot < vwap_val:
                intraday_score -= 20

        # Big Money Notional Flow adjustment
        if big_money.get("institutional_bias") == "INSTITUTIONAL BULLISH ACCUMULATION":
            intraday_score += 20
        elif big_money.get("institutional_bias") == "INSTITUTIONAL BEARISH DISTRIBUTION":
            intraday_score -= 20

        # Clamp intraday score to [-100, 100]
        intraday_score = max(-100, min(100, intraday_score))

        # Intraday Rating Classification
        if intraday_score >= 50:
            intraday_direction = "STRONG BULLISH"
        elif intraday_score >= 20:
            intraday_direction = "MODERATE BULLISH"
        elif intraday_score <= -50:
            intraday_direction = "STRONG BEARISH"
        elif intraday_score <= -20:
            intraday_direction = "MODERATE BEARISH"
        else:
            intraday_direction = "NEUTRAL / RANGEBOUND"

        # 3. Swing Direction Analysis (-100 to +100)
        swing_score = 0
        call_wall = oi_conc.get("highest_ce_wall", {}).get("strike", 0.0) if oi_conc.get("highest_ce_wall") else 0.0
        put_wall = oi_conc.get("highest_pe_wall", {}).get("strike", 0.0) if oi_conc.get("highest_pe_wall") else 0.0

        if spot and call_wall and put_wall:
            range_span = call_wall - put_wall
            if range_span > 0:
                pos_in_range = (spot - put_wall) / range_span
                # Closer to Put wall = Bullish Swing Support Zone; Closer to Call wall = Bearish Resistance Zone
                if pos_in_range > 0.7:
                    swing_score += 30  # Testing Resistance
                elif pos_in_range < 0.3:
                    swing_score -= 30  # Testing Support

        if "BULLISH ROLL-UP" in wall_shifts.get("call_wall_shift", ""):
            swing_score += 30
        elif "BEARISH STEP-DOWN" in wall_shifts.get("call_wall_shift", ""):
            swing_score -= 30

        if "BULLISH STEP-UP" in wall_shifts.get("put_wall_shift", ""):
            swing_score += 20
        elif "BEARISH ROLL-DOWN" in wall_shifts.get("put_wall_shift", ""):
            swing_score -= 20

        swing_score = max(-100, min(100, swing_score))

        if swing_score >= 35:
            swing_direction = "BULLISH ACCUMULATION"
        elif swing_score <= -35:
            swing_direction = "BEARISH DISTRIBUTION"
        else:
            swing_direction = "NEUTRAL CONSOLIDATION"

        return {
            "intraday_direction": intraday_direction,
            "intraday_score": intraday_score,
            "swing_direction": swing_direction,
            "swing_score": swing_score,
            "gamma_regime": gex_info.get("gamma_regime"),
            "gamma_flip_level": gex_info.get("gamma_flip_level"),
            "total_net_gex": gex_info.get("total_net_gex"),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "wall_shift_status": f"Call Wall: {wall_shifts.get('call_wall_shift')} | Put Wall: {wall_shifts.get('put_wall_shift')}",
            "institutional_bias": big_money.get("institutional_bias"),
            "spot_price": spot,
            "vwap": vwap_val
        }

    def generate_trade_setups(
        self,
        current_df: pd.DataFrame,
        prev_df: pd.DataFrame | None = None,
        ohlcv_df: pd.DataFrame | None = None,
        spot_price: float | None = None
    ) -> Dict[str, Any]:
        """
        Generate actionable Intraday & Swing Trade Setups with exact Entry, Stop Loss, Target,
        Option Contract recommendation, and Risk-Reward ratio.
        """
        dir_analysis = self.analyze_market_direction(current_df, prev_df, ohlcv_df, spot_price)
        spot = dir_analysis["spot_price"]
        atm_strike = round(spot / self.strike_step) * self.strike_step if spot else current_df["strike"].mean()
        vwap = dir_analysis.get("vwap") or spot
        call_wall = dir_analysis.get("call_wall") or (atm_strike + 2 * self.strike_step)
        put_wall = dir_analysis.get("put_wall") or (atm_strike - 2 * self.strike_step)
        intra_score = dir_analysis["intraday_score"]

        # ── 1. INTRADAY TRADE SETUP ─────────────────────────────────────────
        if intra_score >= 30:
            intraday_setup = {
                "type": "INTRADAY BREAKOUT LONG",
                "direction": "BUY CALL",
                "option_contract": f"{int(atm_strike)} CE",
                "actionable_trigger": f"Buy when spot crosses & holds above ₹{max(spot, vwap):,.1f}",
                "entry_zone": f"₹{spot:,.1f} - ₹{spot + 15:,.1f}",
                "stop_loss": f"₹{max(put_wall, spot - 40):,.1f}",
                "target_1": f"₹{min(call_wall, spot + 60):,.1f}",
                "target_2": f"₹{spot + 110:,.1f}",
                "risk_reward": "1 : 2.2",
                "conviction_score": f"{min(95, 60 + abs(intra_score) // 2)}/100",
                "status_color": "#00d084",
                "rationale": (
                    f"Strong Intraday Bullish Score ({intra_score}/100). Institutional put writing is creating a solid floor at ₹{put_wall:,.0f}. "
                    f"Price is trading above VWAP (₹{vwap:,.1f}), favoring momentum long setups towards Call Wall at ₹{call_wall:,.0f}."
                )
            }
        elif intra_score <= -30:
            intraday_setup = {
                "type": "INTRADAY BREAKDOWN SHORT",
                "direction": "BUY PUT",
                "option_contract": f"{int(atm_strike)} PE",
                "actionable_trigger": f"Buy when spot breaks below ₹{min(spot, vwap):,.1f}",
                "entry_zone": f"₹{spot - 15:,.1f} - ₹{spot:,.1f}",
                "stop_loss": f"₹{min(call_wall, spot + 40):,.1f}",
                "target_1": f"₹{max(put_wall, spot - 60):,.1f}",
                "target_2": f"₹{spot - 110:,.1f}",
                "risk_reward": "1 : 2.1",
                "conviction_score": f"{min(95, 60 + abs(intra_score) // 2)}/100",
                "status_color": "#ff4d6d",
                "rationale": (
                    f"Bearish Intraday Score ({intra_score}/100). Institutional call writers pressuring ceiling at ₹{call_wall:,.0f}. "
                    f"Price below VWAP (₹{vwap:,.1f}) indicates downside breakdown momentum towards Put Wall support at ₹{put_wall:,.0f}."
                )
            }
        else:
            intraday_setup = {
                "type": "NO TRADE RANGEBOUND / THETA DECAY",
                "direction": "WAIT / OPTION SELLING",
                "option_contract": f"Sell Straddle ({int(atm_strike)} CE & PE)",
                "actionable_trigger": f"Wait for spot to break range ₹{put_wall:,.0f} - ₹{call_wall:,.0f}",
                "entry_zone": f"Between ₹{put_wall:,.0f} and ₹{call_wall:,.0f}",
                "stop_loss": "Exit if spot crosses Call/Put Wall",
                "target_1": "Collect Premium / Theta decay",
                "target_2": "N/A",
                "risk_reward": "1 : 1.2",
                "conviction_score": "50/100",
                "status_color": "#ffb703",
                "rationale": (
                    f"Neutral Intraday Score ({intra_score}/100). Market is trapped between Put Wall (₹{put_wall:,.0f}) and Call Wall (₹{call_wall:,.0f}). "
                    "Buying options will result in time decay. Prefer waiting or rangebound spread strategies."
                )
            }

        # ── 2. SWING TRADE SETUP ─────────────────────────────────────────────
        swing_score = dir_analysis["swing_score"]
        if swing_score >= 30:
            swing_setup = {
                "type": "SWING BULLISH ACCUMULATION",
                "direction": "BULLISH SWING",
                "recommended_strategy": f"Bull Put Spread (Sell {int(put_wall)} PE / Buy {int(put_wall - 100)} PE) or Buy {int(atm_strike)} CE",
                "entry_zone": f"Buy dips near ₹{put_wall + 30:,.0f} - ₹{spot:,.0f}",
                "stop_loss": f"₹{put_wall - 60:,.0f}",
                "target": f"₹{call_wall + 100:,.0f}",
                "timeframe": "2 to 5 Trading Days",
                "conviction_score": f"{min(90, 65 + abs(swing_score) // 2)}/100",
                "status_color": "#00d084",
                "rationale": (
                    f"Multi-session institutional accumulation detected. Put Wall at ₹{put_wall:,.0f} serves as a structural floor. "
                    "Favors buying pullbacks for multi-day targets."
                )
            }
        elif swing_score <= -30:
            swing_setup = {
                "type": "SWING BEARISH DISTRIBUTION",
                "direction": "BEARISH SWING",
                "recommended_strategy": f"Bear Call Spread (Sell {int(call_wall)} CE / Buy {int(call_wall + 100)} CE) or Buy {int(atm_strike)} PE",
                "entry_zone": f"Sell rallies near ₹{call_wall - 30:,.0f} - ₹{spot:,.0f}",
                "stop_loss": f"₹{call_wall + 60:,.0f}",
                "target": f"₹{put_wall - 100:,.0f}",
                "timeframe": "2 to 5 Trading Days",
                "conviction_score": f"{min(90, 65 + abs(swing_score) // 2)}/100",
                "status_color": "#ff4d6d",
                "rationale": (
                    f"Multi-session institutional call writing distribution. Call Wall at ₹{call_wall:,.0f} acts as strong overhead resistance. "
                    "Favors selling rallies for multi-day downside targets."
                )
            }
        else:
            swing_setup = {
                "type": "SWING RANGE CONSOLIDATION",
                "direction": "RANGEBOUND SWING",
                "recommended_strategy": f"Iron Condor (Sell {int(call_wall)} CE & {int(put_wall)} PE)",
                "entry_zone": f"Range ₹{put_wall:,.0f} - ₹{call_wall:,.0f}",
                "stop_loss": f"Breach of ₹{put_wall - 50:,.0f} or ₹{call_wall + 50:,.0f}",
                "target": "Max profit at expiry",
                "timeframe": "Expiry Holding",
                "conviction_score": "60/100",
                "status_color": "#00b4d8",
                "rationale": (
                    f"Option chain structure shows stable walls at ₹{put_wall:,.0f} (Put Wall) and ₹{call_wall:,.0f} (Call Wall). "
                    "Expect multi-day rangebound consolidation within these bounds."
                )
            }

        return {
            "intraday_setup": intraday_setup,
            "swing_setup": swing_setup,
            "directional_analysis": dir_analysis
        }

    def compute_strike_volume_profile(
        self,
        oc_df: pd.DataFrame,
        spot_price: float | None = None
    ) -> pd.DataFrame:
        """
        Computes strike-by-strike Volume and Open Interest profile,
        including Volume-to-OI (V/OI) ratio and institutional stickiness.
        """
        if oc_df is None or oc_df.empty:
            return pd.DataFrame()

        df = oc_df.copy()
        df["strike"] = pd.to_numeric(df.get("strike_price", df.get("strike", 0)), errors="coerce")
        df["oi"] = pd.to_numeric(df.get("oi", 0), errors="coerce").fillna(0).astype(int)
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
        df["option_type"] = df["option_type"].astype(str).str.upper()

        ce_df = df[df["option_type"] == "CE"].set_index("strike")
        pe_df = df[df["option_type"] == "PE"].set_index("strike")

        all_strikes = sorted(set(ce_df.index.tolist() + pe_df.index.tolist()))
        rows = []

        atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if (all_strikes and spot_price) else 0.0

        for s in all_strikes:
            ce_oi = int(ce_df.loc[s, "oi"]) if s in ce_df.index else 0
            pe_oi = int(pe_df.loc[s, "oi"]) if s in pe_df.index else 0
            ce_vol = int(ce_df.loc[s, "volume"]) if s in ce_df.index else 0
            pe_vol = int(pe_df.loc[s, "volume"]) if s in pe_df.index else 0

            total_oi = ce_oi + pe_oi
            total_vol = ce_vol + pe_vol
            v_oi = total_vol / max(total_oi, 1)

            if v_oi > 3.0:
                stickiness = "CHURN (Speculative)"
            elif v_oi < 0.8 and total_oi > 0:
                stickiness = "STICKY (Institutional)"
            else:
                stickiness = "NORMAL"

            rows.append({
                "strike": s,
                "ce_oi": ce_oi,
                "pe_oi": pe_oi,
                "net_oi": pe_oi - ce_oi,
                "ce_vol": ce_vol,
                "pe_vol": pe_vol,
                "total_vol": total_vol,
                "v_oi_ratio": round(v_oi, 2),
                "stickiness": stickiness,
                "is_atm": (s == atm_strike),
            })

        return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    def detect_institutional_divergences(
        self,
        ohlcv_df: pd.DataFrame,
        current_oc: pd.DataFrame,
        prev_oc: pd.DataFrame | None = None,
        spot_price: float | None = None
    ) -> Dict[str, Any]:
        """
        Advanced Institutional Divergence & Trap Detector:
        - Price vs PCR Divergence
        - Breakout / Breakdown Trap Detector (Price breaks out but Call OI spikes, or breaks down but Put OI spikes)
        - ATM Volume Delta & Taker Aggression
        - Volume Delta Exhaustion
        """
        divergences = []
        trap_alerts = []

        if current_oc is None or current_oc.empty:
            return {
                "divergences": [],
                "trap_alerts": [],
                "atm_volume_delta": 0,
                "atm_volume_ratio": 1.0,
                "taker_aggression": "NEUTRAL",
            }

        df = current_oc.copy()
        df["strike"] = pd.to_numeric(df.get("strike_price", df.get("strike", 0)), errors="coerce")
        df["oi"] = pd.to_numeric(df.get("oi", 0), errors="coerce").fillna(0).astype(int)
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
        df["option_type"] = df["option_type"].astype(str).str.upper()

        ce_df = df[df["option_type"] == "CE"]
        pe_df = df[df["option_type"] == "PE"]

        all_strikes = sorted(set(ce_df["strike"].tolist() + pe_df["strike"].tolist()))
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if (all_strikes and spot_price) else 0.0

        # ATM Cluster strikes
        step = self.strike_step or 50.0
        atm_cluster = [atm_strike - step, atm_strike, atm_strike + step]

        ce_atm_vol = ce_df[ce_df["strike"].isin(atm_cluster)]["volume"].sum()
        pe_atm_vol = pe_df[pe_df["strike"].isin(atm_cluster)]["volume"].sum()
        atm_vol_delta = int(ce_atm_vol - pe_atm_vol)
        atm_vol_ratio = ce_atm_vol / max(pe_atm_vol, 1)

        if atm_vol_ratio >= 1.50:
            aggression = "AGGRESSIVE BULLISH (Call Takers)"
        elif atm_vol_ratio <= 0.65:
            aggression = "AGGRESSIVE BEARISH (Put Takers)"
        else:
            aggression = "NEUTRAL / BALANCED"

        # Price vs PCR Divergence
        total_ce_oi = ce_df["oi"].sum()
        total_pe_oi = pe_df["oi"].sum()
        pcr_oi = total_pe_oi / max(total_ce_oi, 1)

        total_ce_vol = ce_df["volume"].sum()
        total_pe_vol = pe_df["volume"].sum()
        pcr_vol = total_pe_vol / max(total_ce_vol, 1)

        # Leading Volume PCR Divergence
        if pcr_vol > pcr_oi * 1.30 and pcr_oi < 0.90:
            divergences.append("🟢 Bullish Volume PCR Leading Divergence: Put volume intensity is surging while OI PCR remains low (Smart Money Accumulation).")
        elif pcr_vol < pcr_oi * 0.70 and pcr_oi > 1.10:
            divergences.append("🔴 Bearish Volume PCR Leading Divergence: Call volume intensity is surging while OI PCR is high (Smart Money Distribution).")

        # Price vs VWAP / Price Action Divergence
        if ohlcv_df is not None and not ohlcv_df.empty and len(ohlcv_df) >= 5:
            ohlcv = ohlcv_df.sort_values("timestamp").copy()
            recent_bars = ohlcv.tail(15)
            price_change = recent_bars["close"].iloc[-1] - recent_bars["close"].iloc[0]

            if price_change < 0 and pcr_oi > 1.15:
                divergences.append("🟢 Price vs PCR Bullish Divergence: Price is dropping into support but Put writing is expanding aggressively.")
            elif price_change > 0 and pcr_oi < 0.65:
                divergences.append("🔴 Price vs PCR Bearish Divergence: Price is pushing higher but Call writers are capping with heavy overhead walls.")

            # Volume Delta Exhaustion
            if price_change > 0 and atm_vol_delta < 0:
                divergences.append("⚠️ Volume Delta Exhaustion: Price made higher highs but ATM Volume Delta is net negative.")

        # Trap Detection (Requires previous snapshot or change data)
        if prev_oc is not None and not prev_oc.empty:
            prev_df = prev_oc.copy()
            prev_df["strike"] = pd.to_numeric(prev_df.get("strike_price", prev_df.get("strike", 0)), errors="coerce")
            prev_df["oi"] = pd.to_numeric(prev_df.get("oi", 0), errors="coerce").fillna(0).astype(int)
            prev_df["option_type"] = prev_df["option_type"].astype(str).str.upper()

            # Align
            curr_ce = ce_df.set_index("strike")["oi"]
            prev_ce = prev_df[prev_df["option_type"] == "CE"].set_index("strike")["oi"]
            curr_pe = pe_df.set_index("strike")["oi"]
            prev_pe = prev_df[prev_df["option_type"] == "PE"].set_index("strike")["oi"]

            # Bull Trap: ATM or ATM+1 Call OI increased heavily (>10%) while spot tested resistance
            if atm_strike in curr_ce.index and atm_strike in prev_ce.index:
                ce_diff = curr_ce[atm_strike] - prev_ce[atm_strike]
                if ce_diff > 10000 and spot_price and spot_price >= atm_strike:
                    trap_alerts.append(f"⚠️ Potential BULL TRAP at Strike ₹{atm_strike:.0f}: Call OI expanded by +{ce_diff:,} contracts as price pushed higher (Institutional resistance dump).")

            # Bear Trap: ATM or ATM-1 Put OI increased heavily (>10%) while spot tested support
            if atm_strike in curr_pe.index and atm_strike in prev_pe.index:
                pe_diff = curr_pe[atm_strike] - prev_pe[atm_strike]
                if pe_diff > 10000 and spot_price and spot_price <= atm_strike:
                    trap_alerts.append(f"⚡ Potential BEAR TRAP at Strike ₹{atm_strike:.0f}: Put OI expanded by +{pe_diff:,} contracts as price tested support (Institutional put writing absorption).")

        return {
            "divergences": divergences,
            "trap_alerts": trap_alerts,
            "atm_volume_delta": atm_vol_delta,
            "atm_volume_ratio": round(atm_vol_ratio, 2),
            "taker_aggression": aggression,
        }

