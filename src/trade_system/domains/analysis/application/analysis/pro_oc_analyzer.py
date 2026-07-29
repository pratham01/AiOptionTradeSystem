"""
Pro Option Chain Analyzer — Institutional-grade analytics for pro traders.

Provides pure-computation functions that take a raw option chain DataFrame
(from the `option_chain_data` table) and return rich analytical dicts for
rendering on the dashboard.

Key analytics:
- Buyer vs. Seller positioning (CE/PE breakdown)
- OI & Volume concentration walls
- CE − PE OI/Volume difference per strike
- ATM straddle premium & expected move
- IV skew / smile analysis
- Greeks heatmap (Delta, Gamma, Theta, Vega)
- Auto-generated pro-trader narrative
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  1. Buyer vs. Seller Positioning
# ---------------------------------------------------------------------------

def compute_buyer_seller_positioning(
    current_df: pd.DataFrame,
    first_df: Optional[pd.DataFrame],
    spot_price: float,
    strike_step: int = 50,
) -> Dict[str, Any]:
    """
    Classify each near-ATM strike as buyer-dominated or seller-dominated
    based on intraday OI change and LTP change direction.

    Logic (standard option buildup matrix):
        OI ↑ + LTP ↑  → Long Buildup  (Buyers entering)
        OI ↑ + LTP ↓  → Short Buildup (Sellers entering / writing)
        OI ↓ + LTP ↓  → Long Unwinding (Buyers exiting)
        OI ↓ + LTP ↑  → Short Covering (Sellers exiting)

    Returns aggregate buyer/seller OI flow for CE and PE.
    """
    if current_df.empty:
        return _empty_positioning()

    atm = _round_to_strike(spot_price, strike_step)
    df = _merge_with_first(current_df, first_df, atm, strike_step, n_strikes=10)

    results = {
        "ce_buyer_oi": 0,
        "ce_seller_oi": 0,
        "pe_buyer_oi": 0,
        "pe_seller_oi": 0,
        "ce_unwind_oi": 0,
        "pe_unwind_oi": 0,
        "strike_details": [],
    }

    for _, row in df.iterrows():
        oi_chg = row["oi_change_day"]
        ltp_chg = row["ltp_change_day"]
        opt = row["option_type"]
        chg_abs = abs(int(oi_chg))

        if oi_chg > 0 and ltp_chg > 0:
            category = "Long Buildup"
            actor = "Buyers"
        elif oi_chg > 0 and ltp_chg <= 0:
            category = "Short Buildup"
            actor = "Sellers"
        elif oi_chg < 0 and ltp_chg < 0:
            category = "Long Unwinding"
            actor = "Buyers Exiting"
        elif oi_chg < 0 and ltp_chg >= 0:
            category = "Short Covering"
            actor = "Sellers Exiting"
        else:
            category = "No Change"
            actor = "Neutral"

        # Aggregate
        if opt == "CE":
            if actor == "Buyers":
                results["ce_buyer_oi"] += chg_abs
            elif actor == "Sellers":
                results["ce_seller_oi"] += chg_abs
            else:
                results["ce_unwind_oi"] += chg_abs
        else:
            if actor == "Buyers":
                results["pe_buyer_oi"] += chg_abs
            elif actor == "Sellers":
                results["pe_seller_oi"] += chg_abs
            else:
                results["pe_unwind_oi"] += chg_abs

        results["strike_details"].append({
            "strike": float(row["strike"]),
            "option_type": opt,
            "oi": int(row["oi"]),
            "oi_change": int(oi_chg),
            "ltp": float(row["ltp"]),
            "ltp_change": float(ltp_chg),
            "category": category,
            "actor": actor,
        })

    # Dominant side
    total_buyer = results["ce_buyer_oi"] + results["pe_buyer_oi"]
    total_seller = results["ce_seller_oi"] + results["pe_seller_oi"]
    if total_buyer > total_seller * 1.2:
        results["dominant"] = "BUYERS"
    elif total_seller > total_buyer * 1.2:
        results["dominant"] = "SELLERS"
    else:
        results["dominant"] = "BALANCED"

    return results


# ---------------------------------------------------------------------------
#  2. OI Concentration (Top N walls)
# ---------------------------------------------------------------------------

def compute_oi_concentration(
    df: pd.DataFrame,
    spot_price: float,
    strike_step: int = 50,
    n: int = 10,
) -> Dict[str, Any]:
    """Find top N strikes by OI for CE and PE — the support/resistance walls."""
    if df.empty:
        return {"ce_walls": [], "pe_walls": [], "highest_ce_wall": None, "highest_pe_wall": None}

    ce = df[df["option_type"] == "CE"].nlargest(n, "oi")
    pe = df[df["option_type"] == "PE"].nlargest(n, "oi")

    ce_walls = [
        {"strike": float(r["strike"]), "oi": int(r["oi"]), "volume": int(r["volume"]), "ltp": float(r["ltp"])}
        for _, r in ce.iterrows()
    ]
    pe_walls = [
        {"strike": float(r["strike"]), "oi": int(r["oi"]), "volume": int(r["volume"]), "ltp": float(r["ltp"])}
        for _, r in pe.iterrows()
    ]

    return {
        "ce_walls": ce_walls,
        "pe_walls": pe_walls,
        "highest_ce_wall": ce_walls[0] if ce_walls else None,
        "highest_pe_wall": pe_walls[0] if pe_walls else None,
    }


# ---------------------------------------------------------------------------
#  3. Volume Concentration (Top N active strikes)
# ---------------------------------------------------------------------------

def compute_volume_concentration(
    df: pd.DataFrame,
    spot_price: float,
    strike_step: int = 50,
    n: int = 10,
) -> Dict[str, Any]:
    """Find top N strikes by volume for CE and PE — where intraday action is."""
    if df.empty:
        return {"ce_active": [], "pe_active": []}

    ce = df[df["option_type"] == "CE"].nlargest(n, "volume")
    pe = df[df["option_type"] == "PE"].nlargest(n, "volume")

    return {
        "ce_active": [
            {"strike": float(r["strike"]), "volume": int(r["volume"]), "oi": int(r["oi"]), "ltp": float(r["ltp"])}
            for _, r in ce.iterrows()
        ],
        "pe_active": [
            {"strike": float(r["strike"]), "volume": int(r["volume"]), "oi": int(r["oi"]), "ltp": float(r["ltp"])}
            for _, r in pe.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
#  4. CE − PE Difference Per Strike
# ---------------------------------------------------------------------------

def compute_ce_pe_difference(
    df: pd.DataFrame,
    spot_price: float,
    strike_step: int = 50,
    n_strikes: int = 12,
) -> Dict[str, Any]:
    """
    Per-strike CE − PE OI and Volume difference.

    Interpretation:
    - Positive CE−PE OI diff → more CE writing → BEARISH (resistance wall)
    - Negative CE−PE OI diff → more PE writing → BULLISH (support wall)
    """
    if df.empty:
        return {"strike_diff": [], "net_bias": "NEUTRAL"}

    atm = _round_to_strike(spot_price, strike_step)
    nearby = df[df["strike"].apply(lambda s: abs(s - atm) <= n_strikes * strike_step)].copy()

    ce_by_strike = nearby[nearby["option_type"] == "CE"].set_index("strike")
    pe_by_strike = nearby[nearby["option_type"] == "PE"].set_index("strike")

    all_strikes = sorted(set(ce_by_strike.index) | set(pe_by_strike.index))

    diffs = []
    total_oi_diff = 0
    total_vol_diff = 0

    for s in all_strikes:
        ce_oi = int(ce_by_strike.loc[s, "oi"]) if s in ce_by_strike.index else 0
        pe_oi = int(pe_by_strike.loc[s, "oi"]) if s in pe_by_strike.index else 0
        ce_vol = int(ce_by_strike.loc[s, "volume"]) if s in ce_by_strike.index else 0
        pe_vol = int(pe_by_strike.loc[s, "volume"]) if s in pe_by_strike.index else 0

        oi_diff = ce_oi - pe_oi
        vol_diff = ce_vol - pe_vol
        total_oi_diff += oi_diff
        total_vol_diff += vol_diff

        diffs.append({
            "strike": float(s),
            "ce_oi": ce_oi,
            "pe_oi": pe_oi,
            "oi_diff": oi_diff,
            "ce_vol": ce_vol,
            "pe_vol": pe_vol,
            "vol_diff": vol_diff,
            "is_atm": abs(s - atm) < strike_step,
        })

    # Net bias: if total CE OI > PE OI → more call writing → bearish
    if total_oi_diff > 0:
        net_bias = "BEARISH"
    elif total_oi_diff < 0:
        net_bias = "BULLISH"
    else:
        net_bias = "NEUTRAL"

    return {
        "strike_diff": diffs,
        "total_oi_diff": total_oi_diff,
        "total_vol_diff": total_vol_diff,
        "net_bias": net_bias,
    }


# ---------------------------------------------------------------------------
#  5. ATM Premium Analysis
# ---------------------------------------------------------------------------

def compute_atm_premium_analysis(
    df: pd.DataFrame,
    spot_price: float,
    strike_step: int = 50,
) -> Dict[str, Any]:
    """
    ATM straddle premium, expected move, and CE/PE premium ratio.

    The ATM straddle premium tells you how much the market "expects" the
    underlying to move before expiry.
    """
    if df.empty:
        return _empty_atm()

    atm = _round_to_strike(spot_price, strike_step)

    ce_atm = df[(df["strike"] == atm) & (df["option_type"] == "CE")]
    pe_atm = df[(df["strike"] == atm) & (df["option_type"] == "PE")]

    ce_ltp = float(ce_atm["ltp"].iloc[0]) if not ce_atm.empty else 0.0
    pe_ltp = float(pe_atm["ltp"].iloc[0]) if not pe_atm.empty else 0.0

    straddle_premium = ce_ltp + pe_ltp
    expected_move_pct = (straddle_premium / spot_price * 100) if (spot_price is not None and spot_price > 0) else 0.0
    expected_move_pts = straddle_premium

    # CE/PE ratio > 1 means CE is costlier (bullish skew in premium)
    ce_pe_ratio = (ce_ltp / pe_ltp) if pe_ltp > 0 else 0.0

    # Determine premium skew interpretation
    if ce_pe_ratio > 1.15:
        premium_skew = "BULLISH SKEW (CE costlier)"
    elif ce_pe_ratio < 0.85:
        premium_skew = "BEARISH SKEW (PE costlier — fear premium)"
    else:
        premium_skew = "BALANCED"

    # ATM IV
    ce_iv = float(ce_atm["iv"].iloc[0]) if not ce_atm.empty and pd.notna(ce_atm["iv"].iloc[0]) else None
    pe_iv = float(pe_atm["iv"].iloc[0]) if not pe_atm.empty and pd.notna(pe_atm["iv"].iloc[0]) else None
    atm_iv = ((ce_iv or 0) + (pe_iv or 0)) / 2 if (ce_iv or pe_iv) else None

    # ATM OI
    ce_oi = int(ce_atm["oi"].iloc[0]) if not ce_atm.empty else 0
    pe_oi = int(pe_atm["oi"].iloc[0]) if not pe_atm.empty else 0

    return {
        "atm_strike": atm,
        "ce_ltp": ce_ltp,
        "pe_ltp": pe_ltp,
        "straddle_premium": straddle_premium,
        "expected_move_pct": expected_move_pct,
        "expected_move_pts": expected_move_pts,
        "ce_pe_ratio": ce_pe_ratio,
        "premium_skew": premium_skew,
        "atm_iv": atm_iv,
        "ce_oi": ce_oi,
        "pe_oi": pe_oi,
        "upper_breakeven": atm + straddle_premium,
        "lower_breakeven": atm - straddle_premium,
    }


# ---------------------------------------------------------------------------
#  6. IV Skew / Smile
# ---------------------------------------------------------------------------

def compute_iv_skew(
    df: pd.DataFrame,
    spot_price: float,
    strike_step: int = 50,
    n_strikes: int = 10,
) -> Dict[str, Any]:
    """
    IV across strikes for CE and PE — detects put skew (fear) or call skew.

    Standard patterns:
    - Volatility Smile: OTM puts and OTM calls both have higher IV than ATM
    - Put Skew: OTM puts have significantly higher IV (crash protection demand)
    - Flat: IV is uniform (rare, usually means low activity)
    """
    if df.empty:
        return {"ce_iv_curve": [], "pe_iv_curve": [], "skew_type": "UNKNOWN", "atm_iv": None}

    atm = _round_to_strike(spot_price, strike_step)
    nearby = df[df["strike"].apply(lambda s: abs(s - atm) <= n_strikes * strike_step)].copy()

    ce_iv = nearby[nearby["option_type"] == "CE"][["strike", "iv"]].dropna(subset=["iv"]).sort_values("strike")
    pe_iv = nearby[nearby["option_type"] == "PE"][["strike", "iv"]].dropna(subset=["iv"]).sort_values("strike")

    ce_curve = [{"strike": float(r["strike"]), "iv": float(r["iv"])} for _, r in ce_iv.iterrows()]
    pe_curve = [{"strike": float(r["strike"]), "iv": float(r["iv"])} for _, r in pe_iv.iterrows()]

    # ATM IV
    atm_iv_vals = nearby[(nearby["strike"] == atm) & nearby["iv"].notna()]["iv"]
    atm_iv = float(atm_iv_vals.mean()) if not atm_iv_vals.empty else None

    # OTM Put IV (strikes < ATM) vs OTM Call IV (strikes > ATM)
    otm_put_iv = pe_iv[pe_iv["strike"] < atm]["iv"]
    otm_call_iv = ce_iv[ce_iv["strike"] > atm]["iv"]

    avg_otm_put = float(otm_put_iv.mean()) if not otm_put_iv.empty else None
    avg_otm_call = float(otm_call_iv.mean()) if not otm_call_iv.empty else None

    # Classify skew
    if atm_iv and avg_otm_put and avg_otm_call:
        put_excess = avg_otm_put - atm_iv
        call_excess = avg_otm_call - atm_iv
        if put_excess > 1.5 and put_excess > call_excess * 1.5:
            skew_type = "PUT SKEW (Fear Premium)"
        elif call_excess > 1.5 and call_excess > put_excess * 1.5:
            skew_type = "CALL SKEW (Euphoria)"
        elif put_excess > 0.5 and call_excess > 0.5:
            skew_type = "VOLATILITY SMILE"
        else:
            skew_type = "FLAT / NORMAL"
    else:
        skew_type = "INSUFFICIENT DATA"

    return {
        "ce_iv_curve": ce_curve,
        "pe_iv_curve": pe_curve,
        "atm_iv": atm_iv,
        "avg_otm_put_iv": avg_otm_put,
        "avg_otm_call_iv": avg_otm_call,
        "skew_type": skew_type,
    }


# ---------------------------------------------------------------------------
#  7. Greeks Heatmap
# ---------------------------------------------------------------------------

def compute_greeks_heatmap(
    df: pd.DataFrame,
    spot_price: float,
    strike_step: int = 50,
    n_strikes: int = 8,
) -> Dict[str, Any]:
    """
    Prepare near-ATM Greeks data for heatmap visualization.
    Returns CE and PE Greeks side by side.
    """
    if df.empty:
        return {"heatmap_data": [], "has_meaningful_greeks": False}

    atm = _round_to_strike(spot_price, strike_step)
    nearby = df[df["strike"].apply(lambda s: abs(s - atm) <= n_strikes * strike_step)].copy()
    nearby = nearby.sort_values("strike")

    greek_cols = ["delta", "gamma", "theta", "vega"]
    has_meaningful = False

    rows = []
    strikes = sorted(nearby["strike"].unique())
    for s in strikes:
        ce_row = nearby[(nearby["strike"] == s) & (nearby["option_type"] == "CE")]
        pe_row = nearby[(nearby["strike"] == s) & (nearby["option_type"] == "PE")]

        entry = {"strike": float(s), "is_atm": abs(s - atm) < strike_step}

        for g in greek_cols:
            ce_val = float(ce_row[g].iloc[0]) if not ce_row.empty and pd.notna(ce_row[g].iloc[0]) else None
            pe_val = float(pe_row[g].iloc[0]) if not pe_row.empty and pd.notna(pe_row[g].iloc[0]) else None
            entry[f"ce_{g}"] = ce_val
            entry[f"pe_{g}"] = pe_val
            # Check if we have varying values (not all same)
            if ce_val and abs(ce_val) > 0.001:
                has_meaningful = True

        # Also include LTP and OI for context
        entry["ce_ltp"] = float(ce_row["ltp"].iloc[0]) if not ce_row.empty else None
        entry["pe_ltp"] = float(pe_row["ltp"].iloc[0]) if not pe_row.empty else None
        entry["ce_oi"] = int(ce_row["oi"].iloc[0]) if not ce_row.empty else 0
        entry["pe_oi"] = int(pe_row["oi"].iloc[0]) if not pe_row.empty else 0

        rows.append(entry)

    return {
        "heatmap_data": rows,
        "has_meaningful_greeks": has_meaningful,
    }


# ---------------------------------------------------------------------------
#  8. Pro Summary (Narrative)
# ---------------------------------------------------------------------------

def generate_pro_summary(
    positioning: Dict,
    oi_conc: Dict,
    atm_analysis: Dict,
    ce_pe_diff: Dict,
    iv_skew: Dict,
    spot_price: float,
) -> str:
    """Auto-generate a pro-trader narrative from all metrics."""
    lines: List[str] = []

    # 1. Positioning
    dom = positioning.get("dominant", "BALANCED")
    if dom == "BUYERS":
        lines.append("🟢 **Buyer-Dominated Market**: Option buyers are aggressively entering positions. "
                      "This indicates directional conviction — expect a trending move.")
    elif dom == "SELLERS":
        lines.append("🔴 **Seller-Dominated Market**: Option writers (sellers) are in control. "
                      "They are collecting premium, betting on range-bound action or mean reversion.")
    else:
        lines.append("⚪ **Balanced Positioning**: Neither buyers nor sellers have a clear upper hand. "
                      "Wait for a decisive OI shift before committing capital.")

    # 2. CE/PE buyer-seller breakdown
    ce_b, ce_s = positioning.get("ce_buyer_oi", 0), positioning.get("ce_seller_oi", 0)
    pe_b, pe_s = positioning.get("pe_buyer_oi", 0), positioning.get("pe_seller_oi", 0)
    if ce_s > ce_b and pe_s > pe_b:
        lines.append("📝 **Writers Dominate Both Sides**: Calls and Puts are being written heavily — classic range-bound / iron condor territory.")
    elif ce_b > ce_s and pe_b > pe_s:
        lines.append("🎯 **Buyers Active on Both Sides**: Straddle/strangle buyers are positioning for a big move. Expect volatility expansion.")

    # 3. OI Walls
    ce_wall = oi_conc.get("highest_ce_wall")
    pe_wall = oi_conc.get("highest_pe_wall")
    if ce_wall and pe_wall:
        lines.append(
            f"🧱 **Key Walls**: Resistance at **₹{ce_wall['strike']:,.0f}** "
            f"(CE OI: {ce_wall['oi']:,}) | Support at **₹{pe_wall['strike']:,.0f}** "
            f"(PE OI: {pe_wall['oi']:,}). "
            f"Expected trading range: ₹{pe_wall['strike']:,.0f} – ₹{ce_wall['strike']:,.0f}."
        )

    # 4. ATM Premium
    straddle = atm_analysis.get("straddle_premium", 0)
    exp_move = atm_analysis.get("expected_move_pct", 0)
    skew = atm_analysis.get("premium_skew", "BALANCED")
    if straddle > 0:
        lines.append(
            f"💰 **ATM Straddle Premium**: ₹{straddle:,.2f} "
            f"(Expected Move: **{exp_move:.2f}%** or **±{straddle:,.0f} pts**). "
            f"Premium Skew: **{skew}**."
        )

    # 5. CE-PE Bias
    bias = ce_pe_diff.get("net_bias", "NEUTRAL")
    if bias == "BULLISH":
        lines.append("📊 **Net OI Bias**: PE writers dominate → market has **bullish support cushion**.")
    elif bias == "BEARISH":
        lines.append("📊 **Net OI Bias**: CE writers dominate → market faces **bearish resistance ceiling**.")

    # 6. IV Skew
    skew_type = iv_skew.get("skew_type", "UNKNOWN")
    if "PUT SKEW" in skew_type:
        lines.append("📈 **IV Skew**: Put skew detected — traders are paying a **fear premium** for downside protection. This often precedes nervous, range-bound markets.")
    elif "CALL SKEW" in skew_type:
        lines.append("📈 **IV Skew**: Call skew detected — upside euphoria premium. Rare — can indicate a speculative rally.")
    elif "SMILE" in skew_type:
        lines.append("📈 **IV Skew**: Volatility smile — both tails are being bid up. Expect a large move in either direction.")

    # 7. Breakeven levels
    upper_be = atm_analysis.get("upper_breakeven")
    lower_be = atm_analysis.get("lower_breakeven")
    if upper_be and lower_be:
        lines.append(
            f"🎯 **Straddle Breakevens**: Upper **₹{upper_be:,.0f}** | Lower **₹{lower_be:,.0f}**. "
            f"Price must breach these for straddle buyers to profit."
        )

    return "\n\n".join(lines) if lines else "Insufficient data to generate pro-trader summary."


# ---------------------------------------------------------------------------
#  Internal Helpers
# ---------------------------------------------------------------------------

def _round_to_strike(price: float, step: int) -> float:
    if price is None or pd.isna(price):
        return 0.0
    return round(price / step) * step


def _merge_with_first(
    current_df: pd.DataFrame,
    first_df: Optional[pd.DataFrame],
    atm: float,
    strike_step: int,
    n_strikes: int = 10,
) -> pd.DataFrame:
    """Merge current snapshot with first-of-day to get daily OI/LTP change."""
    nearby = current_df[
        current_df["strike"].apply(lambda s: abs(s - atm) <= n_strikes * strike_step)
    ].copy()

    if first_df is not None and not first_df.empty:
        curr_idx = nearby.set_index(["strike", "option_type"])
        first_idx = first_df.set_index(["strike", "option_type"])

        curr_idx["oi_change_day"] = curr_idx["oi"] - first_idx["oi"].reindex(curr_idx.index, fill_value=0)
        curr_idx["ltp_change_day"] = curr_idx["ltp"] - first_idx["ltp"].reindex(curr_idx.index, fill_value=0)
        nearby = curr_idx.reset_index()
    else:
        nearby["oi_change_day"] = 0
        nearby["ltp_change_day"] = 0.0

    nearby["oi_change_day"] = nearby["oi_change_day"].fillna(0).astype(int)
    nearby["ltp_change_day"] = nearby["ltp_change_day"].fillna(0.0)

    return nearby


def _empty_positioning() -> Dict[str, Any]:
    return {
        "ce_buyer_oi": 0, "ce_seller_oi": 0,
        "pe_buyer_oi": 0, "pe_seller_oi": 0,
        "ce_unwind_oi": 0, "pe_unwind_oi": 0,
        "dominant": "NEUTRAL", "strike_details": [],
    }


def _empty_atm() -> Dict[str, Any]:
    return {
        "atm_strike": 0, "ce_ltp": 0, "pe_ltp": 0,
        "straddle_premium": 0, "expected_move_pct": 0,
        "expected_move_pts": 0, "ce_pe_ratio": 0,
        "premium_skew": "N/A", "atm_iv": None,
        "ce_oi": 0, "pe_oi": 0,
        "upper_breakeven": 0, "lower_breakeven": 0,
    }
