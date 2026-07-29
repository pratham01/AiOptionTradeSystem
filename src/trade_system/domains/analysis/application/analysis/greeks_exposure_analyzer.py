"""
Greeks Exposure Analyzer - Computes Gamma Exposure (GEX) and Delta Exposure (DEX).
Models dealer hedging flows to determine market regimes (Trend vs Mean-Reversion)
and key support/resistance levels.
"""

import pandas as pd
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

def compute_greeks_exposure(df: pd.DataFrame, spot_price: float) -> dict:
    """
    Computes GEX and DEX for the given option chain.
    GEX = OI * Gamma * 100 * Spot Price
    DEX = OI * Delta * 100
    
    Standard convention:
    - Call GEX is positive
    - Put GEX is negative
    
    :param df: Option chain dataframe with 'strike', 'option_type', 'oi', 'delta', 'gamma'
    :param spot_price: The current spot price of the underlying
    """
    
    if df.empty or "gamma" not in df.columns or "delta" not in df.columns:
        return {}

    # Deep copy to avoid warnings
    df = df.copy()

    # Fill NaNs in greeks with 0 to prevent math errors
    df["gamma"] = df["gamma"].fillna(0.0)
    df["delta"] = df["delta"].fillna(0.0)
    df["oi"] = df["oi"].fillna(0).astype(int)

    # Calculate GEX
    df["gex"] = df["oi"] * df["gamma"] * 100 * spot_price
    # Put GEX is traditionally negative to show opposing dealer flow
    df.loc[df["option_type"] == "PE", "gex"] *= -1
    
    # Calculate DEX (Put delta is naturally negative, so DEX handles itself)
    df["dex"] = df["oi"] * df["delta"] * 100
    
    # Aggregate by Strike
    exposure_by_strike = df.groupby("strike").agg({
        "gex": "sum",
        "dex": "sum",
        "oi": "sum"
    }).reset_index()

    if exposure_by_strike.empty:
        return {}

    # Total Exposure
    total_gex = float(df["gex"].sum())
    total_dex = float(df["dex"].sum())
    
    # Gamma Walls (Absolute Max Positive and Max Negative GEX)
    call_wall_strike = float(exposure_by_strike.loc[exposure_by_strike["gex"].idxmax()]["strike"])
    put_wall_strike = float(exposure_by_strike.loc[exposure_by_strike["gex"].idxmin()]["strike"])
    
    # Zero Gamma Level (Strike where GEX profile flips sign - approximated by the strike closest to 0 GEX)
    # A more advanced way is finding where cumulative GEX from top down hits 0, but absolute min works well for local flip.
    exposure_by_strike["abs_gex"] = exposure_by_strike["gex"].abs()
    zero_gamma_strike = float(exposure_by_strike.sort_values("abs_gex").iloc[0]["strike"])

    # Market Regime
    if total_gex > 0:
        regime = "POSITIVE GAMMA (Mean Reversion / Low Volatility)"
        regime_action = "Dealers buy dips & sell rips. Expect tight ranges."
    else:
        regime = "NEGATIVE GAMMA (Momentum / High Volatility)"
        regime_action = "Dealers sell dips & buy rips. Expect explosive trends."

    # Strategy signals
    signals = []
    if spot_price > call_wall_strike:
        signals.append(f"Spot > Call Gamma Wall ({call_wall_strike}). Resistance broken, potential short squeeze!")
    elif spot_price < put_wall_strike:
        signals.append(f"Spot < Put Gamma Wall ({put_wall_strike}). Support broken, extreme downside risk!")
    
    if total_gex > 0 and (spot_price > put_wall_strike * 1.01 and spot_price < call_wall_strike * 0.99):
        signals.append("Fade the extremes: Look for short setups near Call Wall and long setups near Put Wall.")
    elif total_gex <= 0:
        signals.append("Trend following preferred. Do NOT fade moves. Breakouts will accelerate.")

    # DataFrames for plotting
    ce_df = df[df["option_type"] == "CE"][["strike", "gex", "dex", "oi"]].copy()
    pe_df = df[df["option_type"] == "PE"][["strike", "gex", "dex", "oi"]].copy()
    
    return {
        "total_gex": total_gex,
        "total_dex": total_dex,
        "zero_gamma_strike": zero_gamma_strike,
        "call_gamma_wall": call_wall_strike,
        "put_gamma_wall": put_wall_strike,
        "regime": regime,
        "regime_action": regime_action,
        "signals": signals,
        "exposure_by_strike": exposure_by_strike,
        "ce_df": ce_df,
        "pe_df": pe_df
    }
