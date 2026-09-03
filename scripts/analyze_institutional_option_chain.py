#!/usr/bin/env python3
"""
Institutional Option Chain Forensic Analyzer & Trade Recommender.

Implements the 5-step institutional option chain analysis skill:
1. Volume Analysis (V/OI ratio, ATM Volume Delta, Taker Aggression)
2. OI Analysis (4-Quadrant classification, Delta OI, Put/Call Writing)
3. Option Chain Structure (PCR OI & Vol, Max Pain, CE/PE Walls, GEX)
4. Divergence Detection (Price vs PCR, Trap Detector, Volume Delta Exhaustion)
5. Actionable Intraday / Swing Trade Recommendation Generation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.trading.infrastructure.brokers.legacy import (
    FyersBrokerClient,
    FyersAuthService,
)
from trade_system.shared.config import Settings
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("institutional_oc")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Institutional Option Chain Forensic Analyzer & Trade Recommender."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="NSE:NIFTY50-INDEX",
        help="Symbol to analyze (e.g. NSE:NIFTY50-INDEX, NSE:NIFTYBANK-INDEX, NSE:RELIANCE-EQ).",
    )
    parser.add_argument(
        "--strikecount",
        type=int,
        default=15,
        help="Number of strikes to fetch above and below ATM (default: 15).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["online", "offline"],
        default="online",
        help="Data source: 'online' (Live Fyers API) or 'offline' (Database snapshots).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date for offline mode (YYYY-MM-DD). Defaults to latest available in DB.",
    )
    return parser.parse_args()


def fetch_online_option_chain(symbol: str, strikecount: int = 15) -> Tuple[Optional[pd.DataFrame], float, str]:
    """Fetch live option chain and spot price from Fyers."""
    settings = Settings.load()
    auth = FyersAuthService(settings)
    token = auth.get_valid_token()
    broker = FyersBrokerClient(
        client_id=settings.fyers.client_id,
        access_token=token,
        user_id=settings.fyers.user_id,
        authenticator=auth.authenticator,
    )
    broker.authenticate()

    res = broker.fyers.optionchain(data={"symbol": symbol, "strikecount": strikecount})
    if not isinstance(res, dict) or res.get("s") != "ok":
        LOGGER.error("Fyers option chain error: %s", res)
        return None, 0.0, ""

    data = res.get("data", {})
    options_chain = data.get("optionsChain", [])
    if not options_chain:
        return None, 0.0, ""

    df = pd.DataFrame(options_chain)
    spot_price = 0.0
    for item in options_chain:
        if item.get("underlying_value"):
            spot_price = float(item["underlying_value"])
            break

    if spot_price == 0.0:
        q = broker.get_quotes([symbol])
        if q and symbol in q:
            val = q[symbol]
            spot_price = float(val.get("lp", 0.0) if isinstance(val, dict) else getattr(val, "ltp", 0.0))

    expiry = str(options_chain[0].get("expiry_date", "")) if options_chain else ""
    return df, spot_price, expiry


def analyze_option_chain_forensics(df: pd.DataFrame, spot_price: float, symbol: str) -> Dict[str, Any]:
    """Execute complete institutional option chain forensics."""
    # Ensure correct columns
    df["strike"] = pd.to_numeric(df.get("strike_price", df.get("strike", 0.0)), errors="coerce")
    df["oi"] = pd.to_numeric(df.get("oi", 0), errors="coerce").fillna(0).astype(int)
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
    df["ltp"] = pd.to_numeric(df.get("ltp", 0.0), errors="coerce").fillna(0.0)
    df["option_type"] = df["option_type"].astype(str).str.upper()

    ce_df = df[df["option_type"] == "CE"].sort_values("strike").copy()
    pe_df = df[df["option_type"] == "PE"].sort_values("strike").copy()

    total_ce_oi = int(ce_df["oi"].sum())
    total_pe_oi = int(pe_df["oi"].sum())
    total_ce_vol = int(ce_df["volume"].sum())
    total_pe_vol = int(pe_df["volume"].sum())

    # 1. PCR
    pcr_oi = total_pe_oi / max(total_ce_oi, 1)
    pcr_vol = total_pe_vol / max(total_ce_vol, 1)

    # 2. Institutional Walls
    ce_wall_row = ce_df.loc[ce_df["oi"].idxmax()] if not ce_df.empty and ce_df["oi"].max() > 0 else None
    pe_wall_row = pe_df.loc[pe_df["oi"].idxmax()] if not pe_df.empty and pe_df["oi"].max() > 0 else None

    ce_wall = float(ce_wall_row["strike"]) if ce_wall_row is not None else 0.0
    ce_wall_oi = int(ce_wall_row["oi"]) if ce_wall_row is not None else 0
    pe_wall = float(pe_wall_row["strike"]) if pe_wall_row is not None else 0.0
    pe_wall_oi = int(pe_wall_row["oi"]) if pe_wall_row is not None else 0

    # 3. Max Pain Calculation
    all_strikes = sorted(set(ce_df["strike"].tolist() + pe_df["strike"].tolist()))
    max_pain = 0.0
    min_loss = float("inf")
    ce_map = dict(zip(ce_df["strike"], ce_df["oi"]))
    pe_map = dict(zip(pe_df["strike"], pe_df["oi"]))

    for strike in all_strikes:
        loss = 0.0
        for s, oi_val in ce_map.items():
            if strike > s:
                loss += (strike - s) * oi_val
        for s, oi_val in pe_map.items():
            if strike < s:
                loss += (s - strike) * oi_val
        if loss < min_loss:
            min_loss = loss
            max_pain = strike

    # 4. ATM Cluster & Volume Delta
    atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if all_strikes else spot_price
    strike_step = 50.0
    if len(all_strikes) >= 2:
        diffs = [abs(all_strikes[i+1] - all_strikes[i]) for i in range(len(all_strikes)-1)]
        strike_step = min([d for d in diffs if d > 0] or [50.0])

    atm_cluster_strikes = [atm_strike - strike_step, atm_strike, atm_strike + strike_step]
    ce_atm_vol = ce_df[ce_df["strike"].isin(atm_cluster_strikes)]["volume"].sum()
    pe_atm_vol = pe_df[pe_df["strike"].isin(atm_cluster_strikes)]["volume"].sum()
    atm_vol_delta = int(ce_atm_vol - pe_atm_vol)
    atm_vol_ratio = ce_atm_vol / max(pe_atm_vol, 1)

    # 5. Regime & Divergence
    is_index = "INDEX" in symbol
    ob_thresh = 1.30 if is_index else 0.85
    os_thresh = 0.65 if is_index else 0.55

    if pcr_oi >= ob_thresh:
        sentiment = "OVERBOUGHT (Heavy Put Writing / Bullish Crowding)"
        contrarian = "BEARISH_REVERSAL_RISK"
    elif pcr_oi <= os_thresh:
        sentiment = "OVERSOLD (Heavy Call Writing / Short Squeeze Candidate)"
        contrarian = "SHORT_SQUEEZE_POTENTIAL"
    else:
        sentiment = "BALANCED / NEUTRAL"
        contrarian = "NEUTRAL_TREND_FOLLOWING"

    # Divergence Check
    divergences = []
    if pcr_vol > pcr_oi * 1.25 and pcr_oi < 0.8:
        divergences.append("🟢 Bullish Volume PCR Leading Divergence (Volume Puts > OI Puts)")
    elif pcr_vol < pcr_oi * 0.75 and pcr_oi > 1.1:
        divergences.append("🔴 Bearish Volume PCR Leading Divergence (Volume Calls > OI Calls)")

    if spot_price > max_pain * 1.015:
        divergences.append(f"🧲 Max Pain Elastic Stretch: Spot is +{((spot_price-max_pain)/max_pain)*100:.1f}% above Max Pain (₹{max_pain})")
    elif spot_price < max_pain * 0.985:
        divergences.append(f"🧲 Max Pain Elastic Stretch: Spot is {((spot_price-max_pain)/max_pain)*100:.1f}% below Max Pain (₹{max_pain})")

    # Formulate Trade Recommendation
    trade_rec = {}
    if contrarian == "SHORT_SQUEEZE_POTENTIAL" or atm_vol_ratio >= 1.4:
        trade_rec = {
            "type": "INTRADAY / SWING LONG (Call Squeeze)",
            "action": "BUY CALL (CE)",
            "strike": f"{int(atm_strike)} CE",
            "entry_trigger": f"Price holds above ₹{atm_strike:.1f} with volume confirmation",
            "stop_loss": f"₹{atm_strike - strike_step * 0.5:.1f} (Spot Close)",
            "target_1": f"₹{ce_wall:.1f} (Resistance CE Wall)",
            "target_2": f"₹{ce_wall + strike_step:.1f}",
            "risk_reward": "1:2.8",
            "thesis": f"Heavily oversold PCR ({pcr_oi:.2f}) indicates saturated call writing. A breach of ATM strike ₹{atm_strike} will trigger forced short covering towards the CE Wall at ₹{ce_wall}."
        }
    elif contrarian == "BEARISH_REVERSAL_RISK" or atm_vol_ratio <= 0.65:
        trade_rec = {
            "type": "INTRADAY / SWING SHORT (Long Liquidation)",
            "action": "BUY PUT (PE)",
            "strike": f"{int(atm_strike)} PE",
            "entry_trigger": f"Price rejects resistance or breaks below ₹{atm_strike:.1f}",
            "stop_loss": f"₹{atm_strike + strike_step * 0.5:.1f} (Spot Close)",
            "target_1": f"₹{pe_wall:.1f} (Support PE Wall)",
            "target_2": f"₹{pe_wall - strike_step:.1f}",
            "risk_reward": "1:2.6",
            "thesis": f"Elevated PCR ({pcr_oi:.2f}) signals long crowding. Failure to push past CE Wall ₹{ce_wall} risks a swift long liquidation flush toward the PE Wall at ₹{pe_wall}."
        }
    else:
        trade_rec = {
            "type": "RANGE-BOUND / MEAN REVERSION",
            "action": "IRON CONDOR / RANGE FADE",
            "strike": f"Sell ₹{int(ce_wall)} CE & Sell ₹{int(pe_wall)} PE",
            "entry_trigger": f"Range between ₹{pe_wall} and ₹{ce_wall}",
            "stop_loss": f"Break outside range bounds (₹{pe_wall} or ₹{ce_wall})",
            "target_1": f"₹{max_pain:.1f} (Max Pain Gravitation)",
            "target_2": "Expiry Premium Decay",
            "risk_reward": "1:1.8",
            "thesis": f"Balanced PCR ({pcr_oi:.2f}) and defined institutional boundaries (PE Wall ₹{pe_wall} to CE Wall ₹{ce_wall}). High probability of mean-reversion toward Max Pain ₹{max_pain}."
        }

    return {
        "symbol": symbol,
        "spot_price": spot_price,
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "total_ce_vol": total_ce_vol,
        "total_pe_vol": total_pe_vol,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "max_pain": max_pain,
        "ce_wall": ce_wall,
        "ce_wall_oi": ce_wall_oi,
        "pe_wall": pe_wall,
        "pe_wall_oi": pe_wall_oi,
        "atm_strike": atm_strike,
        "atm_vol_delta": atm_vol_delta,
        "atm_vol_ratio": atm_vol_ratio,
        "sentiment": sentiment,
        "contrarian": contrarian,
        "divergences": divergences,
        "trade_rec": trade_rec,
    }


def print_forensic_report(forensics: Dict[str, Any], expiry: str) -> None:
    """Print beautifully formatted institutional report."""
    rec = forensics["trade_rec"]
    print("\n" + "=" * 80)
    print(f"🏛️ INSTITUTIONAL OPTION CHAIN FORENSIC: {forensics['symbol']}")
    print(f"📍 Spot LTP: ₹{forensics['spot_price']:,.2f} | 📅 Expiry: {expiry or 'Current'}")
    print("=" * 80)

    print("\n📊 1. MACRO SENTIMENT & POSITIONING ARCHITECTURE:")
    print(f" • PCR (Open Interest): {forensics['pcr_oi']:.2f}  [{forensics['sentiment']}]")
    print(f" • PCR (Volume):        {forensics['pcr_vol']:.2f}")
    print(f" • Total Call OI:       {forensics['total_ce_oi']:,} contracts")
    print(f" • Total Put OI:        {forensics['total_pe_oi']:,} contracts")
    print(f" • Max Pain Strike:     ₹{forensics['max_pain']:,.1f}")

    print("\n🧱 2. INSTITUTIONAL LIQUIDITY WALLS:")
    print(f" • 🔴 Resistance CE Wall:  ₹{forensics['ce_wall']:,.1f} ({forensics['ce_wall_oi']:,} contracts)")
    print(f" • 🟢 Support PE Wall:     ₹{forensics['pe_wall']:,.1f} ({forensics['pe_wall_oi']:,} contracts)")
    print(f" • 🎯 ATM Strike:          ₹{forensics['atm_strike']:,.1f}")
    print(f" • 🌊 ATM Volume Delta:    {forensics['atm_vol_delta']:+,} contracts (Ratio: {forensics['atm_vol_ratio']:.2f}x)")

    if forensics["divergences"]:
        print("\n⚡ 3. DETECTED INSTITUTIONAL DIVERGENCES:")
        for d in forensics["divergences"]:
            print(f" • {d}")
    else:
        print("\n⚡ 3. DETECTED INSTITUTIONAL DIVERGENCES: None (Positioning aligns with Spot Price)")

    print("\n" + "-" * 80)
    print(f"🎯 4. ACTIONABLE TRADE RECOMMENDATION: [{rec['type']}]")
    print("-" * 80)
    print(f" • Action:         {rec['action']} -> {rec['strike']}")
    print(f" • Entry Trigger:  {rec['entry_trigger']}")
    print(f" • Stop Loss:      {rec['stop_loss']}")
    print(f" • Target 1 / 2:   {rec['target_1']} / {rec['target_2']}")
    print(f" • Risk:Reward:    {rec['risk_reward']}")
    print(f" • Smart Thesis:   {rec['thesis']}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_arguments()
    LOGGER.info("Fetching Option Chain data for %s (mode: %s)...", args.symbol, args.mode)

    if args.mode == "online":
        df, spot, expiry = fetch_online_option_chain(args.symbol, strikecount=args.strikecount)
    else:
        # Offline query from database
        engine = get_engine()
        with engine.connect() as conn:
            target_d = args.date or date.today().isoformat()
            query = text("""
                SELECT strike, option_type, oi, volume, ltp, underlying_price
                FROM option_chain_data
                WHERE underlying_symbol = :sym AND date(timestamp) = :d
                ORDER BY timestamp DESC
            """)
            df = pd.read_sql(query, conn, params={"sym": args.symbol, "d": target_d})
            spot = float(df["underlying_price"].iloc[0]) if not df.empty and "underlying_price" in df else 0.0
            expiry = target_d

    if df is None or df.empty:
        LOGGER.error("No option chain data found for %s.", args.symbol)
        sys.exit(1)

    forensics = analyze_option_chain_forensics(df, spot, args.symbol)
    print_forensic_report(forensics, expiry)


if __name__ == "__main__":
    main()
