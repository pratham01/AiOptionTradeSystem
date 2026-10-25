"""
Strike Supertrend Radar Dashboard
=================================
Calculates and visualizes the Supertrend indicator on the near strike prices
(ATM and surrounding OTM/ITM strikes) of the spot price from the live option chain.

Features:
  - Real-time detection of spot price and ATM strike
  - Dynamic extraction of the 5 strikes closest to spot (ATM-2, ATM-1, ATM, ATM+1, ATM+2)
  - Intraday candle fetching and Supertrend calculation for both CE and PE contracts
  - Directional consensus (Bullish / Bearish / Rangebound Theta Decay / Volatility Surge)
  - Visual 5-Strike Supertrend Direction Matrix cards
  - High-performance interactive Plotly Candlestick + Supertrend chart with flip markers
  - Detailed comparison table and institutional option playbook
  - ThreadPoolExecutor parallel fetching with caching and database fallback
"""

import logging
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.strategy.application.indicators.supertrend import calculate_supertrend
from trade_system.interfaces.dashboard.shared_broker import get_cached_broker

LOGGER = logging.getLogger(__name__)

# ── Custom CSS & Aesthetics ──────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 0.8rem !important; }

    /* Header Banner */
    .st-hero {
        background: linear-gradient(135deg, #091224 0%, #0d1e3d 40%, #06162d 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 6px solid #38bdf8;
        border-radius: 14px;
        padding: 1rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .st-hero h2 {
        margin: 0;
        color: #f0f9ff;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .st-hero .subtitle {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 3px;
    }

    /* Direction Badges */
    .badge-bull {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(16, 185, 129, 0.08) 100%);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        letter-spacing: 0.02em;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.15);
    }
    .badge-bear {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(239, 68, 68, 0.08) 100%);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        letter-spacing: 0.02em;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.15);
    }
    .badge-atm {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(245, 158, 11, 0.1) 100%);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.5);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-moneyness {
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* Strike Matrix Card */
    .strike-card {
        background: #0b1324;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .strike-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .strike-card-atm {
        background: linear-gradient(180deg, #101c36 0%, #0b1426 100%);
        border: 1.5px solid rgba(245, 158, 11, 0.6);
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.12);
    }

    /* Metric pill */
    .metric-pill {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Fetching & Extraction Helpers ────────────────────────────────────────

INDEX_OPTIONS = {
    "NIFTY 50": "NSE:NIFTY50-INDEX",
    "BANK NIFTY": "NSE:NIFTYBANK-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX",
    "FIN NIFTY": "NSE:FINNIFTY-INDEX",
    "MIDCAP NIFTY": "NSE:MIDCPNIFTY-INDEX",
}


@st.cache_data(ttl=15)
def fetch_option_chain_raw(symbol: str, expiry_timestamp: Optional[str] = None, strike_count: int = 15) -> Tuple[Optional[List[Dict[str, Any]]], float, List[Dict[str, str]], str]:
    """Fetch live option chain with spot price and available expiries from Fyers broker."""
    broker = get_cached_broker()
    if not broker:
        return None, 0.0, [], "Broker not authenticated."

    req_data: Dict[str, Any] = {"symbol": symbol, "strikecount": strike_count}
    if expiry_timestamp:
        req_data["timestamp"] = str(expiry_timestamp)

    try:
        res = broker.fyers.optionchain(data=req_data)
        if not isinstance(res, dict) or res.get("s") != "ok":
            errmsg = res.get("message") or res.get("errmsg") or "Non-ok response from broker option chain API"
            return None, 0.0, [], str(errmsg)

        data = res.get("data", {})
        options_chain = data.get("optionsChain", [])
        expiry_data = data.get("expiryData", [])

        # Detect spot price
        spot_price = 0.0
        for item in options_chain:
            if item.get("strike_price") == -1 or item.get("symbol") == symbol:
                spot_price = float(item.get("ltp", 0.0))
                if spot_price > 0:
                    break

        if spot_price == 0.0:
            for item in options_chain:
                if item.get("underlying_value"):
                    spot_price = float(item["underlying_value"])
                    if spot_price > 0:
                        break

        if spot_price == 0.0:
            q = broker.get_quotes([symbol])
            if q and symbol in q:
                val = q[symbol]
                spot_price = float(val.get("lp", 0.0) if isinstance(val, dict) else getattr(val, "ltp", 0.0))

        return options_chain, spot_price, expiry_data, ""
    except Exception as exc:
        LOGGER.error("Failed to fetch option chain for %s: %s", symbol, exc)
        return None, 0.0, [], str(exc)


@st.cache_data(ttl=15)
def fetch_contract_candles(symbol: str, resolution: str = "1") -> pd.DataFrame:
    """Fetch intraday candles for a single contract or spot from broker, with DB fallback."""
    today_str = date.today().strftime("%Y-%m-%d")
    broker = get_cached_broker()
    if broker:
        try:
            df = broker.fetch_history(
                symbol=symbol,
                resolution=resolution,
                range_from=today_str,
                range_to=today_str
            )
            if df is not None and not df.empty and len(df) >= 2:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as exc:
            LOGGER.debug("Broker fetch_history failed for %s: %s", symbol, exc)

    # Fallback to option_chain_data database snapshots
    try:
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            query = text("""
                SELECT timestamp, ltp, volume
                FROM option_chain_data
                WHERE symbol = :sym AND timestamp >= CURRENT_DATE
                ORDER BY timestamp ASC
            """)
            db_df = pd.read_sql(query, conn, params={"sym": symbol})
            if not db_df.empty and len(db_df) >= 3:
                db_df["timestamp"] = pd.to_datetime(db_df["timestamp"])
                db_df = db_df.set_index("timestamp")
                resample_rule = f"{resolution}min" if resolution.isdigit() else "1min"
                ohlc = db_df["ltp"].resample(resample_rule).ohlc().dropna()
                vol = db_df["volume"].resample(resample_rule).last().fillna(0)
                ohlc["volume"] = vol
                return ohlc.reset_index()
    except Exception as exc:
        LOGGER.debug("DB fallback failed for %s: %s", symbol, exc)

    return pd.DataFrame()


def compute_supertrend_stats(df: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> Optional[Dict[str, Any]]:
    """Compute Supertrend indicator and extract key directional metrics."""
    if df is None or df.empty or len(df) < period:
        return None

    try:
        st_df = calculate_supertrend(df, period=period, multiplier=multiplier)
    except Exception as exc:
        LOGGER.warning("calculate_supertrend error: %s", exc)
        return None

    if st_df is None or st_df.empty or "supertrend" not in st_df.columns:
        return None

    last = st_df.iloc[-1]
    prev = st_df.iloc[-2] if len(st_df) > 1 else last

    direction = int(last.get("supertrend_direction", 0))
    st_level = float(last.get("supertrend", 0.0))
    ltp = float(last.get("close", 0.0))
    dist_pts = ltp - st_level
    dist_pct = (dist_pts / st_level * 100.0) if st_level > 0 else 0.0

    # Trend duration (how many consecutive bars in current direction)
    trend_bars = 0
    for i in range(len(st_df) - 1, -1, -1):
        if int(st_df.iloc[i].get("supertrend_direction", 0)) == direction:
            trend_bars += 1
        else:
            break

    # Flip detection
    just_flipped = (int(last.get("supertrend_direction", 0)) != int(prev.get("supertrend_direction", 0)))
    flip_type = "NONE"
    if just_flipped:
        flip_type = "BULLISH_FLIP" if direction == 1 else "BEARISH_FLIP"

    # Price change compared to session open
    session_open = float(st_df.iloc[0].get("open", ltp))
    day_change_pts = ltp - session_open
    day_change_pct = (day_change_pts / session_open * 100.0) if session_open > 0 else 0.0

    return {
        "df": st_df,
        "direction": direction,
        "dir_label": "BULLISH" if direction == 1 else "BEARISH",
        "ltp": ltp,
        "st_level": st_level,
        "dist_pts": dist_pts,
        "dist_pct": dist_pct,
        "trend_bars": trend_bars,
        "just_flipped": just_flipped,
        "flip_type": flip_type,
        "day_change_pts": day_change_pts,
        "day_change_pct": day_change_pct,
        "high": float(st_df["high"].max()),
        "low": float(st_df["low"].min()),
        "volume": int(last.get("volume", 0)),
        "timestamp": last.get("timestamp"),
    }


def find_near_strikes(all_strikes: List[float], spot_price: float, count: int = 5) -> Tuple[List[float], float]:
    """
    Select the N strikes closest to spot price (default 5: ATM-2, ATM-1, ATM, ATM+1, ATM+2).
    Returns (near_strikes, atm_strike).
    """
    if not all_strikes:
        return [], 0.0

    sorted_strikes = sorted(all_strikes)
    atm_strike = min(sorted_strikes, key=lambda s: abs(s - spot_price))
    atm_idx = sorted_strikes.index(atm_strike)

    half = count // 2
    start_idx = max(0, atm_idx - half)
    end_idx = min(len(sorted_strikes), start_idx + count)

    # Adjust start if end hit boundary
    if (end_idx - start_idx) < count and start_idx > 0:
        start_idx = max(0, end_idx - count)

    return sorted_strikes[start_idx:end_idx], atm_strike


# ── Render Hero Banner ───────────────────────────────────────────────────────
st.markdown("""
<div class="st-hero">
    <div>
        <h2>🎯 Near-Spot Strike Supertrend Radar</h2>
        <div class="subtitle">
            Dynamic Strike-Level Supertrend Momentum, Directional Confluence & Trailing Traps on Near 5 Strikes
        </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="badge-moneyness" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-color: rgba(56, 189, 248, 0.3);">
            ● Live Engine Active
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar & Navigation Controls ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Strike Radar Settings")

    index_name = st.selectbox("Underlying Index", list(INDEX_OPTIONS.keys()), index=0)
    selected_symbol = INDEX_OPTIONS[index_name]

    # Fetch initial chain to populate expiries & spot
    raw_oc, spot_est, expiries_list, err_msg = fetch_option_chain_raw(selected_symbol, strike_count=15)

    expiry_choices = []
    expiry_map = {}
    if expiries_list:
        for item in expiries_list[:8]:
            d_str = item.get("date", "")
            flag = item.get("expiry_flag", "")
            label = f"{d_str} ({'Weekly' if flag == 'W' else 'Monthly'})"
            expiry_choices.append(label)
            expiry_map[label] = item.get("expiry", "")

    selected_expiry_label = st.selectbox(
        "Expiry Contract",
        expiry_choices if expiry_choices else ["Nearest Current Expiry"],
        index=0
    )
    chosen_expiry_ts = expiry_map.get(selected_expiry_label, None) if expiry_choices else None

    strike_count_option = st.select_slider(
        "Strike Range",
        options=[3, 5, 7, 9],
        value=5,
        help="Number of strikes centered around spot price to analyze (Default: 5)"
    )

    col_res, col_period = st.columns(2)
    with col_res:
        candle_resolution = st.selectbox("Timeframe", ["1", "3", "5", "15"], index=0, format_func=lambda x: f"{x} min")
    with col_period:
        st_period = st.number_input("ST Period", min_value=3, max_value=30, value=7, step=1)

    st_multiplier = st.slider("ST Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

    st.markdown("---")
    if st.button("🔄 Refresh Radar Now", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()


# ── Load Option Chain Data ───────────────────────────────────────────────────
with st.spinner("Fetching live option chain and computing Supertrend across near strikes..."):
    options_chain, spot_price, expiries_data, fetch_err = fetch_option_chain_raw(
        selected_symbol,
        expiry_timestamp=chosen_expiry_ts,
        strike_count=max(strike_count_option * 2, 12)
    )

if not options_chain:
    st.error(f"⚠️ Unable to load option chain data for **{selected_symbol}**: {fetch_err or 'No data returned.'}")
    st.info("💡 Please verify broker connectivity in the **Broker Health** dashboard tab or check market hours.")
    st.stop()

# Filter option chain to valid CE/PE contracts
contracts = [c for c in options_chain if c.get("option_type") in ("CE", "PE") and c.get("strike_price", -1) > 0]
if not contracts:
    st.warning("No CE/PE option contracts available in current response.")
    st.stop()

all_unique_strikes = sorted(list(set(float(c["strike_price"]) for c in contracts)))
near_strikes, atm_strike = find_near_strikes(all_unique_strikes, spot_price, count=strike_count_option)

# Prepare parallel fetch of contract candles
symbols_to_fetch: List[Tuple[float, str, str, Dict[str, Any]]] = []
for s in near_strikes:
    ce_item = next((c for c in contracts if float(c.get("strike_price")) == s and c.get("option_type") == "CE"), None)
    pe_item = next((c for c in contracts if float(c.get("strike_price")) == s and c.get("option_type") == "PE"), None)
    if ce_item and ce_item.get("symbol"):
        symbols_to_fetch.append((s, "CE", ce_item["symbol"], ce_item))
    if pe_item and pe_item.get("symbol"):
        symbols_to_fetch.append((s, "PE", pe_item["symbol"], pe_item))

# Also fetch Spot price candles for underlying Supertrend reference
spot_symbol = selected_symbol


def _worker_fetch(item: Tuple[float, str, str, Dict[str, Any]]):
    strike, otype, sym, meta = item
    candles = fetch_contract_candles(sym, resolution=candle_resolution)
    stats = compute_supertrend_stats(candles, period=int(st_period), multiplier=float(st_multiplier))
    return strike, otype, sym, meta, stats


results_dict: Dict[Tuple[float, str], Dict[str, Any]] = {}
with ThreadPoolExecutor(max_workers=min(len(symbols_to_fetch) + 1, 8)) as executor:
    # Option contracts
    for strike, otype, sym, meta, stats in executor.map(_worker_fetch, symbols_to_fetch):
        results_dict[(strike, otype)] = {
            "symbol": sym,
            "meta": meta,
            "stats": stats
        }

# Fetch spot candles
spot_candles = fetch_contract_candles(spot_symbol, resolution=candle_resolution)
spot_st_stats = compute_supertrend_stats(spot_candles, period=int(st_period), multiplier=float(st_multiplier))


# ── Aggregate Directional Metrics ────────────────────────────────────────────
total_ce = 0
ce_bullish_count = 0
total_pe = 0
pe_bullish_count = 0
recent_flips: List[str] = []

strike_summaries: List[Dict[str, Any]] = []
for s in near_strikes:
    ce_data = results_dict.get((s, "CE"), {})
    pe_data = results_dict.get((s, "PE"), {})
    ce_stats = ce_data.get("stats")
    pe_stats = pe_data.get("stats")

    ce_dir = ce_stats["direction"] if ce_stats else 0
    pe_dir = pe_stats["direction"] if pe_stats else 0

    if ce_stats:
        total_ce += 1
        if ce_dir == 1:
            ce_bullish_count += 1
        if ce_stats.get("just_flipped"):
            recent_flips.append(f"{int(s)} CE ({ce_stats['dir_label']})")

    if pe_stats:
        total_pe += 1
        if pe_dir == 1:
            pe_bullish_count += 1
        if pe_stats.get("just_flipped"):
            recent_flips.append(f"{int(s)} PE ({pe_stats['dir_label']})")

    # Strike Net Bias
    # CE Bullish (+1) & PE Bearish (-1) => Strong Bullish
    # CE Bearish (-1) & PE Bullish (+1) => Strong Bearish
    # CE Bearish (-1) & PE Bearish (-1) => Theta Decay / Writers Dominating
    # CE Bullish (+1) & PE Bullish (+1) => Volatility Spike / Straddle Buying
    if ce_dir == 1 and pe_dir == -1:
        net_bias = "🟢 CALL MOMENTUM (BULLISH)"
        bias_code = "BULL"
    elif ce_dir == -1 and pe_dir == 1:
        net_bias = "🔴 PUT MOMENTUM (BEARISH)"
        bias_code = "BEAR"
    elif ce_dir == -1 and pe_dir == -1:
        net_bias = "⚪ THETA DECAY (DUAL SHORT)"
        bias_code = "DECAY"
    elif ce_dir == 1 and pe_dir == 1:
        net_bias = "🟣 VOLATILITY SURGE (DUAL LONG)"
        bias_code = "VOL"
    else:
        net_bias = "🟡 NEUTRAL / MIXED"
        bias_code = "NEUTRAL"

    strike_summaries.append({
        "strike": s,
        "is_atm": (s == atm_strike),
        "dist_from_spot": s - spot_price,
        "ce_data": ce_data,
        "pe_data": pe_data,
        "net_bias": net_bias,
        "bias_code": bias_code
    })

# Market Consensus Verdict
if total_ce > 0 and total_pe > 0:
    ce_bull_pct = (ce_bullish_count / total_ce) * 100
    pe_bull_pct = (pe_bullish_count / total_pe) * 100

    if pe_bull_pct >= 60 and ce_bull_pct <= 40:
        consensus_title = "🔴 BEARISH OPTION DOMINANCE"
        consensus_desc = f"Put buyers in control ({pe_bullish_count}/{total_pe} PEs Bullish, {total_ce - ce_bullish_count}/{total_ce} CEs Bearish). Market under persistent supply."
        consensus_color = "#ef4444"
    elif ce_bull_pct >= 60 and pe_bull_pct <= 40:
        consensus_title = "🟢 BULLISH OPTION DOMINANCE"
        consensus_desc = f"Call buyers in control ({ce_bullish_count}/{total_ce} CEs Bullish, {total_pe - pe_bullish_count}/{total_pe} PEs Bearish). Market in active upward expansion."
        consensus_color = "#10b981"
    elif ce_bull_pct <= 30 and pe_bull_pct <= 30:
        consensus_title = "⚪ SELLER THETA HARVEST (RANGEBOUND)"
        consensus_desc = "Both Calls and Puts are trapped below their Supertrends. Heavy writer shorting & theta decay."
        consensus_color = "#94a3b8"
    elif ce_bull_pct >= 70 and pe_bull_pct >= 70:
        consensus_title = "🟣 HIGH VOLATILITY / STRADDLE EXPANSION"
        consensus_desc = "Both Calls and Puts are trading above their Supertrends. Huge IV spike or large range swing underway."
        consensus_color = "#c084fc"
    else:
        consensus_title = "🟡 ROTATIONAL / INDECISIVE"
        consensus_desc = "Mixed signals across near strikes. Wait for directional confluence before sizing up."
        consensus_color = "#f59e0b"
else:
    consensus_title = "⏳ COMPUTING..."
    consensus_desc = "Collecting historical candles for near strikes."
    consensus_color = "#64748b"


def render_safe_html(html: str) -> None:
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


# ── Top KPI Bar ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.6, 1.4])

with col1:
    render_safe_html(f"""
    <div class="metric-pill">
        <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">UNDERLYING SPOT</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc;">₹{spot_price:,.2f}</div>
        <div style="font-size: 0.75rem; color: #38bdf8;">{selected_symbol}</div>
    </div>
    """)

with col2:
    if spot_st_stats:
        s_dir = spot_st_stats["dir_label"]
        s_lvl = spot_st_stats["st_level"]
        s_badge_class = "badge-bull" if spot_st_stats["direction"] == 1 else "badge-bear"
        s_icon = "▲" if spot_st_stats["direction"] == 1 else "▼"
        spot_st_html = f'<span class="{s_badge_class}">{s_icon} {s_dir}</span> <span style="font-size:0.75rem; color:#94a3b8;">ST: ₹{s_lvl:,.1f}</span>'
    else:
        spot_st_html = '<span style="color:#64748b;">N/A</span>'

    render_safe_html(f"""
    <div class="metric-pill">
        <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">ATM STRIKE & SPOT ST</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #fbbf24;">{int(atm_strike):,} <span class="badge-atm">ATM</span></div>
        <div style="margin-top: 2px;">{spot_st_html}</div>
    </div>
    """)

with col3:
    render_safe_html(f"""
    <div class="metric-pill" style="border-left: 3px solid {consensus_color};">
        <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">OPTION DIRECTION CONSENSUS</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: {consensus_color};">{consensus_title}</div>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">{consensus_desc}</div>
    </div>
    """)

with col4:
    flips_html = ", ".join(recent_flips) if recent_flips else "No fresh flips in last candle"
    render_safe_html(f"""
    <div class="metric-pill">
        <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">RADAR STATS ({candle_resolution}m TF)</div>
        <div style="font-size: 0.85rem; color: #e2e8f0; margin-top: 3px;">
            <b>CEs:</b> <span style="color:#34d399;">{ce_bullish_count} Bull</span> / <span style="color:#f87171;">{total_ce - ce_bullish_count} Bear</span><br/>
            <b>PEs:</b> <span style="color:#34d399;">{pe_bullish_count} Bull</span> / <span style="color:#f87171;">{total_pe - pe_bullish_count} Bear</span>
        </div>
        <div style="font-size: 0.68rem; color: #f59e0b; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            ⚡ {flips_html}
        </div>
    </div>
    """)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ── The Near 5 Strikes Supertrend Direction Matrix ───────────────────────────
st.markdown("### 📊 Strike-by-Strike Supertrend Radar (Near Spot Strikes)")
st.caption(f"Real-time Supertrend ({st_period}, {st_multiplier}) on the {len(near_strikes)} strikes closest to spot (₹{spot_price:,.2f})")

# Render responsive cards for each strike
for item in strike_summaries:
    s = item["strike"]
    is_atm = item["is_atm"]
    dist_pts = item["dist_from_spot"]
    ce_stats = item["ce_data"].get("stats")
    pe_stats = item["pe_data"].get("stats")
    net_bias = item["net_bias"]

    # Moneyness labels
    if s < spot_price:
        ce_money = "ITM Call"
        pe_money = "OTM Put"
    elif s > spot_price:
        ce_money = "OTM Call"
        pe_money = "ITM Put"
    else:
        ce_money = "ATM Call"
        pe_money = "ATM Put"

    dist_str = f"+{dist_pts:,.1f}" if dist_pts > 0 else f"{dist_pts:,.1f}"
    atm_tag = '<span class="badge-atm">★ ATM STRIKE</span>' if is_atm else ""

    with st.container(border=True):
        render_safe_html(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">{int(s):,}</span>
                {atm_tag}
                <span class="badge-moneyness">{dist_str} pts from spot</span>
            </div>
            <div style="font-size: 0.85rem; font-weight: 600;">
                {net_bias}
            </div>
        </div>
        """)

        c_col, p_col = st.columns(2)
        with c_col:
            if ce_stats:
                c_dir = ce_stats["dir_label"]
                c_badge = "badge-bull" if ce_stats["direction"] == 1 else "badge-bear"
                c_icon = "▲" if ce_stats["direction"] == 1 else "▼"
                c_border = "#10b981" if ce_stats["direction"] == 1 else "#ef4444"
                c_flip = f'<span style="color:#fbbf24; font-size:0.70rem; font-weight:700; margin-left:6px;">⚡ {ce_stats["flip_type"]}</span>' if ce_stats["just_flipped"] else ""
                c_chg_color = "#34d399" if ce_stats["day_change_pct"] >= 0 else "#f87171"

                render_safe_html(f"""
                <div style="background: rgba(15, 23, 42, 0.55); padding: 12px 14px; border-radius: 10px; border-left: 4px solid {c_border}; border-top: 1px solid rgba(255,255,255,0.06); border-right: 1px solid rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #38bdf8; font-size: 0.88rem;">CALL (CE) <small style="color:#94a3b8; font-weight:normal;">• {ce_money}</small></span>
                        <div><span class="{c_badge}">{c_icon} {c_dir}</span>{c_flip}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">₹{ce_stats['ltp']:,.2f}</span>
                        <span style="color:{c_chg_color}; font-size:0.82rem; font-weight:600;">{ce_stats['day_change_pct']:+,.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94a3b8; margin-top: 6px;">
                        <span>ST Stop: <b style="color:#cbd5e1;">₹{ce_stats['st_level']:,.2f}</b></span>
                        <span>Dist: <b style="color:{c_border};">{ce_stats['dist_pts']:+,.2f} ({ce_stats['dist_pct']:+,.1f}%)</b></span>
                    </div>
                    <div style="font-size: 0.70rem; color: #64748b; margin-top: 4px;">
                        <span>⏱️ In Trend: <b>{ce_stats['trend_bars']} bars</b> ({candle_resolution}m TF)</span>
                    </div>
                </div>
                """)
            else:
                st.caption("CE Data not available")

        with p_col:
            if pe_stats:
                p_dir = pe_stats["dir_label"]
                p_badge = "badge-bull" if pe_stats["direction"] == 1 else "badge-bear"
                p_icon = "▲" if pe_stats["direction"] == 1 else "▼"
                p_border = "#10b981" if pe_stats["direction"] == 1 else "#ef4444"
                p_flip = f'<span style="color:#fbbf24; font-size:0.70rem; font-weight:700; margin-left:6px;">⚡ {pe_stats["flip_type"]}</span>' if pe_stats["just_flipped"] else ""
                p_chg_color = "#34d399" if pe_stats["day_change_pct"] >= 0 else "#f87171"

                render_safe_html(f"""
                <div style="background: rgba(15, 23, 42, 0.55); padding: 12px 14px; border-radius: 10px; border-left: 4px solid {p_border}; border-top: 1px solid rgba(255,255,255,0.06); border-right: 1px solid rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #f472b6; font-size: 0.88rem;">PUT (PE) <small style="color:#94a3b8; font-weight:normal;">• {pe_money}</small></span>
                        <div><span class="{p_badge}">{p_icon} {p_dir}</span>{p_flip}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">₹{pe_stats['ltp']:,.2f}</span>
                        <span style="color:{p_chg_color}; font-size:0.82rem; font-weight:600;">{pe_stats['day_change_pct']:+,.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94a3b8; margin-top: 6px;">
                        <span>ST Stop: <b style="color:#cbd5e1;">₹{pe_stats['st_level']:,.2f}</b></span>
                        <span>Dist: <b style="color:{p_border};">{pe_stats['dist_pts']:+,.2f} ({pe_stats['dist_pct']:+,.1f}%)</b></span>
                    </div>
                    <div style="font-size: 0.70rem; color: #64748b; margin-top: 4px;">
                        <span>⏱️ In Trend: <b>{pe_stats['trend_bars']} bars</b> ({candle_resolution}m TF)</span>
                    </div>
                </div>
                """)
            else:
                st.caption("PE Data not available")


# ── Interactive Candlestick + Supertrend Chart Visualizer ────────────────────
st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
st.markdown("### 📈 Interactive Candlestick & Supertrend Chart")
st.caption("Deep-dive into premium price action, trailing stop-losses, and historical Supertrend flips for any contract.")

chart_dict: Dict[str, Dict[str, Any]] = {}

# Add spot
if spot_st_stats and spot_st_stats.get("df") is not None:
    spot_lbl = f"Underlying Spot: {selected_symbol}"
    chart_dict[spot_lbl] = {
        "symbol": spot_symbol,
        "df": spot_st_stats["df"],
        "stats": spot_st_stats,
    }

# Add near strikes CE and PE
for item in strike_summaries:
    s = item["strike"]
    tag = " (ATM)" if item["is_atm"] else ""
    ce_stats = item["ce_data"].get("stats")
    pe_stats = item["pe_data"].get("stats")

    if ce_stats and ce_stats.get("df") is not None:
        ce_lbl = f"{int(s):,} CE{tag} — {ce_stats['dir_label']} (LTP: ₹{ce_stats['ltp']:.1f})"
        chart_dict[ce_lbl] = {
            "symbol": item["ce_data"]["symbol"],
            "df": ce_stats["df"],
            "stats": ce_stats,
        }
    if pe_stats and pe_stats.get("df") is not None:
        pe_lbl = f"{int(s):,} PE{tag} — {pe_stats['dir_label']} (LTP: ₹{pe_stats['ltp']:.1f})"
        chart_dict[pe_lbl] = {
            "symbol": item["pe_data"]["symbol"],
            "df": pe_stats["df"],
            "stats": pe_stats,
        }

if chart_dict:
    choice_labels = list(chart_dict.keys())
    default_idx = 0 if len(choice_labels) <= 1 else 1
    selected_label = st.selectbox(
        "Select Contract or Spot to Inspect",
        options=choice_labels,
        index=default_idx
    )

    selected_info = chart_dict[selected_label]
    contract_sym = selected_info["symbol"]
    chart_df = selected_info["df"]
    stats_obj = selected_info["stats"]

    if chart_df is not None and not chart_df.empty:
        # Build Plotly figure with Candlesticks + Supertrend Line + Volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25]
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=chart_df["timestamp"],
                open=chart_df["open"],
                high=chart_df["high"],
                low=chart_df["low"],
                close=chart_df["close"],
                name="Price",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="rgba(16, 185, 129, 0.4)",
                decreasing_fillcolor="rgba(239, 68, 68, 0.4)",
            ),
            row=1, col=1
        )

        # Supertrend Bullish (Green Line) and Bearish (Red Line)
        st_bull = np.where(chart_df["supertrend_direction"] == 1, chart_df["supertrend"], np.nan)
        st_bear = np.where(chart_df["supertrend_direction"] == -1, chart_df["supertrend"], np.nan)

        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp"],
                y=st_bull,
                mode="lines",
                line=dict(color="#10b981", width=2.5),
                name="Supertrend (Bullish Support)"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=chart_df["timestamp"],
                y=st_bear,
                mode="lines",
                line=dict(color="#ef4444", width=2.5),
                name="Supertrend (Bearish Resistance)"
            ),
            row=1, col=1
        )

        # Flip Markers (Buy / Sell signals)
        buy_flips = chart_df[chart_df.get("supertrend_signal", 0) == 1]
        sell_flips = chart_df[chart_df.get("supertrend_signal", 0) == -1]

        if not buy_flips.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_flips["timestamp"],
                    y=buy_flips["low"] * 0.99,
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=11, color="#10b981"),
                    text="BUY FLIP",
                    textposition="bottom center",
                    textfont=dict(color="#34d399", size=10),
                    name="Bullish Flip"
                ),
                row=1, col=1
            )

        if not sell_flips.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_flips["timestamp"],
                    y=sell_flips["high"] * 1.01,
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
                    text="SELL FLIP",
                    textposition="top center",
                    textfont=dict(color="#f87171", size=10),
                    name="Bearish Flip"
                ),
                row=1, col=1
            )

        # Volume Bar Subplot
        if "volume" in chart_df.columns:
            vol_colors = np.where(chart_df["close"] >= chart_df["open"], "rgba(16, 185, 129, 0.6)", "rgba(239, 68, 68, 0.6)")
            fig.add_trace(
                go.Bar(
                    x=chart_df["timestamp"],
                    y=chart_df["volume"],
                    marker_color=vol_colors,
                    name="Volume",
                    showlegend=False
                ),
                row=2, col=1
            )

        # Annotate current Supertrend stop level
        if stats_obj:
            curr_st = stats_obj["st_level"]
            curr_ltp = stats_obj["ltp"]
            fig.add_hline(
                y=curr_st,
                line_dash="dot",
                line_color="#fbbf24",
                annotation_text=f"ST Stop: ₹{curr_st:,.2f}",
                annotation_position="right",
                row=1, col=1
            )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(l=30, r=40, t=30, b=30),
            paper_bgcolor="#091120",
            plot_bgcolor="#091120",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1, gridcolor="rgba(255,255,255,0.06)")
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="rgba(255,255,255,0.06)")
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chart data warming up. Select a contract when data arrives.")


# ── Detailed Comparison Table ────────────────────────────────────────────────
st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
st.markdown("### 📋 Multi-Strike Directional Matrix Table")

table_rows = []
for item in strike_summaries:
    s = item["strike"]
    ce_stats = item["ce_data"].get("stats")
    pe_stats = item["pe_data"].get("stats")

    ce_dir_str = ("🟢 BULLISH" if ce_stats["direction"] == 1 else "🔴 BEARISH") if ce_stats else "—"
    pe_dir_str = ("🟢 BULLISH" if pe_stats["direction"] == 1 else "🔴 BEARISH") if pe_stats else "—"

    table_rows.append({
        "Strike": f"{int(s):,}{' ★ ATM' if item['is_atm'] else ''}",
        "Distance (pts)": f"{item['dist_from_spot']:+,.1f}",
        "CE LTP (₹)": f"{ce_stats['ltp']:,.2f}" if ce_stats else "—",
        "CE ST Stop (₹)": f"{ce_stats['st_level']:,.2f}" if ce_stats else "—",
        "CE Direction": ce_dir_str,
        "CE ST Dist %": f"{ce_stats['dist_pct']:+,.1f}%" if ce_stats else "—",
        "CE Trend Bars": ce_stats["trend_bars"] if ce_stats else "—",
        "PE LTP (₹)": f"{pe_stats['ltp']:,.2f}" if pe_stats else "—",
        "PE ST Stop (₹)": f"{pe_stats['st_level']:,.2f}" if pe_stats else "—",
        "PE Direction": pe_dir_str,
        "PE ST Dist %": f"{pe_stats['dist_pct']:+,.1f}%" if pe_stats else "—",
        "PE Trend Bars": pe_stats["trend_bars"] if pe_stats else "—",
        "Net Strike Bias": item["net_bias"]
    })

if table_rows:
    table_df = pd.DataFrame(table_rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)


# ── Institutional Strike Playbook & Confluence Rules ─────────────────────────
st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
with st.expander("📚 Institutional Supertrend Playbook & Confluence Rules", expanded=False):
    st.markdown("""
    #### Institutional Multi-Strike Confluence Framework:
    1. **Trend Confluence (Calls vs Puts)**:
       - **Directional Bullish**: When Near Strikes have **CE in Bullish Supertrend** and **PE in Bearish Supertrend**, option buyers are aggressively paying up for Calls while Put premiums decay. Confluence with Spot Supertrend creates high-probability Call Long entries.
       - **Directional Bearish**: When Near Strikes have **PE in Bullish Supertrend** and **CE in Bearish Supertrend**, Put buyers dominate and Call writers keep caps in place. Confluence with Spot Supertrend creates high-probability Put Long entries.
    2. **Writer Traps & Theta Harvesting**:
       - When **both CE and PE on the ATM strike are Bearish (below their respective Supertrends)**, option writers are harvesting decay from both sides (Short Straddle regime). Do not buy naked options in this regime.
    3. **Volatility Expansion / Breakout**:
       - When **both CE and PE on the ATM strike are Bullish (above their respective Supertrends)**, volatility is expanding exponentially. Straddle buyers or momentum breakout strategies thrive.
    4. **Execution Stop Rules**:
       - For long option positions, the strike's own Supertrend line acts as a **dynamic trailing stop-loss**. Exit immediately if the premium closes below the Supertrend line on the chosen timeframe.
    """)
