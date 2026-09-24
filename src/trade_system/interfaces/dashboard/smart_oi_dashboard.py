import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from trade_system.domains.analysis.application.analysis.smart_oi_analyzer import SmartOIAnalyzer
from trade_system.domains.analysis.application.analysis import pro_oc_analyzer
from trade_system.domains.analysis.application.analysis.market_structure_engine import (
    MarketStructureEngine,
    MarketStructureInfo,
)
from trade_system.domains.analysis.application.analysis.fo_pcr_screener import (
    FOPCRScreener,
    StockPCRInfo,
)
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.shared.config import Settings
from trade_system.interfaces.dashboard.shared_broker import get_cached_broker
from trade_system.domains.advisory.application.agent.option_chain_monitor_agent import OptionChainMonitorAgent
from trade_system.domains.analysis.application.analysis.stockmojo_smart_oi_engine import (
    aggregate_option_snapshots,
    resample_intraday_data,
    detect_price_volume_divergences,
    format_indian_number,
    DivergenceSignal,
)
from trade_system.interfaces.dashboard.autonomous_execution_dashboard import _load_autonomous_state, _get_live_position_manager
from trade_system.domains.trading.application.execution.autonomous_router import AutonomousExecutionRouter
from trade_system.domains.trading.application.execution.execution_engine import ExecutionEngine
import json

LOGGER = logging.getLogger(__name__)

# Options signal color mapping
SIGNAL_STYLES = {
    "long buildup": {"color": "#00d084", "desc": "Put Writing Accumulation (Bullish Positioning)"},
    "short buildup": {"color": "#ff4d6d", "desc": "Call Writing (Bearish Positioning)"},
    "unwind": {"color": "#9b5de5", "desc": "Position Exiting (Unwinding)"},
    "neutral": {"color": "#00b4d8", "desc": "No Clear Institutional Commitment"}
}


def get_broker():
    """Return the shared cached broker instance."""
    return get_cached_broker()


@st.cache_data(ttl=30)
def fetch_live_vix_quote() -> Dict[str, Any]:
    """Fetch live quote for India VIX."""
    try:
        broker = get_broker()
        if not broker:
            return {"ltp": 13.5, "ch": 0.0, "chp": 0.0}
        q = broker.get_quotes(["NSE:INDIAVIX-INDEX"])
        if q and "NSE:INDIAVIX-INDEX" in q:
            val = q["NSE:INDIAVIX-INDEX"]
            ltp = float(val.get("lp", 13.5) if isinstance(val, dict) else getattr(val, "ltp", 13.5))
            ch = float(val.get("ch", 0.0) if isinstance(val, dict) else getattr(val, "ch", 0.0))
            chp = float(val.get("chp", 0.0) if isinstance(val, dict) else getattr(val, "change_percent", getattr(val, "chp", 0.0)))
            return {"ltp": ltp, "ch": ch, "chp": chp}
        return {"ltp": 13.5, "ch": 0.0, "chp": 0.0}
    except Exception as ex:
        LOGGER.warning("Error fetching VIX quote: %s", ex)
        return {"ltp": 13.5, "ch": 0.0, "chp": 0.0}


@st.cache_data(ttl=15)
def fetch_live_fyers_option_chain(symbol: str, strikecount: int = 15) -> Tuple[Optional[pd.DataFrame], float, str]:
    """Fetch live real-time option chain and spot price directly from Fyers."""
    try:
        broker = get_broker()
        if not broker:
            return None, 0.0, ""
        res = broker.fyers.optionchain(data={"symbol": symbol, "strikecount": strikecount})
        if not isinstance(res, dict) or res.get("s") != "ok":
            LOGGER.warning("Fyers live option chain call returned non-ok: %s", res)
            return None, 0.0, ""
        data = res.get("data", {})
        options_chain = data.get("optionsChain", [])
        if not options_chain:
            return None, 0.0, ""
        df = pd.DataFrame(options_chain)
        if "strike" not in df.columns and "strike_price" in df.columns:
            df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce")
        elif "strike_price" not in df.columns and "strike" in df.columns:
            df["strike_price"] = pd.to_numeric(df["strike"], errors="coerce")
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
    except Exception as e:
        LOGGER.error(f"Error fetching live option chain: {e}")
        return None, 0.0, ""


@st.cache_data(ttl=60)
def fetch_live_price_history(symbol: str, resolution: str = "1") -> pd.DataFrame:
    """Fetch recent intraday price candles from Fyers for live VWAP and charts, with CSV/DB fallback."""
    today_str = date.today().strftime("%Y-%m-%d")
    try:
        broker = get_broker()
        if broker:
            df = broker.fetch_history(
                symbol=symbol,
                resolution=resolution,
                range_from=today_str,
                range_to=today_str
            )
            if df is not None and len(df) >= 2:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
                return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        LOGGER.warning(f"Error fetching live price history from broker: {e}")

    # Fallback 1: Local intraday CSV written by live collector
    try:
        clean = symbol.replace(":", "_")
        csv_path = Path("data/fo_historical") / f"{clean}_1min_{today_str}.csv"
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path)
            if not df_csv.empty and "timestamp" in df_csv.columns:
                df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"], format="mixed")
                return df_csv.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        LOGGER.warning(f"Fallback CSV read failed: {e}")

    # Fallback 2: Database ohlcv_1m
    try:
        df_db = load_db_price_data(symbol, today_str)
        if not df_db.empty:
            return df_db
    except Exception as e:
        LOGGER.warning(f"Fallback DB read failed: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_db_dates(symbol: str) -> list[str]:
    """Get all unique dates in the database for option chain snapshots."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            query = text("""
                SELECT DISTINCT date(timestamp) as snapshot_date 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                ORDER BY snapshot_date DESC
            """)
            df = pd.read_sql(query, conn, params={"symbol": symbol})
            return df["snapshot_date"].tolist()
    except Exception as e:
        LOGGER.error(f"Error loading dates: {e}")
        return []


@st.cache_data(ttl=60)
def load_db_snapshots(symbol: str, target_date: str) -> list[tuple[datetime, pd.DataFrame]]:
    """Load all snapshots for a specific symbol and date."""
    try:
        engine = get_engine()
        snapshots = []
        with engine.connect() as conn:
            from sqlalchemy import text
            data_query = text("""
                SELECT * 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                  AND substr(timestamp, 1, 10) = :target_date
                ORDER BY timestamp ASC
            """)
            df = pd.read_sql(data_query, conn, params={"symbol": symbol, "target_date": target_date})
            if df.empty:
                return []
            
            for ts, group in df.groupby("timestamp", sort=False):
                ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                snapshots.append((ts_dt, group))
        return snapshots
    except Exception as e:
        LOGGER.error(f"Error loading snapshots: {e}")
        return []


@st.cache_data(ttl=60)
def load_db_price_data(symbol: str, target_date: str) -> pd.DataFrame:
    """Load 1m OHLCV price data for the selected date."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            query = text("""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_1m 
                WHERE symbol = :symbol 
                  AND substr(timestamp, 1, 10) = :target_date
                ORDER BY timestamp ASC
            """)
            df = pd.read_sql(query, conn, params={"symbol": symbol, "target_date": target_date})
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
    except Exception as e:
        LOGGER.error(f"Error loading price data: {e}")
        return pd.DataFrame()


def _clean_html(html_str: str) -> str:
    """Strip leading and trailing whitespace from each line to prevent markdown parser code-block induction."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines())


def inject_custom_css():
    """Inject custom styles and CSS for status cards, hero command bar, and battlefield profile."""
    st.markdown(_clean_html("""
    <style>
        .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
        div[data-testid="stVerticalBlock"] > div { margin-top: -0.25rem !important; }
        
        /* Hero Command Bar */
        .hero-command-bar {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            padding: 16px 22px;
            border-radius: 12px;
            border: 1px solid #30363d;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            margin-bottom: 20px;
        }

        /* Battlefield Cards */
        .battlefield-card {
            background: #1e2130;
            padding: 14px 16px;
            border-radius: 10px;
            border: 1px solid #30363d;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
            margin-bottom: 12px;
        }
        
        /* Smart Cards */
        .status-card {
            background: #1e2130;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #30363d;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }
        .status-title {
            font-size: 0.8rem;
            color: #8b949e;
            margin-bottom: 4px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .status-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 4px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
        }
        .status-desc {
            font-size: 0.8rem;
            color: #8b949e;
        }
        
        /* Transition Timeline */
        .timeline-item {
            background: #161b22;
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 4px solid #30363d;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
        .timeline-time {
            color: #8b949e;
            font-size: 0.75rem;
            font-weight: bold;
        }
        .timeline-signal {
            font-weight: bold;
            margin-left: 5px;
        }
        .timeline-desc {
            color: #c9d1d9;
            margin-top: 3px;
        }
    </style>
    """), unsafe_allow_html=True)


def calculate_price_vwap(price_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate VWAP columns on the price dataframe."""
    if not price_df.empty:
        df = price_df.copy()
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["cum_vol"] = df["volume"].cumsum()
        df["cum_tp_vol"] = df["tp_vol"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, 1)
        return df
    return price_df


def determine_option_buyer_signal(
    price_df: pd.DataFrame,
    spot_price: float,
    dist_to_call_wall: float,
    dist_to_put_wall: float,
    call_wall: float,
    put_wall: float,
    ce_unwinding_atm: pd.DataFrame,
    pe_unwinding_atm: pd.DataFrame,
    atm_strike: float
) -> tuple[str, str, str]:
    """Determine the verdict, description, and color for Option Buyer Actionable Signal."""
    buyer_verdict = "⚠️ NO TRADE (Rangebound Chop / High Theta Decay)"
    buyer_desc = "Spot is trapped between major option walls. Buying options now will lead to time decay. Wait for a breakout."
    buyer_color = "#8b949e"

    if dist_to_call_wall > 0 and dist_to_call_wall <= 0.6:
        is_vwap_above = True
        if not price_df.empty:
            is_vwap_above = spot_price > price_df["vwap"].iloc[-1]
        
        if is_vwap_above:
            buyer_verdict = "🚀 CALL BUYING ALIGNMENT (Near Call Wall Breakout)"
            buyer_desc = f"Spot is only {dist_to_call_wall:.2f}% below the major Call Wall (₹{call_wall:,.0f}). A breakout above this level will trigger aggressive Call short covering."
            buyer_color = "#00d084"
    elif dist_to_put_wall > 0 and dist_to_put_wall <= 0.6:
        is_vwap_below = True
        if not price_df.empty:
            is_vwap_below = spot_price < price_df["vwap"].iloc[-1]
            
        if is_vwap_below:
            buyer_verdict = "🔥 PUT BUYING ALIGNMENT (Near Put Wall Breakdown)"
            buyer_desc = f"Spot is only {dist_to_put_wall:.2f}% above the major Put Wall (₹{put_wall:,.0f}). A breakdown below this level will trigger rapid Put short covering and panic selling."
            buyer_color = "#ff4d6d"
    elif not ce_unwinding_atm.empty:
        buyer_verdict = "🟢 CALL BUYING ALIGNMENT (Intraday Short Covering)"
        buyer_desc = f"Institutional Call writers are covering positions at the ATM strike ₹{atm_strike:,.0f} (OI Change: {ce_unwinding_atm.iloc[0]['oi_change']:,} contracts). Strong bullish tailwind."
        buyer_color = "#00d084"
    elif not pe_unwinding_atm.empty:
        buyer_verdict = "🔴 PUT BUYING ALIGNMENT (Intraday Put Long Liquidation)"
        buyer_desc = f"Put writers are unwinding/exiting support at the ATM strike ₹{atm_strike:,.0f} (OI Change: {pe_unwinding_atm.iloc[0]['oi_change']:,} contracts). Momentum shifting bearish."
        buyer_color = "#ff4d6d"

    return buyer_verdict, buyer_desc, buyer_color


def render_hero_command_hub(
    symbol_label: str,
    db_symbol: str,
    spot_price: float,
    price_df: pd.DataFrame,
    expiry_str: str,
    dte: int,
    gex_info: dict,
    vix_quote: dict,
    vix_regime: str,
):
    """Render the unified Hero Command Hub with Spot, Range, VWAP, Expiry & Regime Badge."""
    vwap_val = price_df["vwap"].iloc[-1] if not price_df.empty and "vwap" in price_df.columns else spot_price
    day_open = price_df["open"].iloc[0] if not price_df.empty and "open" in price_df.columns else spot_price
    day_high = price_df["high"].max() if not price_df.empty and "high" in price_df.columns else spot_price
    day_low = price_df["low"].min() if not price_df.empty and "low" in price_df.columns else spot_price

    net_chg = spot_price - day_open
    net_chg_pct = (net_chg / day_open * 100) if day_open > 0 else 0.0
    chg_color = "#00d084" if net_chg >= 0 else "#ff4d6d"

    tot_gex = gex_info.get("total_net_gex", 0.0) if gex_info else 0.0

    if tot_gex > 50.0:
        regime_title = "PINNED RANGEBOUND (+GEX / LOW VOLATILITY / THETA BLEED)"
        regime_badge_color = "#00d084"
        regime_desc = "Dealers are Long Gamma: buying dips and selling rallies. Range-bound pinning & mean reversion favored."
    elif tot_gex < -50.0:
        regime_title = "GAMMA ACCELERATION (-GEX / SQUEEZE VELOCITY)"
        regime_badge_color = "#ff4d6d"
        regime_desc = "Dealers are Short Gamma: forced to buy rallies and sell flushes. High-velocity directional breakout moves."
    else:
        regime_title = "TRANSITIONAL CONSOLIDATION (GAMMA FLIP INFLECTION)"
        regime_badge_color = "#ffb703"
        regime_desc = "Market is testing the inflection barrier. Watch for volume expansion above or below the walls."

    st.markdown(_clean_html(f"""
    <div class="hero-command-bar">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    Institutional Order Flow • {db_symbol}
                </div>
                <div style="display:flex; align-items:baseline; gap:12px; margin-top:2px;">
                    <span style="font-size:2.0rem; font-weight:800; color:#f0f6fc; font-family:monospace;">₹{spot_price:,.2f}</span>
                    <span style="font-size:1.1rem; font-weight:bold; color:{chg_color}; font-family:monospace;">
                        {net_chg:+,.2f} ({net_chg_pct:+.2f}%)
                    </span>
                </div>
            </div>
            
            <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap;">
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;">Day Range (H / L)</div>
                    <div style="font-size:0.95rem; font-weight:bold; color:#f0f6fc; font-family:monospace;">₹{day_high:,.0f} — ₹{day_low:,.0f}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;">Session VWAP</div>
                    <div style="font-size:0.95rem; font-weight:bold; color:#ffb703; font-family:monospace;">₹{vwap_val:,.1f}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;">Active Expiry</div>
                    <div style="font-size:0.95rem; font-weight:bold; color:#00e6ff;">{expiry_str or 'Weekly'} ({dte} DTE)</div>
                </div>
            </div>
        </div>

        <div style="margin-top:14px; padding-top:12px; border-top:1px solid #21262d; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; font-weight:bold;">Market Regime:</span>
                <span style="background:{regime_badge_color}22; color:{regime_badge_color}; border:1px solid {regime_badge_color}; padding:3px 10px; border-radius:6px; font-size:0.8rem; font-weight:bold;">
                    {regime_title}
                </span>
            </div>
            <div style="font-size:0.8rem; color:#8b949e;">
                {regime_desc}
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)


def render_writer_battlefield_chart(
    latest_oc: pd.DataFrame,
    spot_price: float,
    max_pain_strike: float,
    call_wall: float,
    put_wall: float,
    strike_step: float,
    snapshots: list,
    pcr: float
):
    """
    Render Pillar 1: The Writer Battlefield Map (Where Maximum Writers Are Seated).
    Mirrored horizontal profile showing Put Writers on Left, Call Writers on Right,
    with Spot, Max Pain, and Call/Put Wall callouts.
    """
    st.markdown("### 🏛️ The Writer Battlefield Map (Where Maximum Writers Are Seated)")
    st.caption("Visualizes where smart money option writers have concentrated their risk capital. Writers defend these levels with delta hedging.")

    if latest_oc is None or latest_oc.empty or spot_price <= 0:
        st.info("Waiting for option chain data to render the Writer Battlefield Map...")
        return

    # 1. Metric Callout Cards above Chart
    dist_call = ((call_wall - spot_price) / spot_price * 100) if call_wall and spot_price else 0.0
    dist_put = ((spot_price - put_wall) / spot_price * 100) if put_wall and spot_price else 0.0
    dist_mp = (spot_price - max_pain_strike)

    ce_oc = latest_oc[latest_oc["option_type"] == "CE"]
    pe_oc = latest_oc[latest_oc["option_type"] == "PE"]
    call_wall_oi = ce_oc.loc[ce_oc["strike"] == call_wall, "oi"].sum() if not ce_oc.empty else 0
    put_wall_oi = pe_oc.loc[pe_oc["strike"] == put_wall, "oi"].sum() if not pe_oc.empty else 0

    pcr_status = "Oversold (Squeeze Risk)" if pcr < 0.7 else ("Overbought (Reversal Risk)" if pcr > 1.3 else "Neutral Zone")
    pcr_color = "#00d084" if pcr < 0.7 else ("#ff4d6d" if pcr > 1.3 else "#ffb703")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="battlefield-card" style="border-left: 5px solid #ff4d6d;">
            <div class="status-title">🔴 Call Wall (Ceiling)</div>
            <div class="status-value" style="color:#ff4d6d; font-size:1.5rem;">₹{call_wall:,.0f}</div>
            <div class="status-desc">OI: <b>{call_wall_oi:,.0f}</b> | <b>+{dist_call:.2f}%</b> above Spot</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="battlefield-card" style="border-left: 5px solid #00d084;">
            <div class="status-title">🟢 Put Wall (Floor)</div>
            <div class="status-value" style="color:#00d084; font-size:1.5rem;">₹{put_wall:,.0f}</div>
            <div class="status-desc">OI: <b>{put_wall_oi:,.0f}</b> | <b>-{dist_put:.2f}%</b> below Spot</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="battlefield-card" style="border-left: 5px solid #a855f7;">
            <div class="status-title">🧲 Max Pain (Expiry Magnet)</div>
            <div class="status-value" style="color:#a855f7; font-size:1.5rem;">₹{max_pain_strike:,.0f}</div>
            <div class="status-desc">Distance from Spot: <b>{dist_mp:+.1f} pts</b></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="battlefield-card" style="border-left: 5px solid {pcr_color};">
            <div class="status-title">⚖️ Put-Call Ratio (PCR)</div>
            <div class="status-value" style="color:{pcr_color}; font-size:1.5rem;">{pcr:.2f}</div>
            <div class="status-desc">{pcr_status}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. View Toggle Controls
    col_mode, col_range = st.columns([1.5, 1])
    with col_mode:
        view_metric = st.radio(
            "Battlefield Perspective",
            options=["Total Open Interest (Writers' Seated Positions)", "Intraday ΔOI (Today's Addition / Covering)"],
            horizontal=True,
            key="battlefield_perspective_mode"
        )
    with col_range:
        num_strikes = st.slider("Strikes Range (± from ATM)", 6, 20, 12, 1, key="battlefield_range_slider")

    # 3. Filter Strikes around ATM
    atm_strike = round(spot_price / strike_step) * strike_step
    min_strike = atm_strike - (num_strikes * strike_step)
    max_strike = atm_strike + (num_strikes * strike_step)

    df_filtered = latest_oc[(latest_oc["strike"] >= min_strike) & (latest_oc["strike"] <= max_strike)].copy()
    if df_filtered.empty:
        df_filtered = latest_oc.copy()

    # Calculate Intraday dOI if needed
    if "oich" in df_filtered.columns and not df_filtered["oich"].isna().all():
        df_filtered["oi_change"] = pd.to_numeric(df_filtered["oich"], errors="coerce").fillna(0)
    elif "oi_change" not in df_filtered.columns or df_filtered["oi_change"].isna().all():
        first_oc = snapshots[0][1] if snapshots else None
        if first_oc is not None and not first_oc.empty:
            first_aligned = first_oc.set_index(["strike", "option_type"])
            curr_aligned = df_filtered.set_index(["strike", "option_type"])
            curr_aligned["oi_change"] = curr_aligned["oi"] - first_aligned["oi"]
            curr_aligned["oi_change"] = curr_aligned["oi_change"].fillna(0)
            df_filtered = curr_aligned.reset_index()
        else:
            df_filtered["oi_change"] = 0

    strikes_sorted = sorted(df_filtered["strike"].unique())

    # Build data arrays
    put_vals = []
    call_vals = []
    put_hover = []
    call_hover = []
    is_doi = "Intraday ΔOI" in view_metric

    for s in strikes_sorted:
        pe_row = df_filtered[(df_filtered["strike"] == s) & (df_filtered["option_type"] == "PE")]
        ce_row = df_filtered[(df_filtered["strike"] == s) & (df_filtered["option_type"] == "CE")]

        pe_oi = pe_row["oi"].iloc[0] if not pe_row.empty and pd.notna(pe_row["oi"].iloc[0]) else 0
        ce_oi = ce_row["oi"].iloc[0] if not ce_row.empty and pd.notna(ce_row["oi"].iloc[0]) else 0
        pe_doi = pe_row["oi_change"].iloc[0] if not pe_row.empty and "oi_change" in pe_row.columns and pd.notna(pe_row["oi_change"].iloc[0]) else 0
        ce_doi = ce_row["oi_change"].iloc[0] if not ce_row.empty and "oi_change" in ce_row.columns and pd.notna(ce_row["oi_change"].iloc[0]) else 0

        if is_doi:
            # Negative x for puts (left side of chart), positive x for calls (right side)
            put_vals.append(- pe_doi)
            call_vals.append(ce_doi)
            put_hover.append(f"<b>Put Strike ₹{int(s):,}</b><br>Put ΔOI: {pe_doi:+,.0f}<br>Total Put OI: {pe_oi:,.0f}")
            call_hover.append(f"<b>Call Strike ₹{int(s):,}</b><br>Call ΔOI: {ce_doi:+,.0f}<br>Total Call OI: {ce_oi:,.0f}")
        else:
            put_vals.append(- pe_oi)
            call_vals.append(ce_oi)
            put_hover.append(f"<b>Put Strike ₹{int(s):,} (Floor)</b><br>Total Put OI: {pe_oi:,.0f}<br>Put ΔOI: {pe_doi:+,.0f}")
            call_hover.append(f"<b>Call Strike ₹{int(s):,} (Ceiling)</b><br>Total Call OI: {ce_oi:,.0f}<br>Call ΔOI: {ce_doi:+,.0f}")

    # Plotly mirrored figure
    fig = go.Figure()

    # Left: Put Writers (Emerald Green)
    put_color = "#00d084" if not is_doi else ["#00d084" if v <= 0 else "#ff9f1c" for v in put_vals]
    fig.add_trace(go.Bar(
        y=strikes_sorted,
        x=put_vals,
        orientation='h',
        name="Put Writers (Floor / Support)",
        marker=dict(color=put_color, line=dict(color="#00f5d4", width=1)),
        hoverinfo="text",
        hovertext=put_hover,
    ))

    # Right: Call Writers (Coral Red)
    call_color = "#ff4d6d" if not is_doi else ["#ff4d6d" if v >= 0 else "#00b4d8" for v in call_vals]
    fig.add_trace(go.Bar(
        y=strikes_sorted,
        x=call_vals,
        orientation='h',
        name="Call Writers (Ceiling / Resistance)",
        marker=dict(color=call_color, line=dict(color="#ff758f", width=1)),
        hoverinfo="text",
        hovertext=call_hover,
    ))

    # Horizontal Line for Spot LTP
    fig.add_hline(
        y=spot_price,
        line_dash="dash",
        line_color="#ffb703",
        line_width=2.5,
        annotation_text=f"🟡 Spot LTP: ₹{spot_price:,.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#ffb703", size=11, family="monospace"),
    )

    # Horizontal Line for Max Pain
    fig.add_hline(
        y=max_pain_strike,
        line_dash="dot",
        line_color="#a855f7",
        line_width=2.5,
        annotation_text=f"🧲 Max Pain: ₹{max_pain_strike:,.0f}",
        annotation_position="bottom left",
        annotation_font=dict(color="#a855f7", size=11, family="monospace"),
    )

    # Annotate Call Wall & Put Wall
    if call_wall in strikes_sorted:
        fig.add_annotation(
            y=call_wall,
            x=call_wall_oi if not is_doi else 0,
            text=f"🔴 Call Wall (Ceiling ₹{int(call_wall):,})",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ff4d6d",
            font=dict(color="#ffffff", size=10),
            bgcolor="#ff4d6d",
            bordercolor="#ffffff",
            borderwidth=1,
            ax=50,
            ay=0
        )

    if put_wall in strikes_sorted:
        fig.add_annotation(
            y=put_wall,
            x=-put_wall_oi if not is_doi else 0,
            text=f"🟢 Put Wall (Floor ₹{int(put_wall):,})",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#00d084",
            font=dict(color="#ffffff", size=10),
            bgcolor="#00d084",
            bordercolor="#ffffff",
            borderwidth=1,
            ax=-50,
            ay=0
        )

    # Calculate tick symmetric bounds
    raw_max = max(max([abs(v) for v in put_vals] or [1]), max([abs(v) for v in call_vals] or [1])) * 1.15
    max_x = max(raw_max, 1000)
    tick_vals = [-max_x, -max_x * 0.5, 0, max_x * 0.5, max_x]
    tick_text = [f"{abs(v)/1e6:.1f}M" if abs(v) >= 1e6 else (f"{abs(v)/1e3:.0f}K" if abs(v) >= 1e3 else f"{abs(v):.0f}") for v in tick_vals]

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=30, r=30, t=30, b=30),
        barmode='overlay',
        xaxis=dict(
            title="◄ Put Writers (Floor / Support)  |  Call Writers (Ceiling / Resistance) ►",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[-max_x, max_x],
            zeroline=True,
            zerolinecolor="#ffffff",
            zerolinewidth=1.5,
            gridcolor="#21262d"
        ),
        yaxis=dict(
            title="Strike Price",
            tickformat="₹%,.0f",
            gridcolor="#21262d"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1
        ),
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True, key="battlefield_mirrored_chart")

    # Bottom Summary Callout
    st.markdown(_clean_html(f"""
    <div style="background:#161b22; padding:12px 18px; border-radius:8px; border:1px solid #30363d; font-size:0.9rem; color:#c9d1d9; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
            🔴 <b>Max Call Writers:</b> Seated at <b style="color:#ff4d6d;">₹{call_wall:,.0f}</b> ({call_wall_oi:,.0f} contracts) — <i>Institutional ceiling betting price stays below.</i>
        </div>
        <div>
            🟢 <b>Max Put Writers:</b> Seated at <b style="color:#00d084;">₹{put_wall:,.0f}</b> ({put_wall_oi:,.0f} contracts) — <i>Institutional floor betting price stays above.</i>
        </div>
        <div>
            🧲 <b>Expiry Magnet (Max Pain):</b> <b style="color:#a855f7;">₹{max_pain_strike:,.0f}</b> — <i>Gravitational anchor where writer profit is maximized.</i>
        </div>
    </div>
    """), unsafe_allow_html=True)


def render_vix_greeks_structure_matrix(
    market_structure: Any,
    gex_info: dict,
    iv_skew: dict,
    pro_atm: dict,
    spot_price: float,
    pcr: float,
    vix_quote: dict,
    dir_analysis: dict
):
    """
    Render Pillar 2: How VIX, Market Structure, and Dealer Greeks (GEX)
    Dictate the Overall Market Structure.
    """
    st.markdown("### ⚡ How VIX, Market Structure & Greeks Dictate The Market")
    st.caption("Smart money positions are governed by volatility regimes and market maker gamma hedging constraints.")

    c_left, c_right = st.columns(2)

    with c_left:
        # Left: Market Structure & Institutional Traps
        amd_phase = dir_analysis.get("amd_phase", "CONSOLIDATION")
        amd_low = dir_analysis.get("amd_range_low", 0.0)
        amd_high = dir_analysis.get("amd_range_high", 0.0)
        amd_action = dir_analysis.get("amd_action", "STAND_ASIDE_ACCUMULATION")
        vwap = dir_analysis.get("vwap") or spot_price
        vwap_status = "BULLISH (> VWAP)" if spot_price >= vwap else "BEARISH (< VWAP)"
        vwap_color = "#00d084" if spot_price >= vwap else "#ff4d6d"

        struct_state = getattr(market_structure, "structure_state", "CHOP_CONSOLIDATION") if market_structure else "CHOP_CONSOLIDATION"
        struct_score = getattr(market_structure, "structure_score", 0) if market_structure else 0
        struct_color = "#00d084" if "BULLISH" in struct_state else ("#ff4d6d" if "BEARISH" in struct_state else "#ffb703")

        atm_delta = getattr(market_structure, "atm_volume_delta", 0) if market_structure else 0
        taker_agg = getattr(market_structure, "taker_aggression", "Neutral") if market_structure else "Neutral"
        wall_mig = getattr(market_structure, "wall_migration", "STABLE") if market_structure else "STABLE"
        exp_phase = getattr(market_structure, "expiry_phase", "NORMAL") if market_structure else "NORMAL"
        dte = getattr(market_structure, "days_to_expiry", 5) if market_structure else 5

        st.markdown(_clean_html(f"""
        <div style="background:#1e2130; padding:20px; border-radius:12px; border:1px solid #30363d; height:100%;">
            <div style="font-size:1.1rem; font-weight:bold; color:#f0f6fc; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                <span>🏛️ Market Structure & Institutional Flow</span>
                <span style="font-size:0.8rem; background:#21262d; color:#58a6ff; padding:3px 8px; border-radius:4px; font-weight:normal;">Price Action Anchor</span>
            </div>
            
            <div style="margin-bottom:14px; padding:12px; background:#161b22; border-radius:8px; border-left:4px solid #58a6ff;">
                <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; font-weight:bold;">📦 Wyckoff / Power-of-3 (AMD) Cycle</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#f0f6fc; margin:3px 0;">{amd_phase}</div>
                <div style="font-size:0.85rem; color:#8b949e;">
                    Accumulation Range: <b style="color:#c9d1d9;">₹{amd_low:,.1f} — ₹{amd_high:,.1f}</b><br>
                    Prescribed Strategy: <b style="color:#ffb703;">{amd_action}</b>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:14px;">
                <div style="background:#161b22; padding:10px 12px; border-radius:8px; border-left:4px solid {vwap_color};">
                    <div style="font-size:0.75rem; color:#8b949e;">VWAP Alignment</div>
                    <div style="font-size:1.0rem; font-weight:bold; color:{vwap_color};">{vwap_status}</div>
                    <div style="font-size:0.8rem; color:#c9d1d9;">VWAP: ₹{vwap:,.1f}</div>
                </div>
                <div style="background:#161b22; padding:10px 12px; border-radius:8px; border-left:4px solid {struct_color};">
                    <div style="font-size:0.75rem; color:#8b949e;">Structure State</div>
                    <div style="font-size:1.0rem; font-weight:bold; color:{struct_color};">{struct_state.replace('_', ' ')}</div>
                    <div style="font-size:0.8rem; color:#c9d1d9;">Score: {struct_score:+d} / 25</div>
                </div>
            </div>

            <div style="padding:12px; background:#161b22; border-radius:8px;">
                <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; font-weight:bold; margin-bottom:4px;">🎯 Institutional Order Flow Forensics</div>
                <div style="font-size:0.85rem; color:#c9d1d9; line-height:1.5;">
                    • <b>ATM Volume Delta:</b> {atm_delta:,d} contracts ({taker_agg})<br>
                    • <b>Wall Migration:</b> {wall_mig}<br>
                    • <b>Expiry Week Phase:</b> {exp_phase} ({dte} DTE)
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with c_right:
        # Right: Volatility & Greeks Engine (How VIX & Greeks dictate the market)
        vix_lvl = vix_quote.get("ltp", getattr(market_structure, "vix_level", 13.5) if market_structure else 13.5)
        vix_chp = vix_quote.get("chp", getattr(market_structure, "vix_change_pct", 0.0) if market_structure else 0.0)
        
        # Determine VIX regime
        if vix_lvl < 12.0:
            vix_regime_label = "COMPLACENT (<12)"
            vix_desc = "Low VIX compresses daily range. Options are cheap, but naked buyers face brutal theta decay without quick follow-through. Pinned action & false breakouts predominate."
            vix_badge_color = "#00d084"
        elif vix_lvl <= 16.0:
            vix_regime_label = "NORMAL (12-16)"
            vix_desc = "Balanced volatility. Range permits systematic directional breakout trends when backed by institutional volume."
            vix_badge_color = "#00b4d8"
        elif vix_lvl <= 22.0:
            vix_regime_label = "ELEVATED (16-22)"
            vix_desc = "Heightened volatility expansion. Option premiums are expensive; fast momentum swings require wider stop losses."
            vix_badge_color = "#ffb703"
        else:
            vix_regime_label = "FEAR SPIKE (>22)"
            vix_desc = "Extreme panic / fear pricing. Wide bid-ask spreads and severe IV crush risk after catalyst events."
            vix_badge_color = "#ff4d6d"

        # GEX
        tot_gex = gex_info.get("total_net_gex", 0.0) if gex_info else 0.0
        gex_flip = gex_info.get("gamma_flip_level", spot_price) if gex_info else spot_price

        if tot_gex > 50.0:
            gex_badge = "+GEX (LONG GAMMA)"
            gex_color = "#00d084"
            gex_impact = "Market Makers are LONG GAMMA. To remain delta-neutral, they sell into rallies and buy into dips. This suppresses volatility, causing mean-reversion, fake breakouts, and pinning between walls."
        elif tot_gex < -50.0:
            gex_badge = "-GEX (SHORT GAMMA)"
            gex_color = "#ff4d6d"
            gex_impact = "Market Makers are SHORT GAMMA. To remain delta-neutral, they must buy rising markets and sell falling markets. This accelerates volatility, fueling explosive short squeezes or liquidation waterfalls."
        else:
            gex_badge = "TRANSITIONAL GAMMA"
            gex_color = "#ffb703"
            gex_impact = f"Price is near the Gamma Flip Pivot (₹{gex_flip:,.0f}). Crosses above this strike dampen volatility; breaks below trigger volatility acceleration."

        # IV Skew & ATM Straddle
        skew_type = iv_skew.get("skew_type", "FLAT / NORMAL") if iv_skew else "FLAT / NORMAL"
        skew_desc = "Institutions paying premium for downside crash protection." if "PUT" in skew_type else ("Call buying euphoria / upside FOMO." if "CALL" in skew_type else "Balanced put/call implied volatility surface.")
        straddle_pr = pro_atm.get("straddle_premium", 0.0) if pro_atm else 0.0
        exp_move = pro_atm.get("expected_move_pts", 0.0) if pro_atm else 0.0

        st.markdown(_clean_html(f"""
        <div style="background:#1e2130; padding:20px; border-radius:12px; border:1px solid #30363d; height:100%;">
            <div style="font-size:1.1rem; font-weight:bold; color:#f0f6fc; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                <span>⚡ How VIX & Greeks Dictate Market Structure</span>
                <span style="font-size:0.8rem; background:#21262d; color:#ffb703; padding:3px 8px; border-radius:4px; font-weight:normal;">Dealer Hedging Engine</span>
            </div>

            <!-- VIX Section -->
            <div style="margin-bottom:14px; padding:12px; background:#161b22; border-radius:8px; border-left:4px solid {vix_badge_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; font-weight:bold;">India VIX Regime</span>
                    <span style="font-size:0.95rem; font-weight:bold; color:{vix_badge_color}; font-family:monospace;">VIX: {vix_lvl:.2f} ({vix_chp:+.2f}%)</span>
                </div>
                <div style="font-size:0.95rem; font-weight:bold; color:#f0f6fc; margin:2px 0;">{vix_regime_label}</div>
                <div style="font-size:0.8rem; color:#c9d1d9; margin-top:2px;">
                    💡 <b>Market Impact:</b> {vix_desc}
                </div>
            </div>

            <!-- GEX Section -->
            <div style="margin-bottom:14px; padding:12px; background:#161b22; border-radius:8px; border-left:4px solid {gex_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; font-weight:bold;">Dealer Gamma Exposure (GEX)</span>
                    <span style="font-size:0.9rem; font-weight:bold; color:{gex_color}; font-family:monospace;">{gex_badge} ({tot_gex:+.1f} M)</span>
                </div>
                <div style="font-size:0.85rem; color:#8b949e; margin-top:2px;">
                    Gamma Flip Strike: <b style="color:#ffb703; font-family:monospace;">₹{gex_flip:,.0f}</b>
                </div>
                <div style="font-size:0.8rem; color:#c9d1d9; margin-top:3px;">
                    💡 <b>Market Impact:</b> {gex_impact}
                </div>
            </div>

            <!-- Skew & Greeks Mini Grid -->
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                <div style="background:#161b22; padding:10px 12px; border-radius:8px;">
                    <div style="font-size:0.75rem; color:#8b949e;">IV Skew Pattern</div>
                    <div style="font-size:0.95rem; font-weight:bold; color:#f0f6fc;">{skew_type}</div>
                    <div style="font-size:0.75rem; color:#8b949e; margin-top:2px;">{skew_desc}</div>
                </div>
                <div style="background:#161b22; padding:10px 12px; border-radius:8px;">
                    <div style="font-size:0.75rem; color:#8b949e;">Expected Expiry Range</div>
                    <div style="font-size:0.95rem; font-weight:bold; color:#00e6ff; font-family:monospace;">±₹{exp_move:,.1f}</div>
                    <div style="font-size:0.75rem; color:#8b949e; margin-top:2px;">ATM Straddle: ₹{straddle_pr:,.1f}</div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)


def render_actionable_trade_blueprint(
    trade_setups: dict,
    buyer_verdict: str,
    buyer_desc: str,
    buyer_color: str,
    spot_price: float,
    vwap: float
):
    """Render a clean, unified Actionable Trade Blueprint replacing multiple redundant cards."""
    intra = trade_setups.get("intraday_setup", {})
    swing = trade_setups.get("swing_setup", {})

    st.markdown("### 🎯 Actionable Trading Blueprint")
    st.caption("Real-time decision matrix synthesizing Writer Walls, Dealer GEX, and Intraday Taker Flow.")

    b_col1, b_col2 = st.columns([1.2, 1])

    with b_col1:
        st.markdown(_clean_html(f"""
        <div style="background:#1e2130; padding:18px; border-radius:12px; border-top:5px solid {intra.get('status_color', '#8b949e')}; border:1px solid #30363d; height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:0.8rem; color:#8b949e; font-weight:600; text-transform:uppercase;">⚡ Primary Intraday Trade</span>
                <span style="font-size:0.8rem; background:#21262d; color:{intra.get('status_color')}; font-weight:bold; padding:2px 8px; border-radius:4px;">Conviction: {intra.get('conviction_score')}</span>
            </div>
            <div style="font-size:1.35rem; font-weight:bold; color:{intra.get('status_color')}; margin-bottom:8px;">
                {intra.get('direction')} — <span style="color:#ffb703;">{intra.get('option_contract')}</span>
            </div>
            <div style="background:#161b22; padding:10px 12px; border-radius:6px; font-size:0.85rem; color:#f0f6fc; margin-bottom:12px;">
                <b>Action Trigger:</b> {intra.get('actionable_trigger')}
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap:8px; font-size:0.8rem; text-align:center; margin-bottom:10px;">
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Entry Zone</span><br><b style="color:#f0f6fc;">{intra.get('entry_zone')}</b></div>
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Stop Loss</span><br><b style="color:#ff4d6d;">{intra.get('stop_loss')}</b></div>
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Target 1</span><br><b style="color:#00d084;">{intra.get('target_1')}</b></div>
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">R : R</span><br><b style="color:#00e6ff;">{intra.get('risk_reward')}</b></div>
            </div>
            <div style="font-size:0.8rem; color:#c9d1d9; line-height:1.4;">
                💡 <b>Rationale:</b> {intra.get('rationale')}
            </div>
        </div>
        """), unsafe_allow_html=True)

    with b_col2:
        st.markdown(_clean_html(f"""
        <div style="background:#1e2130; padding:18px; border-radius:12px; border-top:5px solid {swing.get('status_color', '#8b949e')}; border:1px solid #30363d; height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:0.8rem; color:#8b949e; font-weight:600; text-transform:uppercase;">🌊 Multi-Session Swing Setup</span>
                <span style="font-size:0.8rem; background:#21262d; color:{swing.get('status_color')}; font-weight:bold; padding:2px 8px; border-radius:4px;">Conviction: {swing.get('conviction_score')}</span>
            </div>
            <div style="font-size:1.35rem; font-weight:bold; color:{swing.get('status_color')}; margin-bottom:8px;">
                {swing.get('type')}
            </div>
            <div style="background:#161b22; padding:10px 12px; border-radius:6px; font-size:0.85rem; color:#f0f6fc; margin-bottom:12px;">
                <b>Structure:</b> <span style="color:#00e6ff; font-weight:bold;">{swing.get('recommended_strategy')}</span>
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px; font-size:0.8rem; text-align:center; margin-bottom:10px;">
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Entry Zone</span><br><b style="color:#f0f6fc;">{swing.get('entry_zone')}</b></div>
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Stop Loss</span><br><b style="color:#ff4d6d;">{swing.get('stop_loss')}</b></div>
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Target</span><br><b style="color:#00d084;">{swing.get('target')}</b></div>
            </div>
            <div style="font-size:0.8rem; color:#c9d1d9; line-height:1.4;">
                💡 <b>Rationale:</b> {swing.get('rationale')}
            </div>
        </div>
        """), unsafe_allow_html=True)


def render_stockmojo_smart_oi_terminal(
    symbol_label: str,
    db_symbol: str,
    spot_price: float,
    price_df: pd.DataFrame,
    snapshots: list,
    analyzer: SmartOIAnalyzer
):
    """
    Render 1:1 StockMojo-style Smart OI Terminal:
    - Left Panel (65%): Multi-pane synchronized Price Candles, Volume Delta / Smart OI Delta, and Volume.
    - Right Panel (35%): Put/Call OI Change + PE-CE Net Flow, and Intraday OI vs Vol PCR.
    - Divergence Forensics Banner: Automatic detection and high-conviction alerts.
    """
    from plotly.subplots import make_subplots

    st.markdown("### 🎯 StockMojo Smart OI Terminal")
    st.caption("Institutional Order Flow, Volume Delta & Real-Time Open Interest Divergence Cockpit")

    if price_df is None or price_df.empty:
        st.info("ℹ️ Intraday candlestick price history is not available for this session. Please check broker connection or database historical logs.")
        return

    # Top Controls Bar: Timeframe & Delta Mode
    c_tf, c_mode, c_scope = st.columns([1.6, 2.4, 1.2])
    with c_tf:
        tf = st.radio("⏱️ Timeframe", ["1m", "3m", "5m", "15m"], index=1, horizontal=True, key="sm_terminal_tf")
    with c_mode:
        delta_mode = st.radio(
            "⚡ Delta Mode",
            ["Smart OI Delta", "Candle Volume Delta", "Cumulative Delta (CVD)"],
            index=0,
            horizontal=True,
            key="sm_terminal_delta_mode"
        )
    with c_scope:
        st.markdown(_clean_html(f"""
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(51, 65, 85, 0.4); border-radius: 8px; padding: 6px 12px; text-align: center; margin-top: 10px;">
                <span style="font-size: 11px; color: #94a3b8;">Spot Reference</span><br>
                <b style="font-size: 14px; color: #38bdf8;">₹{spot_price:,.1f}</b>
            </div>
        """), unsafe_allow_html=True)

    # 1. Aggregate and resample data
    snap_df = aggregate_option_snapshots(snapshots)
    aligned_df = resample_intraday_data(price_df, snap_df, timeframe=tf)

    if aligned_df.empty:
        st.warning("⚠️ No aligned price/volume data found for the selected timeframe.")
        return

    # 2. Select delta series
    if delta_mode == "Smart OI Delta":
        delta_col = "smart_oi_delta"
        delta_label = f"Smart OI Delta (ΔPE - ΔCE) [{tf}]"
    elif delta_mode == "Candle Volume Delta":
        delta_col = "candle_volume_delta"
        delta_label = f"Candle Volume Delta [{tf}]"
    else:
        delta_col = "cvd"
        delta_label = "Cumulative Volume Delta (CVD)"

    # 3. Detect Divergences
    divergences = detect_price_volume_divergences(aligned_df, delta_col=delta_col, window=2)

    # Metrics summary
    cur_spot = aligned_df.iloc[-1]["close"]
    open_spot = aligned_df.iloc[0]["open"]
    p_chg = ((cur_spot - open_spot) / open_spot) * 100
    latest_delta = aligned_df.iloc[-1][delta_col]
    latest_net_oi_chg = aligned_df.iloc[-1].get("net_oi_chg", 0.0)
    latest_oi_pcr = aligned_df.iloc[-1].get("oi_pcr", 1.0)
    latest_vol_pcr = aligned_df.iloc[-1].get("vol_pcr", 1.0)

    # Find recent active divergence in last 10 candles
    recent_div: Optional[DivergenceSignal] = None
    if divergences:
        latest_div = divergences[-1]
        div_indices = aligned_df[aligned_df["timestamp"] == latest_div.timestamp].index
        if not div_indices.empty and (len(aligned_df) - 1 - div_indices[0]) <= 10:
            recent_div = latest_div

    # Active Autonomous Position Banner
    auto_state = _load_autonomous_state()
    active_positions = auto_state.get("active_positions", [])
    matching_pos = None
    for p in active_positions:
        p_sym = p.get("symbol", "")
        if (db_symbol and db_symbol in p_sym) or (symbol_label and symbol_label in p_sym):
            matching_pos = p
            break

    if matching_pos:
        p_side = matching_pos.get("direction", "LONG")
        p_qty = matching_pos.get("quantity", 0)
        p_entry = matching_pos.get("entry_price", 0.0)
        p_sl = matching_pos.get("current_stop_loss", 0.0)
        p_tp2 = matching_pos.get("target_2", 0.0)
        p_unrealized = (cur_spot - p_entry) * p_qty if p_side == "LONG" else (p_entry - cur_spot) * p_qty
        p_unrealized_pct = ((cur_spot - p_entry) / p_entry * 100.0) if p_side == "LONG" else ((p_entry - cur_spot) / p_entry * 100.0)
        p_be = matching_pos.get("is_breakeven_locked", False)
        be_badge = "🛡️ BREAKEVEN LOCKED" if p_be else "TARGET 1 HUNTING"
        p_side_color = "#00e676" if p_side == "LONG" else "#ff3b30"

        pos_col1, pos_col2 = st.columns([4.2, 0.8])
        with pos_col1:
            st.markdown(_clean_html(f"""
                <div style="background: linear-gradient(90deg, rgba(0, 230, 118, 0.12), rgba(15, 23, 42, 0.7)); border: 1px solid rgba(0, 230, 118, 0.4); border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 13px; color: #f0f6fc;">
                        ⚡ <b>ACTIVE AUTONOMOUS POSITION:</b> <span style="color: {p_side_color}; font-weight: 800;">{p_side} {p_qty} Qty</span> @ ₹{p_entry:,.1f} &nbsp;|&nbsp;
                        LTP: <b>₹{cur_spot:,.1f}</b> &nbsp;|&nbsp;
                        PnL: <b style="color: {'#00e676' if p_unrealized>=0 else '#ff3b30'};">₹{p_unrealized:+,.2f} ({p_unrealized_pct:+.2f}%)</b> &nbsp;|&nbsp;
                        SL: ₹{p_sl:,.1f} &nbsp;|&nbsp; TP2: ₹{p_tp2:,.1f}
                    </div>
                    <span style="background: rgba(0, 230, 118, 0.25); color: #00e676; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">
                        {be_badge}
                    </span>
                </div>
            """), unsafe_allow_html=True)
        with pos_col2:
            if st.button("🛑 Close Position", key="term_close_pos_btn", use_container_width=True):
                pm = _get_live_position_manager()
                pm.manual_close_position(matching_pos["symbol"], exit_price=cur_spot, reason="TERMINAL_MANUAL_CLOSE")
                st.toast(f"Closed {matching_pos['symbol']} at ₹{cur_spot:,.1f}!", icon="✅")
                st.rerun()

    # Executive Divergence Forensics Banner + Quick Execution Action
    if recent_div:
        is_bearish = recent_div.divergence_type == "BEARISH_DIVERGENCE"
        b_col1, b_col2 = st.columns([4.2, 0.8])
        with b_col1:
            if is_bearish:
                st.markdown(_clean_html(f"""
                    <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <div style="font-weight: 700; color: #ef4444; font-size: 14px; letter-spacing: 0.3px;">🚨 INSTITUTIONAL BEARISH DIVERGENCE (BULL TRAP ALERT)</div>
                            <div style="color: #cbd5e1; font-size: 12.5px; margin-top: 3px;">{recent_div.description}</div>
                        </div>
                        <div style="background: #ef4444; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 11px; text-transform: uppercase;">DISTRIBUTION</div>
                    </div>
                """), unsafe_allow_html=True)
            else:
                st.markdown(_clean_html(f"""
                    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <div style="font-weight: 700; color: #10b981; font-size: 14px; letter-spacing: 0.3px;">🔥 INSTITUTIONAL BULLISH DIVERGENCE (SMART ABSORPTION)</div>
                            <div style="color: #cbd5e1; font-size: 12.5px; margin-top: 3px;">{recent_div.description}</div>
                        </div>
                        <div style="background: #10b981; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 11px; text-transform: uppercase;">ABSORPTION</div>
                    </div>
                """), unsafe_allow_html=True)
        with b_col2:
            btn_txt = "⚡ Auto-Execute Put" if is_bearish else "⚡ Auto-Execute Call"
            btn_type = "secondary" if is_bearish else "primary"
            if st.button(btn_txt, key="exec_div_action_btn", type=btn_type, use_container_width=True):
                pm = _get_live_position_manager()
                engine = ExecutionEngine(broker=pm.broker, risk_manager=pm.risk_manager)
                router = AutonomousExecutionRouter(execution_engine=engine, position_manager=pm, risk_manager=pm.risk_manager)
                sym_full = f"NSE:{symbol_label}-INDEX" if "-INDEX" not in symbol_label else symbol_label
                router.route_divergence_signal(symbol=sym_full, signal=recent_div, current_price=cur_spot)
                st.toast(f"Executed Autonomous Divergence Trade for {sym_full}!", icon="🚀")
                st.rerun()
    else:
        net_flow_color = "#10b981" if latest_net_oi_chg > 0 else ("#ef4444" if latest_net_oi_chg < 0 else "#94a3b8")
        st.markdown(_clean_html(f"""
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(51, 65, 85, 0.3); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between;">
                <span style="color: #94a3b8; font-size: 13px;">
                    🛡️ <b>Trend In-Sync:</b> Price Action & Volume Flow are aligned. Spot: <b>₹{cur_spot:,.1f}</b> ({p_chg:+.2f}%) | Net Writer Flow: <b style="color: {net_flow_color};">{format_indian_number(latest_net_oi_chg)}</b>
                </span>
                <span style="color: #cbd5e1; font-size: 12px;">
                    OI PCR: <b style="color: #00b4d8;">{latest_oi_pcr:.3f}</b> &nbsp;|&nbsp; Vol PCR: <b style="color: #f59e0b;">{latest_vol_pcr:.3f}</b>
                </span>
            </div>
        """), unsafe_allow_html=True)

    # 4. Institutional Grid (65% Left / 35% Right)
    col_left, col_right = st.columns([0.65, 0.35])

    with col_left:
        # 3-Pane Synchronized Plotly Subplots
        fig_left = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=[
                f"<b>{symbol_label} Spot Price & VWAP ({tf})</b>",
                f"<b>{delta_label}</b>",
                f"<b>Traded Volume ({tf})</b>"
            ]
        )

        # Row 1: Candlesticks
        fig_left.add_trace(go.Candlestick(
            x=aligned_df["timestamp"],
            open=aligned_df["open"],
            high=aligned_df["high"],
            low=aligned_df["low"],
            close=aligned_df["close"],
            increasing_line_color="#00d084",
            decreasing_line_color="#ff4d6d",
            increasing_fillcolor="#00d084",
            decreasing_fillcolor="#ff4d6d",
            name="Price"
        ), row=1, col=1)

        # VWAP
        if "vwap" in aligned_df.columns:
            fig_left.add_trace(go.Scatter(
                x=aligned_df["timestamp"],
                y=aligned_df["vwap"],
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                name="VWAP"
            ), row=1, col=1)

        # Spot Reference Line
        fig_left.add_hline(
            y=cur_spot,
            line_dash="dot",
            line_color="#94a3b8",
            line_width=1,
            row=1, col=1
        )

        # Divergence Markers on Row 1
        for d in divergences:
            if d.divergence_type == "BEARISH_DIVERGENCE":
                fig_left.add_trace(go.Scatter(
                    x=[d.timestamp],
                    y=[d.price_level],
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
                    text=["▼ Bearish Div"],
                    textposition="top center",
                    textfont=dict(size=10, color="#ef4444", family="sans-serif"),
                    hoverinfo="text",
                    hovertext=f"⚠️ {d.summary}<br>{d.description}",
                    showlegend=False
                ), row=1, col=1)
            elif d.divergence_type == "BULLISH_DIVERGENCE":
                fig_left.add_trace(go.Scatter(
                    x=[d.timestamp],
                    y=[d.price_level],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=11, color="#10b981"),
                    text=["▲ Bullish Div"],
                    textposition="bottom center",
                    textfont=dict(size=10, color="#10b981", family="sans-serif"),
                    hoverinfo="text",
                    hovertext=f"🔥 {d.summary}<br>{d.description}",
                    showlegend=False
                ), row=1, col=1)

        # Row 2: Delta
        if delta_mode == "Cumulative Delta (CVD)":
            fig_left.add_trace(go.Scatter(
                x=aligned_df["timestamp"],
                y=aligned_df["cvd"],
                line=dict(color="#00f5d4", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 245, 212, 0.12)",
                name="CVD",
                hovertemplate="Time: %{x}<br>CVD: %{y:,.0f}<extra></extra>"
            ), row=2, col=1)
        else:
            delta_vals = aligned_df[delta_col]
            delta_colors = ["#00d084" if v >= 0 else "#ff4d6d" for v in delta_vals]
            fig_left.add_trace(go.Bar(
                x=aligned_df["timestamp"],
                y=delta_vals,
                marker_color=delta_colors,
                name="Delta",
                hovertemplate="Time: %{x}<br>Delta: %{y:,.0f}<extra></extra>"
            ), row=2, col=1)

        fig_left.add_hline(y=0, line_color="#475569", line_width=1, row=2, col=1)

        # Row 3: Volume Histogram
        vol_colors = ["#00d084" if c >= o else "#ff4d6d" for c, o in zip(aligned_df["close"], aligned_df["open"])]
        fig_left.add_trace(go.Bar(
            x=aligned_df["timestamp"],
            y=aligned_df["volume"],
            marker_color=vol_colors,
            name="Volume",
            hovertemplate="Time: %{x}<br>Vol: %{y:,.0f}<extra></extra>"
        ), row=3, col=1)

        fig_left.update_layout(
            height=720,
            template="plotly_dark",
            margin=dict(l=10, r=25, t=30, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            hovermode="x unified",
            paper_bgcolor="rgba(11, 15, 25, 0.7)",
            plot_bgcolor="rgba(15, 23, 42, 0.35)",
        )
        fig_left.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#64748b",
            spikethickness=1,
            spikedash="dot"
        )
        fig_left.update_yaxes(showgrid=True, gridcolor="rgba(51, 65, 85, 0.25)", zeroline=False)
        st.plotly_chart(fig_left, use_container_width=True)

    with col_right:
        # Right Top: Put OI vs Call OI Change & Net PE-CE Flow
        fig_right_top = make_subplots(specs=[[{"secondary_y": True}]])
        fig_right_top.add_trace(go.Scatter(
            x=aligned_df["timestamp"],
            y=aligned_df["pe_oi_chg"],
            line=dict(color="#ff4d6d", width=2),
            name="Put OI (Chg Day)",
            hovertemplate="%{y:,.0f}"
        ), secondary_y=False)
        fig_right_top.add_trace(go.Scatter(
            x=aligned_df["timestamp"],
            y=aligned_df["ce_oi_chg"],
            line=dict(color="#00d084", width=2),
            name="Call OI (Chg Day)",
            hovertemplate="%{y:,.0f}"
        ), secondary_y=False)
        fig_right_top.add_trace(go.Scatter(
            x=aligned_df["timestamp"],
            y=aligned_df["net_oi_chg"],
            line=dict(color="#9b5de5", width=2.5),
            name="PE-CE (Chg Day)",
            hovertemplate="%{y:,.0f}"
        ), secondary_y=True)

        # Highlight latest PE-CE endpoint
        fig_right_top.add_annotation(
            x=aligned_df["timestamp"].iloc[-1],
            y=latest_net_oi_chg,
            yref="y2",
            text=f"<b>{format_indian_number(latest_net_oi_chg)}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#9b5de5",
            bgcolor="#9b5de5",
            font=dict(color="white", size=10, family="monospace"),
            borderpad=3
        )
        fig_right_top.update_layout(
            title="<b>Put vs Call OI (Chg Day) & PE-CE</b>",
            title_font_size=13,
            height=350,
            template="plotly_dark",
            margin=dict(l=10, r=40, t=40, b=10),
            hovermode="x unified",
            paper_bgcolor="rgba(11, 15, 25, 0.7)",
            plot_bgcolor="rgba(15, 23, 42, 0.35)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
        )
        fig_right_top.update_yaxes(title_text="OI Chg", secondary_y=False, showgrid=True, gridcolor="rgba(51, 65, 85, 0.25)")
        fig_right_top.update_yaxes(title_text="PE-CE Diff", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_right_top, use_container_width=True)

        # Right Bottom: OI PCR vs Volume PCR (Sentiment)
        fig_right_bottom = go.Figure()
        fig_right_bottom.add_trace(go.Scatter(
            x=aligned_df["timestamp"],
            y=aligned_df["oi_pcr"],
            line=dict(color="#00b4d8", width=2),
            name="OI PCR",
            hovertemplate="%{y:.3f}"
        ))
        fig_right_bottom.add_trace(go.Scatter(
            x=aligned_df["timestamp"],
            y=aligned_df["vol_pcr"],
            line=dict(color="#f59e0b", width=2),
            name="Vol PCR (Total)",
            hovertemplate="%{y:.3f}"
        ))
        fig_right_bottom.add_hline(y=1.0, line_dash="dot", line_color="#64748b", line_width=1)

        # Latest PCR annotations
        fig_right_bottom.add_annotation(
            x=aligned_df["timestamp"].iloc[-1],
            y=latest_oi_pcr,
            text=f"<b>{latest_oi_pcr:.3f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#00b4d8",
            bgcolor="#00b4d8",
            font=dict(color="white", size=10, family="monospace"),
            borderpad=3
        )
        fig_right_bottom.add_annotation(
            x=aligned_df["timestamp"].iloc[-1],
            y=latest_vol_pcr,
            text=f"<b>{latest_vol_pcr:.3f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#f59e0b",
            bgcolor="#f59e0b",
            font=dict(color="black", size=10, family="monospace"),
            borderpad=3
        )
        fig_right_bottom.update_layout(
            title="<b>OI PCR vs Volume PCR (Sentiment)</b>",
            title_font_size=13,
            height=355,
            template="plotly_dark",
            margin=dict(l=10, r=40, t=40, b=10),
            hovermode="x unified",
            paper_bgcolor="rgba(11, 15, 25, 0.7)",
            plot_bgcolor="rgba(15, 23, 42, 0.35)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
        )
        fig_right_bottom.update_yaxes(title_text="Ratio", showgrid=True, gridcolor="rgba(51, 65, 85, 0.25)")
        st.plotly_chart(fig_right_bottom, use_container_width=True)

    # 5. Multi-Strike OI Migration & Distribution Heatmap
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    _render_multistrike_oi_heatmap(snapshots, cur_spot)


def _render_multistrike_oi_heatmap(snapshots: Any, spot_price: float) -> None:
    """Renders Multi-Strike Net Open Interest Heatmap (PE - CE) across snapshot timestamps."""
    if not snapshots:
        return

    df = pd.DataFrame(snapshots) if isinstance(snapshots, list) else snapshots.copy()
    if df.empty or "strike_price" not in df.columns or "open_interest" not in df.columns:
        return

    with st.expander("🔥 Multi-Strike Net OI Migration Heatmap (Institutional Wall Tracking)", expanded=True):
        st.caption("Visualizes dynamic strike-by-strike Call vs Put OI accumulation over time. Green = Put Support (PE > CE), Red = Call Resistance (CE > PE).")

        # Filter to ±10 strikes around spot
        if "strike_price" in df.columns and spot_price > 0:
            strikes = sorted(df["strike_price"].unique())
            nearest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
            start_idx = max(0, nearest_idx - 10)
            end_idx = min(len(strikes), nearest_idx + 11)
            target_strikes = set(strikes[start_idx:end_idx])
            df = df[df["strike_price"].isin(target_strikes)].copy()

        # Normalize timestamp to HH:MM
        if "timestamp" in df.columns:
            df["time_label"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M")

        # Pivot to get PE OI - CE OI per strike and timestamp
        pe_sub = df[df["option_type"].str.upper() == "PE"]
        ce_sub = df[df["option_type"].str.upper() == "CE"]
        if pe_sub.empty or ce_sub.empty:
            return

        pe_df = pe_sub.groupby(["strike_price", "time_label"])["open_interest"].sum().unstack(fill_value=0)
        ce_df = ce_sub.groupby(["strike_price", "time_label"])["open_interest"].sum().unstack(fill_value=0)

        # Net OI: PE - CE
        net_matrix = pe_df.subtract(ce_df, fill_value=0)
        if net_matrix.empty:
            return

        # Sort strikes ascending
        net_matrix = net_matrix.sort_index(ascending=True)

        fig = go.Figure(data=go.Heatmap(
            z=net_matrix.values,
            x=list(net_matrix.columns),
            y=[f"₹{int(s)}" for s in net_matrix.index],
            colorscale=[
                [0.0, "#ff3b30"],      # Heavy Call Resistance (CE dominates)
                [0.45, "#7f1d1d"],
                [0.5, "#1e293b"],      # Equilibrium
                [0.55, "#064e3b"],
                [1.0, "#00e676"]       # Heavy Put Support (PE dominates)
            ],
            zmid=0,
            colorbar=dict(title="Net OI (PE-CE)", tickfont=dict(size=10)),
            hovertemplate="Strike: %{y}<br>Time: %{x}<br>Net OI: %{z:,.0f}<extra></extra>"
        ))

        # Add Spot Price horizontal marker line
        nearest_strike = min(net_matrix.index, key=lambda s: abs(s - spot_price))
        fig.add_hline(
            y=f"₹{int(nearest_strike)}",
            line_dash="dot",
            line_color="#38bdf8",
            line_width=2,
            annotation_text=f"Spot ₹{spot_price:,.0f}",
            annotation_position="top left",
            annotation_font_color="#38bdf8"
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=40, r=40, t=30, b=30),
            xaxis_title="Intraday Snapshot Time",
            yaxis_title="Option Strike Price",
            paper_bgcolor="rgba(11, 15, 25, 0.7)",
            plot_bgcolor="rgba(15, 23, 42, 0.35)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_tab_chart(price_df: pd.DataFrame, snapshots: list, max_pain_strike: float, analyzer: SmartOIAnalyzer, selected_symbol_label: str):
    """Render the Price Action & Smart OI candlestick overlay."""
    st.subheader("Price vs. Smart OI & technical levels")
    has_candles = price_df is not None and not price_df.empty and "timestamp" in price_df.columns

    if has_candles:
        from plotly.subplots import make_subplots
        
        # Create dual-axis Plotly Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add Candlesticks on primary y-axis
        fig.add_trace(go.Candlestick(
            x=price_df['timestamp'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name="Price"
        ), secondary_y=False)
        
        # Add VWAP Line on primary y-axis
        if "vwap" in price_df.columns:
            fig.add_trace(go.Scatter(
                x=price_df['timestamp'],
                y=price_df['vwap'],
                line=dict(color="orange", width=2, dash="dash"),
                name="VWAP"
            ), secondary_y=False)
        
        # Add Max Pain Horizontal Line
        fig.add_hline(
            y=max_pain_strike,
            line_dash="dot",
            line_color="#9b5de5",
            annotation_text=f"Max Pain (₹{int(max_pain_strike)})",
            annotation_position="bottom right"
        )
        
        # Calculate signals and PE-CE difference over all database snapshots
        signals_ts = []
        for i in range(len(snapshots)):
            ts_item, oc_item = snapshots[i]
            prev_item = snapshots[i-1][1] if i > 0 else None
            
            # Calculate PE - CE difference (Net Option Writer Positioning)
            ce_oi = oc_item[oc_item["option_type"] == "CE"]["oi"].sum()
            pe_oi = oc_item[oc_item["option_type"] == "PE"]["oi"].sum()
            net_oi_diff = pe_oi - ce_oi
            
            # Find close price
            close_item = price_df[price_df["timestamp"] <= ts_item]
            item_spot = close_item.iloc[-1]["close"] if not close_item.empty and "close" in close_item.columns else None
            res_item = analyzer.analyze_smart_oi(oc_item, prev_item, spot_price=item_spot)
            
            # Find matching price row
            price_row = price_df[price_df["timestamp"] <= ts_item]
            row_idx = price_row.index[-1] if not price_row.empty else None
            
            signals_ts.append({
                "timestamp": ts_item,
                "net_oi_diff": net_oi_diff,
                "signal": res_item["signal"],
                "price_idx": row_idx
            })
            
        sig_df = pd.DataFrame(signals_ts)
        
        # Add Net OI Line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=sig_df['timestamp'],
            y=sig_df['net_oi_diff'],
            line=dict(color="#00f5d4", width=2),
            name="Net Writer Flow (PE-CE OI)",
            fill='tozeroy',
            fillcolor='rgba(0, 245, 212, 0.1)'
        ), secondary_y=True)

        fig.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Intraday candlestick price history is not available for this session. Real-time option chain analytics, volume profile, and Greek heatmaps below remain fully active.")


def render_tab_transitions(snapshots: list, price_df: pd.DataFrame, analyzer: SmartOIAnalyzer):
    """Render the intraday signal transitions timeline."""
    st.subheader("Intraday Signal Transitions Timeline")
    st.caption("Transitions show when institutions are adding or closing positions")
    
    # Calculate transition table
    transition_list = []
    last_sig = None
    has_ts = price_df is not None and not price_df.empty and "timestamp" in price_df.columns

    for i in range(len(snapshots)):
        ts_item, oc_item = snapshots[i]
        prev_item = snapshots[i-1][1] if i > 0 else None
        
        item_spot = None
        if has_ts:
            close_item = price_df[price_df["timestamp"] <= ts_item]
            if not close_item.empty and "close" in close_item.columns:
                item_spot = close_item.iloc[-1]["close"]
        
        res_item = analyzer.analyze_smart_oi(oc_item, prev_item, spot_price=item_spot)
        
        current_sig = res_item["signal"]
        if current_sig != last_sig:
            time_str = ts_item.strftime("%H:%M:%S") if hasattr(ts_item, "strftime") else str(ts_item)
            transition_list.append({
                "time": time_str,
                "signal": current_sig,
                "spot": item_spot,
                "reasons": f"Bullish flow: {res_item['summary']['bullish_oi_flow']:,} | Bearish flow: {res_item['summary']['bearish_oi_flow']:,} | Unwinding: {res_item['summary']['unwind_oi_flow']:,}"
            })
            last_sig = current_sig

    if transition_list:
        # Display timeline items
        for t in reversed(transition_list):
            sig_col = SIGNAL_STYLES.get(t["signal"], {"color": "#8b949e"})["color"]
            spot_txt = f" @ Spot ₹{t['spot']:,.2f}" if t["spot"] else ""
            st.markdown(f"""
            <div class="timeline-item" style="border-left-color: {sig_col};">
                <span class="timeline-time">⏱️ {t['time']}</span>
                <span class="timeline-signal" style="color: {sig_col};">{t['signal'].upper()}</span>
                <span style="color: #8b949e;">{spot_txt}</span>
                <div class="timeline-desc">{t['reasons']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No signal transitions detected on this day.")


def render_tab_strikes(signal_strikes: list):
    """Render the active institutional signal strikes table."""
    st.subheader("🎯 Active Institutional Signal Strikes")
    st.caption("Noise-filtered strikes where serious institutional positions are being committed.")
    
    if signal_strikes:
        df_strikes = pd.DataFrame(signal_strikes)
        
        # Beautify column names and formats
        df_strikes = df_strikes.rename(columns={
            "strike": "Strike Price",
            "option_type": "Option Type",
            "ltp": "Option Price (LTP)",
            "oi": "Open Interest (OI)",
            "oi_change": "OI Change (Contracts)",
            "oi_change_pct": "OI Change %",
            "volume": "Interval Volume",
            "buildup": "Buildup Category",
            "action": "Institutional Action"
        })
        
        df_strikes = df_strikes[[
            "Strike Price", "Option Type", "Option Price (LTP)", 
            "Open Interest (OI)", "OI Change (Contracts)", "OI Change %",
            "Interval Volume", "Buildup Category", "Institutional Action"
        ]]
        
        st.dataframe(
            df_strikes,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike Price": st.column_config.NumberColumn("Strike Price", format="₹%d"),
                "Option Price (LTP)": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                "Open Interest (OI)": st.column_config.NumberColumn("OI", format="%d"),
                "OI Change (Contracts)": st.column_config.NumberColumn("OI Change", format="%+d"),
                "OI Change %": st.column_config.NumberColumn("OI Change %", format="%.2f%%"),
                "Interval Volume": st.column_config.NumberColumn("Volume", format="%d"),
            }
        )
    else:
        st.info("No active signal strikes passed the filters at this snapshot.")


def render_institutional_walls(latest_oc: pd.DataFrame, snapshots: list, summary: dict, analyzer: SmartOIAnalyzer, signal_strikes: list):
    """Render the cumulative open interest or fresh buildup walls chart below the tabs."""
    st.subheader("🧱 Institutional Walls: CE vs. PE Open Interest")
    st.caption("Compare cumulative open interest walls or daily fresh positioning (buildup/unwinding) across filtered Smart OI strikes.")
    
    if not latest_oc.empty:
        col_toggle1, col_toggle2 = st.columns(2)
        with col_toggle1:
            view_mode = st.radio(
                "Select Strikes Filter",
                options=["Filtered Smart OI", "All Near-ATM Strikes"],
                horizontal=True,
                help="Filtered Smart OI shows only the high-conviction institutional strikes after filtering noise. All Near-ATM shows raw option chain data."
            )
        with col_toggle2:
            metric_mode = st.radio(
                "Select Metric",
                options=["OI Change (Daily)", "Total Open Interest"],
                horizontal=True,
                help="OI Change shows the net contracts added (buildup) or unwound (exit) since market open. Total Open Interest shows outstanding position walls."
            )

        df_to_plot = latest_oc.copy()
        if metric_mode == "OI Change (Daily)" and snapshots:
            first_oc = snapshots[0][1]
            latest_aligned = df_to_plot.set_index(["strike", "option_type"])
            first_aligned = first_oc.set_index(["strike", "option_type"])
            latest_aligned["oi_change_daily"] = latest_aligned["oi"] - first_aligned["oi"]
            latest_aligned["oi_change_daily"] = latest_aligned["oi_change_daily"].fillna(latest_aligned["oi"])
            df_to_plot = latest_aligned.reset_index()
            y_col = "oi_change_daily"
            yaxis_title = "Change in Open Interest (Contracts)"
            chart_title_metric = "Change in Open Interest (Buildup vs. Unwinding)"
        else:
            y_col = "oi"
            yaxis_title = "Open Interest (Contracts)"
            chart_title_metric = "Total Open Interest Concentration"

        atm_strike = summary["atm_strike"]
        if view_mode == "Filtered Smart OI" and signal_strikes:
            sig_keys = {(s["strike"], s["option_type"]) for s in signal_strikes}
            visual_df = df_to_plot[df_to_plot.apply(lambda row: (row["strike"], row["option_type"]) in sig_keys, axis=1)].copy()
            if visual_df.empty:
                st.info("No active signal strikes passed the filters at this snapshot. Showing all near-ATM strikes.")
                visual_df = df_to_plot[df_to_plot["strike"].apply(lambda s: abs(s - atm_strike) <= 5 * analyzer.strike_step)].copy()
                chart_title_filter = "All Near-ATM Strikes"
            else:
                chart_title_filter = "Filtered Smart OI Strikes"
        else:
            visual_df = df_to_plot[df_to_plot["strike"].apply(lambda s: abs(s - atm_strike) <= 5 * analyzer.strike_step)].copy()
            chart_title_filter = "All Near-ATM Strikes"

        visual_df = visual_df.sort_values("strike")

        fig_walls = go.Figure()
        ce_data = visual_df[visual_df["option_type"] == "CE"]
        pe_data = visual_df[visual_df["option_type"] == "PE"]
        
        fig_walls.add_trace(go.Bar(
            x=ce_data["strike"],
            y=ce_data[y_col],
            name="Call OI (Resistance)",
            marker_color="#ff4d6d"
        ))
        
        fig_walls.add_trace(go.Bar(
            x=pe_data["strike"],
            y=pe_data[y_col],
            name="Put OI (Support)",
            marker_color="#00d084"
        ))
        
        fig_walls.update_layout(
            title=f"{chart_title_filter} — {chart_title_metric}",
            xaxis_title="Strike Price",
            yaxis_title=yaxis_title,
            barmode="group",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_walls, use_container_width=True, key="oi_walls_chart")
    else:
        st.info("Waiting for option chain data to render visual walls...")


def render_tab_pro_trader(
    latest_oc: pd.DataFrame, snapshots: list, spot_price: float, analyzer: SmartOIAnalyzer, prev_oc: pd.DataFrame, summary: dict
):
    """Render the Pro Trader Analytics tab."""
    st.subheader("🏦 Pro Trader Option Chain Analytics")
    st.caption("Institutional-grade breakdown: who is buying, who is selling, where the walls are, and what the premium market is pricing.")

    first_oc = snapshots[0][1] if snapshots else None
    pro_positioning = pro_oc_analyzer.compute_buyer_seller_positioning(
        latest_oc, first_oc, spot_price, analyzer.strike_step
    )
    pro_oi_conc = pro_oc_analyzer.compute_oi_concentration(
        latest_oc, spot_price, analyzer.strike_step
    )
    pro_vol_conc = pro_oc_analyzer.compute_volume_concentration(
        latest_oc, spot_price, analyzer.strike_step
    )
    pro_ce_pe = pro_oc_analyzer.compute_ce_pe_difference(
        latest_oc, spot_price, analyzer.strike_step
    )
    pro_atm = pro_oc_analyzer.compute_atm_premium_analysis(
        latest_oc, spot_price, analyzer.strike_step
    )
    pro_iv = pro_oc_analyzer.compute_iv_skew(
        latest_oc, spot_price, analyzer.strike_step
    )

    # Scoreboard
    st.markdown("### 🥊 Buyer vs Seller Scoreboard")
    st.caption("Aggregate OI flow breakdown — who is adding positions and which side dominates.")

    dom = pro_positioning.get("dominant", "BALANCED")
    dom_colors = {"BUYERS": "#00d084", "SELLERS": "#ff4d6d", "BALANCED": "#ffb703"}
    dom_color = dom_colors.get(dom, "#8b949e")

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #00d084;">
            <div class="status-title">CE Buyers (Long Buildup)</div>
            <div class="status-value" style="color: #00d084; font-size:1.6rem;">{pro_positioning['ce_buyer_oi']:,}</div>
            <div class="status-desc">Contracts added by call buyers</div>
        </div>
        """, unsafe_allow_html=True)
    with pc2:
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #ff4d6d;">
            <div class="status-title">CE Sellers (Short Buildup)</div>
            <div class="status-value" style="color: #ff4d6d; font-size:1.6rem;">{pro_positioning['ce_seller_oi']:,}</div>
            <div class="status-desc">Contracts added by call writers</div>
        </div>
        """, unsafe_allow_html=True)
    with pc3:
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #00b4d8;">
            <div class="status-title">PE Buyers (Long Buildup)</div>
            <div class="status-value" style="color: #00b4d8; font-size:1.6rem;">{pro_positioning['pe_buyer_oi']:,}</div>
            <div class="status-desc">Contracts added by put buyers</div>
        </div>
        """, unsafe_allow_html=True)
    with pc4:
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #9b5de5;">
            <div class="status-title">PE Sellers (Short Buildup)</div>
            <div class="status-value" style="color: #9b5de5; font-size:1.6rem;">{pro_positioning['pe_seller_oi']:,}</div>
            <div class="status-desc">Contracts added by put writers</div>
        </div>
        """, unsafe_allow_html=True)

    # Dominant badge
    st.markdown(f"""
    <div style="background:#1e2130; padding:12px 20px; border-radius:8px; border-left:6px solid {dom_color}; margin:10px 0 20px 0;">
        <span style="color:{dom_color}; font-weight:bold; font-size:1.1rem;">Market Dominant: {dom}</span>
        <span style="color:#8b949e; margin-left:15px;">CE Unwind: {pro_positioning['ce_unwind_oi']:,} | PE Unwind: {pro_positioning['pe_unwind_oi']:,}</span>
    </div>
    """, unsafe_allow_html=True)

    # ATM Premium Analysis
    st.markdown("### 💰 ATM Straddle Premium & Expected Move")
    atm_c1, atm_c2, atm_c3, atm_c4 = st.columns(4)
    with atm_c1:
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #ffb703;">
            <div class="status-title">ATM Strike</div>
            <div class="status-value" style="color: #ffb703; font-size:1.5rem;">₹{pro_atm['atm_strike']:,.0f}</div>
            <div class="status-desc">CE: ₹{pro_atm['ce_ltp']:,.2f} | PE: ₹{pro_atm['pe_ltp']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with atm_c2:
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #00e6ff;">
            <div class="status-title">Straddle Premium</div>
            <div class="status-value" style="color: #00e6ff; font-size:1.5rem;">₹{pro_atm['straddle_premium']:,.2f}</div>
            <div class="status-desc">CE+PE premium at ATM strike</div>
        </div>
        """, unsafe_allow_html=True)
    with atm_c3:
        exp_mv_color = "#ff4d6d" if pro_atm['expected_move_pct'] > 2.0 else ("#ffb703" if pro_atm['expected_move_pct'] > 1.0 else "#00d084")
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid {exp_mv_color};">
            <div class="status-title">Expected Move</div>
            <div class="status-value" style="color: {exp_mv_color}; font-size:1.5rem;">±{pro_atm['expected_move_pct']:.2f}%</div>
            <div class="status-desc">±{pro_atm['expected_move_pts']:,.0f} pts from ATM</div>
        </div>
        """, unsafe_allow_html=True)
    with atm_c4:
        skew_color = "#00d084" if "BULLISH" in pro_atm['premium_skew'] else ("#ff4d6d" if "BEARISH" in pro_atm['premium_skew'] else "#8b949e")
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid {skew_color};">
            <div class="status-title">Premium Skew</div>
            <div class="status-value" style="color: {skew_color}; font-size:1.1rem;">{pro_atm['premium_skew']}</div>
            <div class="status-desc">CE/PE Ratio: {pro_atm['ce_pe_ratio']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Breakeven levels
    st.markdown(f"""
    <div style="background:#161b22; padding:10px 18px; border-radius:8px; margin:5px 0 20px 0; display:flex; justify-content:space-around;">
        <div style="text-align:center;"><span style="color:#8b949e; font-size:0.8rem;">LOWER BREAKEVEN</span><br><span style="color:#ff4d6d; font-weight:bold; font-size:1.1rem;">₹{pro_atm['lower_breakeven']:,.0f}</span></div>
        <div style="text-align:center;"><span style="color:#8b949e; font-size:0.8rem;">ATM IV</span><br><span style="color:#ffb703; font-weight:bold; font-size:1.1rem;">{f"{pro_atm['atm_iv']:.2f}%" if pro_atm['atm_iv'] else 'N/A'}</span></div>
        <div style="text-align:center;"><span style="color:#8b949e; font-size:0.8rem;">ATM OI (CE | PE)</span><br><span style="color:#00b4d8; font-weight:bold; font-size:1.1rem;">{pro_atm['ce_oi']:,} | {pro_atm['pe_oi']:,}</span></div>
        <div style="text-align:center;"><span style="color:#8b949e; font-size:0.8rem;">UPPER BREAKEVEN</span><br><span style="color:#00d084; font-weight:bold; font-size:1.1rem;">₹{pro_atm['upper_breakeven']:,.0f}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # CE - PE OI Difference Chart
    st.markdown("### 📊 CE − PE Open Interest Difference")
    st.caption("Positive = CE OI dominates (bearish wall) | Negative = PE OI dominates (bullish support)")

    diff_data = pro_ce_pe.get("strike_diff", [])
    if diff_data:
        diff_df = pd.DataFrame(diff_data)
        diff_df["color"] = diff_df["oi_diff"].apply(lambda x: "#ff4d6d" if x > 0 else "#00d084")

        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(
            x=diff_df["strike"],
            y=diff_df["oi_diff"],
            marker_color=diff_df["color"],
            name="CE − PE OI",
            hovertemplate="Strike: ₹%{x:,.0f}<br>CE−PE OI: %{y:,}<extra></extra>"
        ))

        fig_diff.add_vline(
            x=pro_atm["atm_strike"],
            line_dash="dash",
            line_color="#ffb703",
            annotation_text=f"ATM ₹{pro_atm['atm_strike']:,.0f}",
            annotation_position="top"
        )

        fig_diff.update_layout(
            title=f"CE − PE OI Difference | Net Bias: {pro_ce_pe['net_bias']}",
            xaxis_title="Strike Price",
            yaxis_title="OI Difference (Contracts)",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_diff, use_container_width=True, key="ce_pe_diff_chart")
    else:
        st.info("No strike data available for CE−PE difference chart.")

    # Concentration tables
    st.markdown("### 🧱 Top OI & Volume Concentration")
    conc_col1, conc_col2 = st.columns(2)
    with conc_col1:
        st.markdown("#### 🔴 Call (CE) Resistance Walls")
        ce_walls = pro_oi_conc.get("ce_walls", [])
        if ce_walls:
            st.dataframe(
                pd.DataFrame(ce_walls),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "strike": st.column_config.NumberColumn("Strike", format="₹%,.0f"),
                    "oi": st.column_config.NumberColumn("Open Interest", format="%,d"),
                    "volume": st.column_config.NumberColumn("Volume", format="%,d"),
                    "ltp": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                }
            )
        else:
            st.info("No CE wall data.")

    with conc_col2:
        st.markdown("#### 🟢 Put (PE) Support Walls")
        pe_walls = pro_oi_conc.get("pe_walls", [])
        if pe_walls:
            st.dataframe(
                pd.DataFrame(pe_walls),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "strike": st.column_config.NumberColumn("Strike", format="₹%,.0f"),
                    "oi": st.column_config.NumberColumn("Open Interest", format="%,d"),
                    "volume": st.column_config.NumberColumn("Volume", format="%,d"),
                    "ltp": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                }
            )
        else:
            st.info("No PE wall data.")

    st.markdown("#### ⚡ Highest Volume Strikes (Intraday Action)")
    vol_col1, vol_col2 = st.columns(2)
    with vol_col1:
        st.caption("🔴 CE — Most Active")
        ce_active = pro_vol_conc.get("ce_active", [])
        if ce_active:
            st.dataframe(
                pd.DataFrame(ce_active),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "strike": st.column_config.NumberColumn("Strike", format="₹%,.0f"),
                    "volume": st.column_config.NumberColumn("Volume", format="%,d"),
                    "oi": st.column_config.NumberColumn("OI", format="%,d"),
                    "ltp": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                }
            )
    with vol_col2:
        st.caption("🟢 PE — Most Active")
        pe_active = pro_vol_conc.get("pe_active", [])
        if pe_active:
            st.dataframe(
                pd.DataFrame(pe_active),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "strike": st.column_config.NumberColumn("Strike", format="₹%,.0f"),
                    "volume": st.column_config.NumberColumn("Volume", format="%,d"),
                    "oi": st.column_config.NumberColumn("OI", format="%,d"),
                    "ltp": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                }
            )

    # ── Big Money Notional Tracker Table ──
    st.markdown("### 💼 Big Money / Large Lot Position Tracker")
    st.caption("Filters out retail lottery tickets to isolate strikes with substantial capital commitment (> ₹50 Lakhs Notional Exposure).")

    big_money_data = pro_oc_analyzer.compute_institutional_big_money(latest_oc, first_oc, spot_price, analyzer.strike_step)
    big_lots = big_money_data.get("big_lot_strikes", [])

    if big_lots:
        big_df = pd.DataFrame(big_lots)
        big_df = big_df.rename(columns={
            "strike": "Strike Price",
            "option_type": "Type",
            "oi": "Total OI",
            "oi_change": "Daily OI Change",
            "ltp": "LTP (₹)",
            "action": "Institutional Action",
            "bias": "Direction Bias",
            "notional_flow_lakhs": "Notional Exposure (₹ Lakhs)",
            "premium_flow_lakhs": "Premium Flow (₹ Lakhs)",
        })
        st.dataframe(
            big_df[["Strike Price", "Type", "LTP (₹)", "Daily OI Change", "Institutional Action", "Direction Bias", "Notional Exposure (₹ Lakhs)", "Premium Flow (₹ Lakhs)"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike Price": st.column_config.NumberColumn("Strike Price", format="₹%d"),
                "LTP (₹)": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                "Daily OI Change": st.column_config.NumberColumn("OI Change", format="%+d"),
                "Notional Exposure (₹ Lakhs)": st.column_config.NumberColumn("Notional Exposure", format="₹%.1f L"),
                "Premium Flow (₹ Lakhs)": st.column_config.NumberColumn("Premium Flow", format="₹%.1f L"),
            }
        )
    else:
        st.info("No single strike reached > ₹50 Lakhs intraday notional exposure threshold yet.")

    # ── Dealer Net GEX Profile Chart ──
    st.markdown("### ⚡ Dealer Net Gamma Exposure (GEX) & Gamma Flip Level")
    st.caption("Dealer Net Gamma reveals whether market makers are dampening volatility (Long Gamma) or accelerating trends (Short Gamma).")

    gex_data = pro_oc_analyzer.compute_gex_profile(latest_oc, spot_price, analyzer.strike_step)
    gex_list = gex_data.get("gex_by_strike", [])
    if gex_list:
        gex_df = pd.DataFrame(gex_list)
        gex_df["color"] = gex_df["net_gex"].apply(lambda x: "#00d084" if x >= 0 else "#ff4d6d")

        fig_gex = go.Figure()
        fig_gex.add_trace(go.Bar(
            x=gex_df["strike"],
            y=gex_df["net_gex"],
            marker_color=gex_df["color"],
            name="Net GEX (M)",
            hovertemplate="Strike: ₹%{x:,.0f}<br>Net GEX: %{y:.2f} M<extra></extra>"
        ))

        gex_flip = gex_data.get("gamma_flip_level", spot_price)
        fig_gex.add_vline(
            x=gex_flip,
            line_dash="dash",
            line_color="#ffb703",
            annotation_text=f"Gamma Flip ₹{gex_flip:,.0f}",
            annotation_position="top"
        )

        fig_gex.update_layout(
            title=f"Net Dealer GEX Profile | Regime: {gex_data.get('gamma_regime')}",
            xaxis_title="Strike Price",
            yaxis_title="Dealer Net GEX (M INR)",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_gex, use_container_width=True, key="dealer_gex_profile_chart")

    st.markdown("---")
    st.markdown("### 🧠 Pro Trader Actionable Insight")
    pro_narrative = pro_oc_analyzer.generate_pro_summary(
        pro_positioning, pro_oi_conc, pro_atm, pro_ce_pe, pro_iv, spot_price
    )
    st.markdown(f"""
    <div style="background:#1e2130; padding:20px 25px; border-radius:12px; border:1px solid #30363d; line-height:1.7;">
    {pro_narrative.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


def render_tab_iv_greeks(latest_oc: pd.DataFrame, spot_price: float, analyzer: SmartOIAnalyzer, pro_atm: dict):
    """Render the IV Skew & Greeks tab."""
    st.subheader("📊 Implied Volatility Skew & Greeks Analysis")
    st.caption("Visualize the volatility surface and risk exposures across strikes.")

    pro_iv_data = pro_oc_analyzer.compute_iv_skew(
        latest_oc, spot_price, analyzer.strike_step
    )
    pro_greeks = pro_oc_analyzer.compute_greeks_heatmap(
        latest_oc, spot_price, analyzer.strike_step
    )

    iv_c1, iv_c2, iv_c3, iv_c4 = st.columns(4)
    with iv_c1:
        atm_iv_val = pro_iv_data.get("atm_iv")
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #ffb703;">
            <div class="status-title">ATM IV</div>
            <div class="status-value" style="color: #ffb703; font-size:1.5rem;">{f"{atm_iv_val:.2f}%" if atm_iv_val else "N/A"}</div>
            <div class="status-desc">At-the-money implied volatility</div>
        </div>
        """, unsafe_allow_html=True)
    with iv_c2:
        otm_put = pro_iv_data.get("avg_otm_put_iv")
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #ff4d6d;">
            <div class="status-title">OTM Put IV (Avg)</div>
            <div class="status-value" style="color: #ff4d6d; font-size:1.5rem;">{f"{otm_put:.2f}%" if otm_put else "N/A"}</div>
            <div class="status-desc">Downside protection cost</div>
        </div>
        """, unsafe_allow_html=True)
    with iv_c3:
        otm_call = pro_iv_data.get("avg_otm_call_iv")
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid #00d084;">
            <div class="status-title">OTM Call IV (Avg)</div>
            <div class="status-value" style="color: #00d084; font-size:1.5rem;">{f"{otm_call:.2f}%" if otm_call else "N/A"}</div>
            <div class="status-desc">Upside speculation cost</div>
        </div>
        """, unsafe_allow_html=True)
    with iv_c4:
        skew_t = pro_iv_data.get("skew_type", "UNKNOWN")
        skew_c = "#ff4d6d" if "PUT" in skew_t else ("#00d084" if "CALL" in skew_t else ("#ffb703" if "SMILE" in skew_t else "#8b949e"))
        st.markdown(f"""
        <div class="status-card" style="border-top: 4px solid {skew_c};">
            <div class="status-title">Skew Pattern</div>
            <div class="status-value" style="color: {skew_c}; font-size:1.0rem;">{skew_t}</div>
            <div class="status-desc">Current volatility surface shape</div>
        </div>
        """, unsafe_allow_html=True)

    # IV curve chart
    st.markdown("### 📈 IV Smile / Skew Curve")
    ce_iv_curve = pro_iv_data.get("ce_iv_curve", [])
    pe_iv_curve = pro_iv_data.get("pe_iv_curve", [])

    if ce_iv_curve or pe_iv_curve:
        fig_iv = go.Figure()
        if ce_iv_curve:
            ce_iv_df = pd.DataFrame(ce_iv_curve)
            fig_iv.add_trace(go.Scatter(
                x=ce_iv_df["strike"],
                y=ce_iv_df["iv"],
                mode="lines+markers",
                name="CE IV",
                line=dict(color="#ff4d6d", width=2.5),
                marker=dict(size=5),
            ))
        if pe_iv_curve:
            pe_iv_df = pd.DataFrame(pe_iv_curve)
            fig_iv.add_trace(go.Scatter(
                x=pe_iv_df["strike"],
                y=pe_iv_df["iv"],
                mode="lines+markers",
                name="PE IV",
                line=dict(color="#00d084", width=2.5),
                marker=dict(size=5),
            ))

        atm_s = pro_atm.get("atm_strike", spot_price)
        fig_iv.add_vline(
            x=atm_s,
            line_dash="dash",
            line_color="#ffb703",
            annotation_text=f"ATM ₹{atm_s:,.0f}",
            annotation_position="top"
        )

        fig_iv.update_layout(
            title="Implied Volatility vs Strike Price",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_iv, use_container_width=True, key="iv_skew_chart")
    else:
        st.info("No IV data available for the selected snapshot.")

    # Greeks heatmap
    st.markdown("### 🔥 Greeks Heatmap (Near-ATM Strikes)")
    st.caption("Delta, Gamma, Theta, Vega for CE and PE side-by-side. ATM row highlighted.")

    heatmap_data = pro_greeks.get("heatmap_data", [])
    if heatmap_data:
        gdf = pd.DataFrame(heatmap_data)
        display_df = gdf[[
            "strike", "ce_ltp", "ce_delta", "ce_gamma", "ce_theta", "ce_vega",
            "pe_ltp", "pe_delta", "pe_gamma", "pe_theta", "pe_vega"
        ]].copy()
        display_df.columns = [
            "Strike", "CE LTP", "CE Δ", "CE Γ", "CE Θ", "CE ν",
            "PE LTP", "PE Δ", "PE Γ", "PE Θ", "PE ν"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn("Strike", format="₹%,.0f"),
                "CE LTP": st.column_config.NumberColumn("CE LTP", format="₹%.2f"),
                "CE Δ": st.column_config.NumberColumn("CE Delta", format="%.4f"),
                "CE Γ": st.column_config.NumberColumn("CE Gamma", format="%.6f"),
                "CE Θ": st.column_config.NumberColumn("CE Theta", format="%.2f"),
                "CE ν": st.column_config.NumberColumn("CE Vega", format="%.2f"),
                "PE LTP": st.column_config.NumberColumn("PE LTP", format="₹%.2f"),
                "PE Δ": st.column_config.NumberColumn("PE Delta", format="%.4f"),
                "PE Γ": st.column_config.NumberColumn("PE Gamma", format="%.6f"),
                "PE Θ": st.column_config.NumberColumn("PE Theta", format="%.2f"),
                "PE ν": st.column_config.NumberColumn("PE Vega", format="%.2f"),
            }
        )
        if not pro_greeks.get("has_meaningful_greeks", False):
            st.warning("⚠️ Greeks values appear uniform — the broker may not be providing granular Greeks data. Delta/Gamma/Theta/Vega accuracy depends on broker feed quality.")
    else:
        st.info("No Greeks data available for the selected snapshot.")

    # Theta decay
    st.markdown("### ⏳ Theta Decay by Strike")
    st.caption("Negative theta = time decay eating premium. Identify max-decay zones favored by option sellers.")

    if heatmap_data:
        theta_df = pd.DataFrame(heatmap_data)
        theta_df = theta_df[theta_df["ce_theta"].notna() | theta_df["pe_theta"].notna()].copy()
        if not theta_df.empty:
            fig_theta = go.Figure()
            fig_theta.add_trace(go.Bar(
                x=theta_df["strike"],
                y=theta_df["ce_theta"],
                name="CE Theta",
                marker_color="#ff4d6d",
            ))
            fig_theta.add_trace(go.Bar(
                x=theta_df["strike"],
                y=theta_df["pe_theta"],
                name="PE Theta",
                marker_color="#00d084",
            ))
            fig_theta.update_layout(
                title="Theta Decay Across Strikes",
                xaxis_title="Strike Price",
                yaxis_title="Theta (₹/day)",
                barmode="group",
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_theta, use_container_width=True, key="theta_decay_chart")
        else:
            st.info("No theta data available.")
    else:
        st.info("No Greeks data available for theta visualization.")


def render_tab_divergence_radar(divergence_data: Dict[str, Any], spot_price: float, max_pain: float, pcr: float, ce_wall: float, pe_wall: float):
    """Render the Institutional Divergence & Market Trap Radar tab."""
    st.subheader("⚡ Institutional Divergence & Market Trap Radar")
    st.caption("Detects where Smart Money positioning contradicts surface retail price action to reveal impending reversals.")

    divergences = divergence_data.get("divergences", [])
    trap_alerts = divergence_data.get("trap_alerts", [])
    atm_delta = divergence_data.get("atm_volume_delta", 0)
    atm_ratio = divergence_data.get("atm_volume_ratio", 1.0)
    aggression = divergence_data.get("taker_aggression", "NEUTRAL")

    # Metric Row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🌊 ATM Volume Delta", f"{atm_delta:+,} contracts", help="Call Volume minus Put Volume across ATM ±1 strikes")
    with c2:
        st.metric("⚔️ Taker Aggression Ratio", f"{atm_ratio:.2f}x", delta="Bullish Takers" if atm_ratio >= 1.5 else ("Bearish Takers" if atm_ratio <= 0.65 else "Balanced"))
    with c3:
        gex_regime = "Positive Gamma (Mean Reverting)" if pcr >= 0.70 and pcr <= 1.25 else "Negative Gamma (Volatility Expansion)"
        st.metric("⚡ Gamma Regime", gex_regime)

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🚨 Detected Market Trap Alerts")
        if trap_alerts:
            for alert in trap_alerts:
                st.error(alert)
        else:
            st.success("✅ No Bull/Bear Traps detected at current key boundary levels.")
            st.caption("Institutional option writers are not heavily fading the current boundary test.")

    with col_right:
        st.markdown("#### 🔄 Price vs OI / PCR Divergences")
        if divergences:
            for div in divergences:
                if "Bullish" in div:
                    st.success(div)
                elif "Bearish" in div:
                    st.error(div)
                else:
                    st.warning(div)
        else:
            st.info("ℹ️ No active price-to-PCR divergences. Flow is aligned with Spot Price.")

    with st.expander("🧠 How Institutional Traps & Divergences Work"):
        st.markdown("""
        * **Bull Trap (Fake Breakout):** Spot breaks above resistance, enticing retail traders to buy calls. However, institutional writers sell calls heavily into the breakout liquidity ($\Delta \text{Call OI} > 0$) rather than unwinding. The breakout fails and collapses.
        * **Bear Trap (Fake Breakdown):** Spot breaks below support, but Put OI increases ($\Delta \text{Put OI} > 0$). Institutional writers absorb the supply, trapping short sellers and triggering a violent short squeeze.
        * **Volume Delta Exhaustion:** Spot reaches a new intraday high while ATM Volume Delta is negative. Indicates lack of aggressive buyers and impending reversal.
        """)


def render_tab_volume_profile(volume_profile_df: pd.DataFrame, spot_price: float, max_pain: float):
    """Render the Strike Volume Profile & Institutional Stickiness tab."""
    st.subheader("📊 Strike Volume Profile & Institutional Stickiness")
    st.caption("Compares Call vs Put volume across strikes and evaluates the Volume-to-OI (V/OI) ratio to distinguish day-trading churn from sticky institutional accumulation.")

    if volume_profile_df is None or volume_profile_df.empty:
        st.info("No volume profile data available.")
        return

    # Dual Bar Charts: OI and Volume Side-by-Side
    col_oi, col_vol = st.columns(2)

    with col_oi:
        st.markdown("##### 🧱 Open Interest Distribution by Strike")
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            y=volume_profile_df["strike"].astype(str),
            x=volume_profile_df["ce_oi"],
            name="Call OI (Resistance)",
            orientation="h",
            marker_color="#ff4d6d"
        ))
        fig_oi.add_trace(go.Bar(
            y=volume_profile_df["strike"].astype(str),
            x=volume_profile_df["pe_oi"],
            name="Put OI (Support)",
            orientation="h",
            marker_color="#00d084"
        ))
        fig_oi.update_layout(
            barmode="group",
            height=520,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_oi, use_container_width=True)

    with col_vol:
        st.markdown("##### 🌊 Traded Volume Profile by Strike")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            y=volume_profile_df["strike"].astype(str),
            x=volume_profile_df["ce_vol"],
            name="Call Volume",
            orientation="h",
            marker_color="#e06666"
        ))
        fig_vol.add_trace(go.Bar(
            y=volume_profile_df["strike"].astype(str),
            x=volume_profile_df["pe_vol"],
            name="Put Volume",
            orientation="h",
            marker_color="#6aa84f"
        ))
        fig_vol.update_layout(
            barmode="group",
            height=520,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # Strike Profile Data Table
    st.markdown("##### 📋 Strike-by-Strike Volume & Stickiness Forensics")
    display_df = volume_profile_df.copy()
    display_df["Marker"] = display_df.apply(
        lambda r: "🎯 ATM" if r["is_atm"] else ("🧲 Max Pain" if r["strike"] == max_pain else ""), axis=1
    )

    styled_df = display_df[[
        "strike", "Marker", "ce_oi", "pe_oi", "net_oi", "ce_vol", "pe_vol", "total_vol", "v_oi_ratio", "stickiness"
    ]].rename(columns={
        "strike": "Strike",
        "ce_oi": "Call OI",
        "pe_oi": "Put OI",
        "net_oi": "Net OI (PE-CE)",
        "ce_vol": "Call Volume",
        "pe_vol": "Put Volume",
        "total_vol": "Total Volume",
        "v_oi_ratio": "V/OI Ratio",
        "stickiness": "Institutional Stickiness"
    })
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_footer_help():
    """Render the explanatory notes at the bottom of the page."""
    st.divider()
    st.markdown("""
    ### 🧠 How to read Nifty Smart OI:
    - **Smart OI shows positioning:** Putting writing accumulation (Put Short Buildup) means institutional traders are selling puts, expecting Nifty to stay above that strike. This is a **Bullish positioning** signal. Call writing accumulation (Call Short Buildup) means institutions are writing calls, creating a **Bearish resistance** wall.
    - **Price Action shows direction:** We compare this positioning with the underlying price relative to **VWAP** and recent highs/lows.
    - **Bullish Confluence:** Put writing accumulating + Price breaking above VWAP/resistance + Volume expanding. Highly reliable long entries.
    - **Bearish Confluence:** Call writing accumulating + Price breaking below VWAP/support + Volume expanding. Highly reliable short entries.
    - **Divergence Handling:**
      - *Rising Price + Bearish Smart OI:* Indicates a false rally or short squeeze with weak institutional buying follow-through. Expect reversals.
      - *Falling Price + Bullish Smart OI:* Indicates a false decline or bull trap. Puts are being aggressively written at lows, expecting a reversal.
    """)


def render_tab_causal_graph(
    symbol: str,
    spot_price: float,
    latest_oc: pd.DataFrame,
    call_wall: float,
    put_wall: float,
    max_pain: float,
    strike_step: float = 50.0,
):
    """Render the In-Memory Option Causal Graph & What-If Shockwave Simulator."""
    st.subheader("🕸️ Option Strike Causal Graph & What-If Shockwave Simulator")
    st.caption(
        "Interactive Directed Graph (DAG) modeling the causal web connecting Spot, India VIX, "
        "heavyweight equities, institutional walls, and dealer gamma rehedging flows."
    )

    from trade_system.domains.analysis.application.analysis.option_causal_graph import OptionCausalGraph

    graph_engine = OptionCausalGraph(symbol=symbol)
    graph_engine.build_graph(
        spot_price=spot_price,
        oc_df=latest_oc,
        vix_level=11.5,
        strike_step=strike_step,
        n_strikes=3,
        call_wall=call_wall,
        put_wall=put_wall,
        max_pain=max_pain,
    )

    # Simulator Controls Card
    with st.expander("⚡ Run What-If Scenario Stress Test (Shockwave Simulator)", expanded=True):
        st.markdown("##### 🎛️ Shockwave Scenario Inputs")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            spot_shock = st.slider("Spot Shock (%)", -2.0, 2.0, 0.0, 0.1, key="shock_spot")
        with c2:
            vix_shock = st.slider("India VIX Shock (%)", -25.0, 25.0, 0.0, 1.0, key="shock_vix")
        with c3:
            hdfc_shock = st.slider("HDFC Bank Shock (%)", -3.0, 3.0, 0.0, 0.2, key="shock_hdfc")
        with c4:
            rel_shock = st.slider("Reliance Shock (%)", -3.0, 3.0, 0.0, 0.2, key="shock_rel")

    # Run simulation
    sim = graph_engine.simulate_shockwave(
        spot_shock_pct=spot_shock,
        vix_shock_pct=vix_shock,
        constituent_shocks={"HDFCBANK": hdfc_shock, "RELIANCE": rel_shock},
    )

    # Shockwave KPI Metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projected Spot", f"₹{sim.projected_spot:,.2f}", f"{sim.spot_delta:+.2f} pts")
    k2.metric("Dealer Futures Rehedge", f"₹{sim.dealer_hedge_flow_cr:,.1f} Cr", sim.dealer_hedge_direction.split()[0])
    k3.metric("Call Wall Status", sim.call_wall_status)
    k4.metric("Put Wall Status", sim.put_wall_status)

    st.markdown("---")

    # Render Plotly Graph
    st.markdown("#### 🌐 Real-Time Interconnection Network Map")
    fig = graph_engine.build_plotly_figure(sim)
    st.plotly_chart(fig, use_container_width=True)

    # Strike Impact Projection Table
    st.markdown("#### 📋 Strike-by-Strike Price & Greeks Sensitivity Matrix")
    if sim.strike_projections:
        table_rows = []
        for s in sim.strike_projections:
            table_rows.append({
                "Strike": f"{int(s.strike)} {s.option_type}",
                "Type": s.option_type,
                "Current LTP": f"₹{s.current_ltp:.2f}",
                "Projected LTP": f"₹{s.projected_ltp:.2f}",
                "Change (₹)": f"{s.ltp_change:+.2f}",
                "Change (%)": f"{s.ltp_change_pct:+.1f}%",
                "Delta (Δ)": f"{s.delta:.2f}",
                "Gamma (Γ)": f"{s.gamma:.4f}",
                "Vega (ν)": f"{s.vega:.2f}",
                "Open Interest": f"{s.oi:,}",
            })
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)


def run_dashboard():
    """Main function to run and render the Smart OI Dashboard."""
    # Main page titles
    st.title("📊 Smart OI & Institutional Option Forensics")
    st.caption("Decodes smart money positioning, volume profile, open interest divergence, and actionable trades across Indices and F&O Stocks.")
    st.markdown("---")

    # Inject Custom CSS styles
    inject_custom_css()

    # Sidebar Controls
    st.sidebar.title("⚙️ Smart OI Controls")
    st.sidebar.markdown("---")

    # 1. Mode Switch: Live vs Database
    mode = st.sidebar.radio(
        "Data Source",
        ["🔴 Live Real-Time (Fyers API)", "📁 Database Snapshots (Historical)"],
        index=0,
        help="Switch between live real-time market data directly from Fyers or historical database snapshots."
    )

    # 2. Asset Class Switch: Indices vs F&O Stocks
    category = st.sidebar.selectbox("Asset Class", ["📊 Major Indices", "🏢 F&O Stocks Universe"], index=0)
    if category == "📊 Major Indices":
        index_map = {
            "NIFTY 50": "NSE:NIFTY50-INDEX",
            "NIFTY BANK": "NSE:NIFTYBANK-INDEX",
            "FINNIFTY": "NSE:FINNIFTY-INDEX",
            "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
            "SENSEX": "BSE:SENSEX-INDEX",
        }
        selected_symbol_label = st.sidebar.selectbox("Select Index", list(index_map.keys()), index=0)
        db_symbol = index_map[selected_symbol_label]
    else:
        try:
            fo_list = get_fo_universe()
        except Exception:
            fo_list = ["NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:KEI-EQ", "NSE:TATAMOTORS-EQ", "NSE:INFY-EQ"]

        display_options = {s.split(":")[-1].replace("-EQ", ""): s for s in sorted(fo_list)}
        selected_stock = st.sidebar.selectbox("Select F&O Stock", list(display_options.keys()), index=0)
        selected_symbol_label = selected_stock
        db_symbol = display_options[selected_stock]

    latest_oc = None
    prev_oc = None
    spot_price = None
    snapshots = []
    price_df = pd.DataFrame()
    expiry_str = ""

    if mode == "🔴 Live Real-Time (Fyers API)":
        col_ref, col_stk = st.sidebar.columns([1.2, 1])
        with col_ref:
            if st.button("🔄 Refresh Live", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col_stk:
            live_strikes = st.slider("Strikes", 10, 30, 15, 5, key="smart_oi_live_strikes")

        with st.spinner(f"Fetching real-time option chain for {selected_symbol_label}..."):
            latest_oc, spot_price, expiry_str = fetch_live_fyers_option_chain(db_symbol, strikecount=live_strikes)
            price_df = fetch_live_price_history(db_symbol, resolution="1")
            if not price_df.empty:
                price_df = calculate_price_vwap(price_df)

            today_str = date.today().strftime("%Y-%m-%d")
            db_snaps = load_db_snapshots(db_symbol, today_str)
            snapshots = list(db_snaps) if db_snaps else []

            if latest_oc is not None and not latest_oc.empty:
                now_ts = datetime.now()
                if not snapshots or (now_ts - snapshots[-1][0]).total_seconds() > 30:
                    snapshots.append((now_ts, latest_oc))
            elif snapshots:
                latest_ts, latest_oc = snapshots[-1]

            prev_oc = snapshots[-2][1] if len(snapshots) > 1 else None
            if (spot_price is None or spot_price == 0.0) and not price_df.empty:
                spot_price = float(price_df.iloc[-1]["close"])
    else:
        available_dates = load_db_dates(db_symbol)
        if not available_dates:
            st.error(f"No option chain data found in database for {selected_symbol_label}. Switch to '🔴 Live Real-Time' to analyze live.")
            st.stop()

        selected_date = st.sidebar.selectbox("Select Analysis Date", available_dates, index=0)
        snapshots = load_db_snapshots(db_symbol, selected_date)
        price_df = load_db_price_data(db_symbol, selected_date)
        if not price_df.empty:
            price_df = calculate_price_vwap(price_df)

        if not snapshots:
            st.warning(f"No snapshots loaded for {selected_symbol_label} on {selected_date}.")
            st.stop()

        latest_ts, latest_oc = snapshots[-1]
        prev_oc = snapshots[-2][1] if len(snapshots) > 1 else None
        if spot_price is None and not price_df.empty:
            spot_price = price_df.iloc[-1]["close"]

    if latest_oc is None or latest_oc.empty:
        st.warning(f"Unable to load Option Chain for {selected_symbol_label}. Please check market hours or broker connection.")
        st.stop()

    # Initialize SmartOI Analyzer & dynamically infer strike step
    analyzer = SmartOIAnalyzer(db_symbol)
    if latest_oc is not None and not latest_oc.empty:
        if "strike" not in latest_oc.columns and "strike_price" in latest_oc.columns:
            latest_oc["strike"] = pd.to_numeric(latest_oc["strike_price"], errors="coerce")
        elif "strike_price" not in latest_oc.columns and "strike" in latest_oc.columns:
            latest_oc["strike_price"] = pd.to_numeric(latest_oc["strike"], errors="coerce")
    if prev_oc is not None and not prev_oc.empty:
        if "strike" not in prev_oc.columns and "strike_price" in prev_oc.columns:
            prev_oc["strike"] = pd.to_numeric(prev_oc["strike_price"], errors="coerce")
        elif "strike_price" not in prev_oc.columns and "strike" in prev_oc.columns:
            prev_oc["strike_price"] = pd.to_numeric(prev_oc["strike"], errors="coerce")

    analyzer.update_strike_step_from_df(latest_oc)

    # Estimate spot price if not available
    if spot_price is None or spot_price == 0.0:
        if "underlying_value" in latest_oc.columns and latest_oc["underlying_value"].iloc[0]:
            spot_price = float(latest_oc["underlying_value"].iloc[0])
        elif not price_df.empty:
            spot_price = float(price_df.iloc[-1]["close"])
        else:
            spot_price = float(latest_oc["strike"].mean())

    # Run Smart OI analysis
    analysis = analyzer.analyze_smart_oi(latest_oc, prev_oc, spot_price=spot_price)
    signal = analysis["signal"]
    signal_strikes = analysis["signal_strikes"]
    summary = analysis["summary"]

    # Compute Divergences & Volume Profile
    divergence_data = analyzer.detect_institutional_divergences(price_df, latest_oc, prev_oc, spot_price=spot_price)
    volume_profile_df = analyzer.compute_strike_volume_profile(latest_oc, spot_price=spot_price)

    # Run Confluence & Divergence detection
    confluence_data = {}
    if not price_df.empty:
        confluence_data = analyzer.detect_confluence_divergence(signal, price_df)

    # Run Market Direction & Actionable Trade Setup Generator
    trade_setups = analyzer.generate_trade_setups(latest_oc, prev_oc, price_df, spot_price=spot_price)
    dir_analysis = trade_setups["directional_analysis"]

    # Compute live VIX quote & Market Structure
    vix_quote = fetch_live_vix_quote()
    market_engine = MarketStructureEngine(broker=get_broker())
    market_structure = market_engine.analyze(
        symbol=db_symbol,
        price_df=price_df,
        oc_df=latest_oc,
        vix_quote=vix_quote,
        spot_price=spot_price
    )

    # Compute Pro Greeks, GEX & ATM Premium
    pro_gex = pro_oc_analyzer.compute_gex_profile(latest_oc, spot_price, analyzer.strike_step)
    pro_iv = pro_oc_analyzer.compute_iv_skew(latest_oc, spot_price, analyzer.strike_step)
    pro_atm = pro_oc_analyzer.compute_atm_premium_analysis(latest_oc, spot_price, analyzer.strike_step)

    # --- WALLS & METRICS CALCULATION ---
    ce_oc = latest_oc[latest_oc["option_type"] == "CE"]
    pe_oc = latest_oc[latest_oc["option_type"] == "PE"]

    call_wall = ce_oc.loc[ce_oc["oi"].idxmax()]["strike"] if not ce_oc.empty and ce_oc["oi"].max() > 0 else 0
    put_wall = pe_oc.loc[pe_oc["oi"].idxmax()]["strike"] if not pe_oc.empty and pe_oc["oi"].max() > 0 else 0

    call_wall_oi = ce_oc.loc[ce_oc["oi"].idxmax()]["oi"] if not ce_oc.empty and ce_oc["oi"].max() > 0 else 0
    put_wall_oi = pe_oc.loc[pe_oc["oi"].idxmax()]["oi"] if not pe_oc.empty and pe_oc["oi"].max() > 0 else 0

    dist_to_call_wall = ((call_wall - spot_price) / spot_price) * 100 if spot_price and call_wall else 0
    dist_to_put_wall = ((spot_price - put_wall) / spot_price) * 100 if spot_price and put_wall else 0

    atm_strike = summary["atm_strike"] if summary.get("atm_strike") else (round(spot_price / analyzer.strike_step) * analyzer.strike_step if spot_price else 0)
    ce_unwinding_atm = ce_oc[(ce_oc["strike"] == atm_strike) & (ce_oc.get("oi_change", 0) < 0)]
    pe_unwinding_atm = pe_oc[(pe_oc["strike"] == atm_strike) & (pe_oc.get("oi_change", 0) < 0)]

    total_ce_oi = latest_oc[latest_oc["option_type"] == "CE"]["oi"].sum()
    total_pe_oi = latest_oc[latest_oc["option_type"] == "PE"]["oi"].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

    # Calculate Max Pain
    strikes = latest_oc["strike"].unique()
    total_pain = []
    for s in strikes:
        pain = 0
        for _, row in latest_oc.iterrows():
            stk = row["strike"]
            oi_val = row["oi"]
            opt_type = row["option_type"]
            if opt_type == "CE" and stk < s:
                pain += (s - stk) * oi_val
            elif opt_type == "PE" and stk > s:
                pain += (stk - s) * oi_val
        total_pain.append(pain)
    max_pain_strike = strikes[np.argmin(total_pain)] if len(total_pain) > 0 else spot_price

    # Determine Option Buyer Signal
    buyer_verdict, buyer_desc, buyer_color = determine_option_buyer_signal(
        price_df, spot_price, dist_to_call_wall, dist_to_put_wall,
        call_wall, put_wall, ce_unwinding_atm, pe_unwinding_atm, atm_strike
    )

    # ── HERO COMMAND HUB ──────────────────────────────────────────
    render_hero_command_hub(
        selected_symbol_label,
        db_symbol,
        spot_price,
        price_df,
        expiry_str,
        getattr(market_structure, 'days_to_expiry', 5),
        pro_gex,
        vix_quote,
        getattr(market_structure, 'vix_regime', 'NORMAL')
    )

    # ── PILLAR 1: THE WRITER BATTLEFIELD MAP ─────────────────────
    render_writer_battlefield_chart(
        latest_oc=latest_oc,
        spot_price=spot_price,
        max_pain_strike=max_pain_strike,
        call_wall=call_wall,
        put_wall=put_wall,
        strike_step=analyzer.strike_step,
        snapshots=snapshots,
        pcr=pcr
    )

    st.markdown("---")

    # ── PILLAR 2: HOW VIX & GREEKS DICTATE MARKET STRUCTURE ──────
    render_vix_greeks_structure_matrix(
        market_structure=market_structure,
        gex_info=pro_gex,
        iv_skew=pro_iv,
        pro_atm=pro_atm,
        spot_price=spot_price,
        pcr=pcr,
        vix_quote=vix_quote,
        dir_analysis=dir_analysis
    )

    st.markdown("---")

    # ── PILLAR 3: ACTIONABLE TRADING BLUEPRINT ────────────────────
    render_actionable_trade_blueprint(
        trade_setups=trade_setups,
        buyer_verdict=buyer_verdict,
        buyer_desc=buyer_desc,
        buyer_color=buyer_color,
        spot_price=spot_price,
        vwap=dir_analysis.get("vwap") or spot_price
    )

    st.markdown("---")

    # ── 6 CONSOLIDATED DEEP-DIVE TABS ─────────────────────────────
    tab_stockmojo, tab_delta, tab_chain, tab_chart, tab_causal, tab_fo = st.tabs([
        "🎯 StockMojo Smart OI (Price, Delta & Flow)",
        "⚡ Dynamic ΔOI & Writer Trap Monitor",
        "📋 Full Option Chain & Greeks Grid",
        "📈 Price Action, VWAP & Net Writer Flow",
        "🕸️ Causal Graph & Volume Shockwave",
        "🎲 F&O Universe PCR Heatmap"
    ])

    with tab_stockmojo:
        render_stockmojo_smart_oi_terminal(selected_symbol_label, db_symbol, spot_price, price_df, snapshots, analyzer)

    with tab_delta:
        render_tab_delta_monitor(selected_symbol_label, db_symbol, spot_price, latest_oc, snapshots, analyzer)

    with tab_chain:
        render_tab_pro_trader(latest_oc, snapshots, spot_price, analyzer, prev_oc, summary)
        st.markdown("---")
        render_tab_iv_greeks(latest_oc, spot_price, analyzer, pro_atm)
        st.markdown("---")
        render_institutional_walls(latest_oc, snapshots, summary, analyzer, signal_strikes)

    with tab_chart:
        render_tab_chart(price_df, snapshots, max_pain_strike, analyzer, selected_symbol_label)
        st.markdown("---")
        render_tab_transitions(snapshots, price_df, analyzer)

    with tab_causal:
        render_tab_causal_graph(
            symbol=db_symbol,
            spot_price=spot_price,
            latest_oc=latest_oc,
            call_wall=call_wall,
            put_wall=put_wall,
            max_pain=max_pain_strike,
            strike_step=analyzer.strike_step,
        )
        st.markdown("---")
        render_tab_volume_profile(volume_profile_df, spot_price, max_pain_strike)
        st.markdown("---")
        render_tab_divergence_radar(divergence_data, spot_price, max_pain_strike, pcr, call_wall, put_wall)

    with tab_fo:
        render_tab_fo_pcr()

    # Render footer information help section
    render_footer_help()


def render_tab_sniper(snapshots: list, spot_price: float, analyzer: SmartOIAnalyzer):
    """Render the Sniper Reversal strategy scanner."""
    st.subheader("🎯 Sniper Reversals (Volatility Contraction & Breakout)")
    st.caption("Scans for sideways premium + volume dry-up, followed by an explosive volume-backed reversal.")

    if not snapshots or len(snapshots) < 6:
        st.info(f"Not enough data to scan for Sniper Reversals (Have {len(snapshots)} snapshots, need at least 6).")
        return
        
    if spot_price is None:
        st.warning("Waiting for spot price data to initialize the Sniper Reversal scanner...")
        return

    try:
        # 1. Identify ATM Strike
        atm_strike = round(spot_price / analyzer.strike_step) * analyzer.strike_step
        
        # We will scan ATM and ±1 strikes
        target_strikes = [atm_strike - analyzer.strike_step, atm_strike, atm_strike + analyzer.strike_step]
        
        try:
            from trade_system.domains.analysis.application.analysis.sniper_reversal import OptionOISniperScanner
        except ImportError:
            st.error("Sniper Reversal scanner not found.")
            return

        scanner = OptionOISniperScanner(lookback_window=5, breakout_vol_multiplier=1.5)
        
        results = []
        all_scans = []
        
        for strike in target_strikes:
            for opt_type in ["CE", "PE"]:
                # Build chronological series for this strike/type
                series_data = []
                for ts, df in snapshots:
                    row = df[(df["strike"] == strike) & (df["option_type"] == opt_type)]
                    if not row.empty:
                        data = row.iloc[0].to_dict()
                        data["timestamp"] = ts
                        series_data.append(data)
                
                if len(series_data) >= 6:
                    series_df = pd.DataFrame(series_data)
                    res = scanner.detect(series_df)
                    res["strike"] = strike
                    res["option_type"] = opt_type
                    
                    if res["status"] != "NEUTRAL":
                        results.append(res)
                    else:
                        all_scans.append(res)
        
        if not results:
            st.success("No active Sniper Reversal patterns detected at the moment.")
            st.caption("The algorithm is actively scanning the ATM and near-ATM strikes for volume contraction and sudden premium/volume spikes.")
        else:
            st.markdown("### 🔥 Detected Reversals")
            for res in results:
                color = "#00d084" if "BULLISH" in res["status"] else "#ff4d6d"
                st.markdown(f'''
                <div class="card" style="border-left: 5px solid {color}; padding: 15px; margin-bottom: 10px;">
                    <h4 style="color: {color}; margin-top: 0;">{res['strike']} {res['option_type']} - {res['status']}</h4>
                    <div style="display: flex; gap: 20px;">
                        <div><b>Trigger LTP:</b> ₹{res.get('trigger_ltp', 'N/A')}</div>
                        <div><b>Premium Surge:</b> {res.get('ltp_surge_pct', 0)}%</div>
                        <div><b>Vol Expansion:</b> {res.get('vol_expansion_ratio', 0)}x</div>
                        <div><b>OI Change:</b> {res.get('oi_change_pct', 0)}%</div>
                        <div><b>Vol Dry-up Slope:</b> {res.get('vol_slope', 0)}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
        with st.expander("🔍 View Live Scanner Status (Why no signal?)"):
            st.write("Here is what the scanner sees under the hood for the active strikes:")
            for scan in all_scans:
                st.markdown(f"**{scan['strike']} {scan['option_type']}**: {scan['reason']}")
                
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in Sniper Reversal scanner: {e}")
        st.error("The Sniper Reversal scanner encountered an issue. Please check the logs.")


def render_tab_fo_pcr():
    """Render the F&O Universe PCR & Overbought/Oversold Scanner in Smart OI Dashboard."""
    st.subheader("🎲 F&O Stock Put-Call Ratio (PCR) & Overbought / Oversold Radar")
    st.caption("Live Option Chain Analysis across ~200 F&O Stocks • Contrarian Short Squeezes & Reversal Risks")

    col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1.0])
    with col_p1:
        pcr_ob = st.slider("Overbought Threshold (>=)", 0.70, 1.50, 0.85, 0.05, key="smart_oi_pcr_ob")
    with col_p2:
        pcr_os = st.slider("Oversold Threshold (<=)", 0.30, 0.70, 0.55, 0.05, key="smart_oi_pcr_os")
    with col_p3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Scan Universe PCR", key="btn_smart_oi_pcr_scan", use_container_width=True)

    if "smart_oi_pcr_results" not in st.session_state or refresh_btn:
        with st.spinner("Fetching option chains across F&O stocks..."):
            try:
                screener = FOPCRScreener(overbought_threshold=pcr_ob, oversold_threshold=pcr_os)
                st.session_state["smart_oi_pcr_results"] = screener.scan_universe_pcr(max_symbols=120)
            except Exception as ex:
                st.error(f"Error scanning F&O PCR: {ex}")
                st.session_state["smart_oi_pcr_results"] = {"overbought": [], "oversold": [], "neutral": [], "all": []}

    res_data = st.session_state.get("smart_oi_pcr_results", {})
    all_stocks = res_data.get("all", [])

    if not all_stocks:
        st.info("No PCR data available. Click '🔄 Scan Universe PCR' to scan.")
        return

    ob_list = [s for s in all_stocks if s.pcr_oi >= pcr_ob]
    os_list = [s for s in all_stocks if s.pcr_oi <= pcr_os]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🔴 Overbought (Put Heavy)", len(ob_list), help=f"PCR >= {pcr_ob:.2f}")
    with k2:
        st.metric("🟢 Oversold (Call Heavy / Squeeze)", len(os_list), help=f"PCR <= {pcr_os:.2f}")
    with k3:
        extreme_count = sum(1 for s in all_stocks if s.pcr_oi >= 1.00 or s.pcr_oi <= 0.45)
        st.metric("💎 Extreme Setups", extreme_count)
    with k4:
        avg_val = np.mean([s.pcr_oi for s in all_stocks]) if all_stocks else 0.0
        st.metric("⚖️ Average F&O PCR", f"{avg_val:.2f}")

    sub_t1, sub_t2, sub_t3 = st.tabs([
        f"🟢 Oversold / Squeeze Candidates ({len(os_list)})",
        f"🔴 Overbought / Reversal Risk ({len(ob_list)})",
        f"📋 All Stocks ({len(all_stocks)})"
    ])

    def _render_table(s_list: list, key: str):
        if not s_list:
            st.info("No stocks matching.")
            return
        df = pd.DataFrame([s.to_dict() for s in s_list])
        cols = ["clean_symbol", "sector", "spot_price", "pcr_oi", "pcr_volume", "sentiment_state", "total_call_oi", "total_put_oi", "max_pain_strike", "highest_ce_oi_strike", "highest_pe_oi_strike", "contrarian_bias"]
        renamed = df[cols].rename(columns={
            "clean_symbol": "Symbol",
            "sector": "Sector",
            "spot_price": "Spot LTP",
            "pcr_oi": "PCR (OI)",
            "pcr_volume": "PCR (Vol)",
            "sentiment_state": "Regime",
            "total_call_oi": "Call OI",
            "total_put_oi": "Put OI",
            "max_pain_strike": "Max Pain",
            "highest_ce_oi_strike": "CE Wall",
            "highest_pe_oi_strike": "PE Wall",
            "contrarian_bias": "Contrarian Bias"
        })
        st.dataframe(
            renamed.style.format({
                "Spot LTP": "₹{:.2f}",
                "PCR (OI)": "{:.2f}",
                "PCR (Vol)": "{:.2f}",
                "Call OI": "{:,.0f}",
                "Put OI": "{:,.0f}",
                "Max Pain": "₹{:.1f}",
                "CE Wall": "₹{:.1f}",
                "PE Wall": "₹{:.1f}",
            }).map(
                lambda v: "background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight:700" if isinstance(v, (int, float)) and v <= 0.55 else ("background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight:700" if isinstance(v, (int, float)) and v >= 0.85 else ""),
                subset=["PCR (OI)"]
            ),
            use_container_width=True,
            hide_index=True,
            key=key
        )

    with sub_t1:
        _render_table(os_list, "smart_oi_pcr_os_tbl")
    with sub_t2:
        _render_table(ob_list, "smart_oi_pcr_ob_tbl")
    with sub_t3:
        _render_table(all_stocks, "smart_oi_pcr_all_tbl")


def render_tab_delta_monitor(
    symbol_label: str,
    db_symbol: str,
    spot_price: float,
    latest_oc: pd.DataFrame,
    snapshots: list,
    analyzer: SmartOIAnalyzer,
):
    """Render Dynamic Option Chain Data Change Forensics Tab."""
    st.markdown("### ⚡ Dynamic Option Chain Data Change Forensics")
    st.caption(
        "Autonomous real-time tracking of strike-by-strike ΔOI velocity (dOI/dt), Max Pain migration drift, "
        "and institutional writer traps vs genuine short squeeze breakouts."
    )

    clean_sym = db_symbol.replace("NSE:", "").replace("BSE:", "").replace("-INDEX", "").replace("-EQ", "")

    # 1. Load live persisted agent state or compute on the fly
    settings = Settings.load()
    state_file = settings.data_dir / "option_chain_monitor_state.json"
    res_data = None
    if state_file.exists():
        try:
            with open(state_file) as f:
                all_states = json.load(f)
                res_data = all_states.get(clean_sym)
                # Discard if stale (from previous days)
                if res_data and res_data.get("timestamp"):
                    ts_str = str(res_data["timestamp"])[:10]
                    if ts_str != date.today().strftime("%Y-%m-%d"):
                        res_data = None
        except Exception:
            pass

    if not res_data and snapshots:
        try:
            agent = OptionChainMonitorAgent(enable_telegram=False)
            dyn_res = None
            for s_ts, s_df in snapshots[-12:]:
                if s_df is not None and not s_df.empty:
                    s_spot = spot_price
                    if "underlying_value" in s_df.columns:
                        u_vals = s_df["underlying_value"].dropna()
                        if not u_vals.empty and float(u_vals.iloc[0]) > 0:
                            s_spot = float(u_vals.iloc[0])
                    dyn_res = agent.process_snapshot(
                        db_symbol,
                        s_df,
                        spot_price=s_spot,
                        timestamp=pd.to_datetime(s_ts)
                    )
            if dyn_res:
                from dataclasses import asdict
                res_data = asdict(dyn_res)
                res_data["timestamp"] = dyn_res.timestamp.isoformat()
        except Exception as e:
            LOGGER.warning("Could not compute on-the-fly dynamic OC forensics: %s", e)

    # Baseline fallback if res_data still None
    if not res_data and latest_oc is not None and not latest_oc.empty:
        ce_oc = latest_oc[latest_oc["option_type"] == "CE"]
        pe_oc = latest_oc[latest_oc["option_type"] == "PE"]
        cw = float(ce_oc.loc[ce_oc["oi"].idxmax()]["strike"]) if not ce_oc.empty and ce_oc["oi"].max() > 0 else spot_price
        pw = float(pe_oc.loc[pe_oc["oi"].idxmax()]["strike"]) if not pe_oc.empty and pe_oc["oi"].max() > 0 else spot_price
        tot_ce = ce_oc["oi"].sum()
        tot_pe = pe_oc["oi"].sum()
        pcr_val = float(tot_pe / tot_ce) if tot_ce > 0 else 1.0

        strikes = latest_oc["strike"].unique()
        total_pain = []
        for s in strikes:
            pain = 0
            for _, row in latest_oc.iterrows():
                stk = row["strike"]
                oi_val = row["oi"]
                opt_type = row["option_type"]
                if opt_type == "CE" and stk < s:
                    pain += (s - stk) * oi_val
                elif opt_type == "PE" and stk > s:
                    pain += (stk - s) * oi_val
            total_pain.append(pain)
        mp_val = float(strikes[np.argmin(total_pain)]) if len(total_pain) > 0 else spot_price

        res_data = {
            "symbol": db_symbol,
            "timestamp": datetime.now().isoformat(),
            "spot_price": spot_price,
            "max_pain": mp_val,
            "prev_max_pain": mp_val,
            "max_pain_shifted": False,
            "max_pain_shift_pts": 0.0,
            "ce_wall": cw,
            "pe_wall": pw,
            "ce_wall_shifted": False,
            "pe_wall_shifted": False,
            "pcr_oi": pcr_val,
            "pcr_velocity": 0.0,
            "top_ce_build_strike": cw,
            "top_ce_build_oi": int(ce_oc["oi"].max()) if not ce_oc.empty else 0,
            "top_pe_build_strike": pw,
            "top_pe_build_oi": int(pe_oc["oi"].max()) if not pe_oc.empty else 0,
            "regime": "BALANCED_ACCUMULATION",
            "traps": [],
            "squeezes": [],
            "signals": [],
        }

    # 2. Render Forensic Status Bar
    if res_data:
        mp = res_data.get("max_pain", 0.0)
        prev_mp = res_data.get("prev_max_pain", mp)
        mp_shift = res_data.get("max_pain_shift_pts", 0.0)
        pcr = res_data.get("pcr_oi", 1.0)
        pcr_vel = res_data.get("pcr_velocity", 0.0)
        top_ce_build = res_data.get("top_ce_build_strike", 0.0)
        top_ce_oi = res_data.get("top_ce_build_oi", 0)
        top_pe_build = res_data.get("top_pe_build_strike", 0.0)
        top_pe_oi = res_data.get("top_pe_build_oi", 0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            shift_text = f"Shifted {mp_shift:+.0f} pts" if mp_shift != 0 else "Pinned / Steady"
            shift_color = "#00d084" if mp_shift > 0 else ("#ff4d6d" if mp_shift < 0 else "#8b949e")
            st.markdown(_clean_html(f"""
                <div class="status-card">
                    <div class="status-title">🧲 Max Pain Anchor</div>
                    <div class="status-value" style="color: {shift_color}">₹{mp:,.0f}</div>
                    <div class="status-desc">Prev: ₹{prev_mp:,.0f} ({shift_text})</div>
                </div>
            """), unsafe_allow_html=True)

        with c2:
            vel_sign = f"{pcr_vel:+.3f}/min"
            vel_color = "#00d084" if pcr_vel > 0 else "#ff4d6d"
            st.markdown(_clean_html(f"""
                <div class="status-card">
                    <div class="status-title">⚡ PCR Momentum (dPCR/dt)</div>
                    <div class="status-value" style="color: {vel_color}">{pcr:.2f}</div>
                    <div class="status-desc">Velocity: {vel_sign}</div>
                </div>
            """), unsafe_allow_html=True)

        with c3:
            st.markdown(_clean_html(f"""
                <div class="status-card">
                    <div class="status-title">🛑 Resistance Call Build</div>
                    <div class="status-value" style="color: #ff4d6d">₹{int(top_ce_build):,}</div>
                    <div class="status-desc">ΔOI: +{top_ce_oi:,} contracts</div>
                </div>
            """), unsafe_allow_html=True)

        with c4:
            st.markdown(_clean_html(f"""
                <div class="status-card">
                    <div class="status-title">🛡️ Support Put Build</div>
                    <div class="status-value" style="color: #00d084">₹{int(top_pe_build):,}</div>
                    <div class="status-desc">ΔOI: +{top_pe_oi:,} contracts</div>
                </div>
            """), unsafe_allow_html=True)

        # 3. Active Institutional Traps & Squeezes
        traps = res_data.get("traps", [])
        squeezes = res_data.get("squeezes", [])
        signals = res_data.get("signals", [])

        st.markdown("#### 🚨 Active Institutional Microstructure Alerts")
        if traps or squeezes:
            for t in traps:
                st.markdown(_clean_html(f"""
                    <div style="background: rgba(255, 77, 109, 0.15); border-left: 4px solid #ff4d6d; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                        {t}
                    </div>
                """), unsafe_allow_html=True)
            for sq in squeezes:
                st.markdown(_clean_html(f"""
                    <div style="background: rgba(0, 208, 132, 0.15); border-left: 4px solid #00d084; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                        {sq}
                    </div>
                """), unsafe_allow_html=True)
        else:
            st.info("⚖️ **Equilibrium State:** No institutional writer traps or squeeze breakouts currently triggered. Smart money positioning is orderly.")

        # 4. Actionable Trade Setups
        if signals:
            st.markdown("#### 🎯 Actionable AI Derivatives Setups")
            for sig in signals:
                badge_col = "#00d084" if sig.get("bias") == "BULLISH" else "#ff4d6d"
                st.markdown(_clean_html(f"""
                    <div style="background: #1e2130; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: 700; font-size: 1.1rem; color: {badge_col};">{sig.get('setup')}</span>
                            <span style="background: {badge_col}; color: #000; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{sig.get('action')} • {sig.get('contract')}</span>
                        </div>
                        <div style="color: #c9d1d9; font-size: 0.9rem; margin-bottom: 8px;"><b>Thesis:</b> {sig.get('rationale')}</div>
                        <div style="display: flex; gap: 20px; font-size: 0.85rem; color: #8b949e;">
                            <span><b>Entry:</b> {sig.get('entry')}</span>
                            <span><b>Stop Loss:</b> {sig.get('stop_loss')}</span>
                            <span><b>Target 1:</b> {sig.get('target_1')}</span>
                            <span><b>Target 2:</b> {sig.get('target_2')}</span>
                            <span><b>Conviction:</b> {sig.get('confidence')}%</span>
                        </div>
                    </div>
                """), unsafe_allow_html=True)

    # 5. Strike-by-Strike Delta OI Chart
    st.markdown("#### 📊 Strike-by-Strike ΔOI Velocity Distribution")
    if snapshots and len(snapshots) >= 2:
        prev_oc = snapshots[-2][1]
        curr_oc = snapshots[-1][1]
        if prev_oc is not None and curr_oc is not None and not prev_oc.empty and not curr_oc.empty:
            p_m = prev_oc.set_index(["strike", "option_type"])["oi"]
            c_m = curr_oc.set_index(["strike", "option_type"])["oi"]
            delta_s = (c_m - p_m).fillna(0).reset_index()
            
            # Filter around ATM
            step = analyzer.strike_step or 50.0
            atm = round(spot_price / step) * step
            min_s = atm - (10 * step)
            max_s = atm + (10 * step)
            delta_s = delta_s[(delta_s["strike"] >= min_s) & (delta_s["strike"] <= max_s)]

            strikes_s = sorted(delta_s["strike"].unique())
            ce_d = []
            pe_d = []
            for s in strikes_s:
                ce_val = delta_s[(delta_s["strike"] == s) & (delta_s["option_type"] == "CE")]["oi"]
                pe_val = delta_s[(delta_s["strike"] == s) & (delta_s["option_type"] == "PE")]["oi"]
                ce_d.append(int(ce_val.iloc[0]) if not ce_val.empty else 0)
                pe_d.append(int(pe_val.iloc[0]) if not pe_val.empty else 0)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=strikes_s,
                y=ce_d,
                name="Call ΔOI (Resistance Addition / Covering)",
                marker_color="#ff4d6d",
            ))
            fig.add_trace(go.Bar(
                x=strikes_s,
                y=pe_d,
                name="Put ΔOI (Support Addition / Covering)",
                marker_color="#00d084",
            ))
            fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="#f1fa8c", annotation_text=f"Spot ₹{spot_price:,.1f}")
            fig.update_layout(
                barmode="group",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="#c9d1d9",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", y=1.1, x=0.2),
                xaxis=dict(title="Strike Price", gridcolor="#21262d"),
                yaxis=dict(title="Contracts Added / Unwound", gridcolor="#21262d"),
            )
            st.plotly_chart(fig, use_container_width=True)
    elif latest_oc is not None and not latest_oc.empty:
        step = analyzer.strike_step or 50.0
        atm = round(spot_price / step) * step
        min_s = atm - (10 * step)
        max_s = atm + (10 * step)
        cur_s = latest_oc[(latest_oc["strike"] >= min_s) & (latest_oc["strike"] <= max_s)]
        strikes_s = sorted(cur_s["strike"].unique())
        ce_d = [int(cur_s[(cur_s["strike"] == s) & (cur_s["option_type"] == "CE")]["oi"].sum()) for s in strikes_s]
        pe_d = [int(cur_s[(cur_s["strike"] == s) & (cur_s["option_type"] == "PE")]["oi"].sum()) for s in strikes_s]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=strikes_s,
            y=ce_d,
            name="Call OI (Resistance Ceiling)",
            marker_color="#ff4d6d",
        ))
        fig.add_trace(go.Bar(
            x=strikes_s,
            y=pe_d,
            name="Put OI (Support Floor)",
            marker_color="#00d084",
        ))
        fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="#f1fa8c", annotation_text=f"Spot ₹{spot_price:,.1f}")
        fig.update_layout(
            title="<b>Open Interest Strike Distribution (Current Snapshot)</b>",
            barmode="group",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#c9d1d9",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", y=1.1, x=0.2),
            xaxis=dict(title="Strike Price", gridcolor="#21262d"),
            yaxis=dict(title="Total Open Interest", gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ Displaying baseline Open Interest distribution. Live strike-by-strike ΔOI velocity and institutional writer traps update continuously as new snapshots arrive.")
    else:
        st.info("Requires at least 2 consecutive snapshots to compute real-time strike ΔOI velocity.")


if __name__ == "__main__":
    run_dashboard()
else:
    # Ensure Streamlit navigation exec() context runs the dashboard
    run_dashboard()
