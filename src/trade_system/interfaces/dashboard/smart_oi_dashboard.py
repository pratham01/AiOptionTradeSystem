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
from trade_system.domains.analysis.application.analysis.fo_pcr_screener import (
    FOPCRScreener,
    StockPCRInfo,
)
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.shared.config import Settings
from trade_system.interfaces.dashboard.shared_broker import get_cached_broker

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
def fetch_live_price_history(symbol: str, resolution: str = "5") -> pd.DataFrame:
    """Fetch recent intraday price candles from Fyers for live VWAP and charts."""
    try:
        broker = get_broker()
        if not broker:
            return pd.DataFrame()
        today_str = date.today().strftime("%Y-%m-%d")
        df = broker.fetch_history(symbol, resolution, today_str, today_str)
        if df is not None and not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        LOGGER.error(f"Error fetching live price history: {e}")
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
            start_time = f"{target_date} 00:00:00"
            end_time = f"{target_date} 23:59:59"
            query = text("""
                SELECT DISTINCT timestamp 
                FROM option_chain_data 
                WHERE underlying_symbol = :symbol 
                  AND timestamp >= :start_time
                  AND timestamp <= :end_time
                ORDER BY timestamp ASC
            """)
            ts_df = pd.read_sql(query, conn, params={"symbol": symbol, "start_time": start_time, "end_time": end_time})
            if ts_df.empty:
                return []
            
            for ts in ts_df["timestamp"]:
                data_query = text("""
                    SELECT * 
                    FROM option_chain_data 
                    WHERE underlying_symbol = :symbol 
                      AND timestamp = :ts
                """)
                df = pd.read_sql(data_query, conn, params={"symbol": symbol, "ts": ts})
                ts_dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                snapshots.append((ts_dt, df))
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
            start_time = f"{target_date} 00:00:00"
            end_time = f"{target_date} 23:59:59"
            query = text("""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_1m 
                WHERE symbol = :symbol 
                  AND timestamp >= :start_time
                  AND timestamp <= :end_time
                ORDER BY timestamp ASC
            """)
            df = pd.read_sql(query, conn, params={"symbol": symbol, "start_time": start_time, "end_time": end_time})
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
    except Exception as e:
        LOGGER.error(f"Error loading price data: {e}")
        return pd.DataFrame()


def inject_custom_css():
    """Inject custom styles and CSS for status cards and transition timeline."""
    st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
        div[data-testid="stVerticalBlock"] > div { margin-top: -0.5rem !important; }
        
        /* Smart Cards */
        .status-card {
            background: #1e2130;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #30363d;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }
        .status-title {
            font-size: 0.9rem;
            color: #8b949e;
            margin-bottom: 5px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .status-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
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
    """, unsafe_allow_html=True)


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


def render_institutional_direction_pulse(dir_analysis: dict):
    """Render the top-level Institutional Market Direction & Gamma Pulse Meter."""
    intra_dir = dir_analysis.get("intraday_direction", "NEUTRAL")
    intra_score = dir_analysis.get("intraday_score", 0)
    swing_dir = dir_analysis.get("swing_direction", "NEUTRAL")
    swing_score = dir_analysis.get("swing_score", 0)
    gex_regime = dir_analysis.get("gamma_regime", "NEUTRAL")
    gex_flip = dir_analysis.get("gamma_flip_level", 0.0)
    inst_bias = dir_analysis.get("institutional_bias", "NEUTRAL")
    wall_status = dir_analysis.get("wall_shift_status", "")

    intra_color = "#00d084" if "BULLISH" in intra_dir else ("#ff4d6d" if "BEARISH" in intra_dir else "#ffb703")
    swing_color = "#00d084" if "BULLISH" in swing_dir else ("#ff4d6d" if "BEARISH" in swing_dir else "#00b4d8")
    gex_color = "#ff4d6d" if "SHORT GAMMA" in gex_regime else ("#00d084" if "LONG GAMMA" in gex_regime else "#ffb703")

    amd_phase = dir_analysis.get("amd_phase", "CONSOLIDATION")
    amd_low = dir_analysis.get("amd_range_low", 0.0)
    amd_high = dir_analysis.get("amd_range_high", 0.0)
    amd_action = dir_analysis.get("amd_action", "STAND_ASIDE_ACCUMULATION")
    amd_color = "#00f5d4" if "SPRING" in amd_phase else ("#ff4d6d" if "UTAD" in amd_phase else ("#70d6ff" if "DISTRIBUTION" in amd_phase else "#ffd166"))

    st.markdown(f"""
    <div style="background:#161b22; padding:18px 22px; border-radius:12px; border:1px solid #30363d; margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="font-size:1.1rem; font-weight:bold; color:#f0f6fc;">🏛️ Institutional Market Direction & Smart Money Pulse</div>
            <div style="font-size:0.85rem; color:#8b949e; background:#21262d; padding:4px 10px; border-radius:6px;">{inst_bias}</div>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:15px;">
            <div style="background:#1e2130; padding:14px; border-radius:8px; border-left:5px solid {intra_color};">
                <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;">Intraday Directional Score</div>
                <div style="font-size:1.4rem; font-weight:bold; color:{intra_color};">{intra_dir}</div>
                <div style="font-size:0.8rem; color:#c9d1d9; margin-top:3px;">Score: <b>{intra_score:+d}</b> / 100</div>
            </div>
            <div style="background:#1e2130; padding:14px; border-radius:8px; border-left:5px solid {swing_color};">
                <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;">Swing Multi-Session Bias</div>
                <div style="font-size:1.4rem; font-weight:bold; color:{swing_color};">{swing_dir}</div>
                <div style="font-size:0.8rem; color:#c9d1d9; margin-top:3px;">Score: <b>{swing_score:+d}</b> / 100</div>
            </div>
            <div style="background:#1e2130; padding:14px; border-radius:8px; border-left:5px solid {gex_color};">
                <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;">Dealer Gamma Regime</div>
                <div style="font-size:1.05rem; font-weight:bold; color:{gex_color};">{gex_regime}</div>
                <div style="font-size:0.8rem; color:#c9d1d9; margin-top:3px;">Gamma Flip Level: <b>₹{gex_flip:,.0f}</b></div>
            </div>
        </div>
        <div style="margin-top:12px; padding:10px 14px; background:#1e2130; border-radius:8px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">📦 Wyckoff / PO3 Cycle:</span>
                <b style="color:{amd_color}; font-size:0.95rem; margin-left:8px;">{amd_phase}</b>
                <span style="font-size:0.8rem; color:#c9d1d9; margin-left:12px;">Accumulation Range: ₹{amd_low:,.1f} — ₹{amd_high:,.1f}</span>
            </div>
            <div style="font-size:0.8rem; color:#f0f6fc; background:#30363d; padding:3px 8px; border-radius:4px;">
                Action: <b>{amd_action}</b>
            </div>
        </div>
        <div style="margin-top:10px; font-size:0.8rem; color:#8b949e;">
            📊 <b>Wall Shifts:</b> {wall_status}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_trade_setups_panel(setups_dict: dict):
    """Render Actionable Intraday & Swing Trade Setup Cards."""
    intra = setups_dict.get("intraday_setup", {})
    swing = setups_dict.get("swing_setup", {})

    st.subheader("🎯 Actionable Trade Setups: Intraday & Swing")
    st.caption("AI-generated trade blueprints derived from Institutional Smart OI flow, Gamma regimes, and Technical confluence.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background:#1e2130; padding:18px; border-radius:12px; border-top:5px solid {intra.get('status_color', '#8b949e')}; border-left:1px solid #30363d; border-right:1px solid #30363d; border-bottom:1px solid #30363d; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.8rem; color:#8b949e; font-weight:600; text-transform:uppercase;">⚡ Intraday Trade Blueprint</span>
                <span style="font-size:0.8rem; background:#21262d; color:{intra.get('status_color')}; font-weight:bold; padding:2px 8px; border-radius:4px;">Conviction {intra.get('conviction_score')}</span>
            </div>
            <div style="font-size:1.3rem; font-weight:bold; color:{intra.get('status_color')}; margin-bottom:6px;">{intra.get('type')} ({intra.get('direction')})</div>
            <div style="background:#161b22; padding:10px; border-radius:6px; font-size:0.85rem; color:#f0f6fc; margin-bottom:10px;">
                <b>Recommended Contract:</b> <span style="color:#ffb703; font-weight:bold;">{intra.get('option_contract')}</span><br>
                <b>Trigger:</b> {intra.get('actionable_trigger')}
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px; font-size:0.8rem; text-align:center; margin-bottom:10px;">
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Entry Zone</span><br><b style="color:#f0f6fc;">{intra.get('entry_zone')}</b></div>
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Stop Loss</span><br><b style="color:#ff4d6d;">{intra.get('stop_loss')}</b></div>
                <div style="background:#21262d; padding:6px; border-radius:4px;"><span style="color:#8b949e;">Target 1</span><br><b style="color:#00d084;">{intra.get('target_1')}</b></div>
            </div>
            <div style="font-size:0.8rem; color:#c9d1d9; line-height:1.4;">
                💡 <b>Rationale:</b> {intra.get('rationale')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:#1e2130; padding:18px; border-radius:12px; border-top:5px solid {swing.get('status_color', '#8b949e')}; border-left:1px solid #30363d; border-right:1px solid #30363d; border-bottom:1px solid #30363d; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.8rem; color:#8b949e; font-weight:600; text-transform:uppercase;">🌊 Swing Trade Blueprint ({swing.get('timeframe')})</span>
                <span style="font-size:0.8rem; background:#21262d; color:{swing.get('status_color')}; font-weight:bold; padding:2px 8px; border-radius:4px;">Conviction {swing.get('conviction_score')}</span>
            </div>
            <div style="font-size:1.3rem; font-weight:bold; color:{swing.get('status_color')}; margin-bottom:6px;">{swing.get('type')}</div>
            <div style="background:#161b22; padding:10px; border-radius:6px; font-size:0.85rem; color:#f0f6fc; margin-bottom:10px;">
                <b>Strategy:</b> <span style="color:#00e6ff; font-weight:bold;">{swing.get('recommended_strategy')}</span>
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
        """, unsafe_allow_html=True)


def render_verdict_card(buyer_verdict: str, buyer_desc: str, buyer_color: str):
    """Render the Actionable Option Buyer Verdict card."""
    st.markdown(f"""
    <div style="background:#1e2130; padding:20px; border-radius:12px; border-left:8px solid {buyer_color}; border-top:1px solid #30363d; border-right:1px solid #30363d; border-bottom:1px solid #30363d; margin-bottom:20px;">
        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; font-weight:600; margin-bottom:5px;">Actionable Option Buyer Verdict</div>
        <div style="font-size:1.6rem; font-weight:bold; color:{buyer_color}; margin-bottom:8px;">{buyer_verdict}</div>
        <div style="font-size:0.95rem; color:#c9d1d9; line-height:1.5;">{buyer_desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_grid(
    call_wall: float, call_wall_oi: float, dist_to_call_wall: float,
    put_wall: float, put_wall_oi: float, dist_to_put_wall: float,
    spot_price: float, max_pain_strike: float,
    pcr: float, atm_iv_val: float, iv_status: str, iv_color: str
):
    """Render the key walls and PCR status card grid."""
    def fmt(v, format_str, prefix="", suffix=""):
        try:
            if v is None or v != v: return "N/A"
            return f"{prefix}{v:{format_str}}{suffix}"
        except Exception:
            return "N/A"

    def fmt_int(v, prefix="", suffix=""):
        try:
            if v is None or v != v: return "N/A"
            return f"{prefix}{int(v):,}{suffix}"
        except Exception:
            return "N/A"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="status-card" style="border-top: 5px solid #ff4d6d;">
            <div class="status-title">🔴 Call Wall (Resistance)</div>
            <div class="status-value" style="color: #ff4d6d; font-size:1.6rem;">{fmt(call_wall, ',.0f', prefix='₹')}</div>
            <div class="status-desc">{fmt(dist_to_call_wall, '.2f', suffix='%')} away | OI: {fmt(call_wall_oi, ',.0f')}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="status-card" style="border-top: 5px solid #00d084;">
            <div class="status-title">🟢 Put Wall (Support)</div>
            <div class="status-value" style="color: #00d084; font-size:1.6rem;">{fmt(put_wall, ',.0f', prefix='₹')}</div>
            <div class="status-desc">{fmt(dist_to_put_wall, '.2f', suffix='%')} away | OI: {fmt(put_wall_oi, ',.0f')}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="status-card" style="border-top: 5px solid #ffb703;">
            <div class="status-title">Spot Price</div>
            <div class="status-value" style="color: #ffb703; font-size:1.6rem;">{fmt(spot_price, ',.2f', prefix='₹')}</div>
            <div class="status-desc">Max Pain Strike: {fmt_int(max_pain_strike, prefix='₹')}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="status-card" style="border-top: 5px solid {iv_color};">
            <div class="status-title">PCR & Premium Cost</div>
            <div class="status-value" style="color: {iv_color}; font-size:1.6rem;">{fmt(pcr, '.2f')}</div>
            <div class="status-desc">{iv_status} ({fmt(atm_iv_val, '.1f', suffix='%')} IV)</div>
        </div>
        """, unsafe_allow_html=True)


def render_confluence_alert(confluence_data: dict):
    """Render the confluence narrative alert box."""
    if confluence_data:
        st.write("")
        alert_type = confluence_data["status"]
        alert_color = "#00d084" if "BULLISH CONFLUENCE" in alert_type else ("#ff4d6d" if "BEARISH CONFLUENCE" in alert_type else ("#ffb703" if "DIVERGENCE" in alert_type else "#00b4d8"))
        
        st.markdown(f"""
        <div style="background:#1e2130; padding:15px; border-radius:8px; border-left:6px solid {alert_color}; margin-bottom:20px;">
            <h4 style="margin:0 0 5px 0; color:{alert_color};">{alert_type}</h4>
            <p style="margin:0; color:#c9d1d9; font-size:0.95rem;">{confluence_data['narrative']}</p>
        </div>
        """, unsafe_allow_html=True)


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
            price_df = fetch_live_price_history(db_symbol, resolution="5")
            if not price_df.empty:
                price_df = calculate_price_vwap(price_df)
            if latest_oc is not None and not latest_oc.empty:
                snapshots = [(datetime.now(), latest_oc)]
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

    # --- 0. INSTITUTIONAL DIRECTION & SMART MONEY PULSE ---
    render_institutional_direction_pulse(dir_analysis)

    # --- 0.5 ACTIONABLE TRADE SETUPS (INTRADAY & SWING) ---
    render_trade_setups_panel(trade_setups)

    st.markdown("---")

    # --- 1. CURRENT SMART OI VERDICT & OPTION BUYER'S PANEL ---
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

    # Render Verdict Panel Card
    render_verdict_card(buyer_verdict, buyer_desc, buyer_color)

    # Calculate IV Status
    atm_iv_val = latest_oc[latest_oc["strike"] == atm_strike]["iv"].mean() if "iv" in latest_oc.columns and not latest_oc.empty else None
    iv_status = "Neutral"
    iv_color = "#00b4d8"
    if atm_iv_val and not np.isnan(atm_iv_val):
        if atm_iv_val < 12.0:
            iv_status = "Cheap Volatility"
            iv_color = "#00d084"
        elif atm_iv_val > 17.0:
            iv_status = "Expensive Premium"
            iv_color = "#ff4d6d"

    # Render Metrics Grid
    render_metrics_grid(
        call_wall, call_wall_oi, dist_to_call_wall,
        put_wall, put_wall_oi, dist_to_put_wall,
        spot_price, max_pain_strike,
        pcr, atm_iv_val, iv_status, iv_color
    )

    # Confluence Alert Box
    render_confluence_alert(confluence_data)

    st.markdown("---")

    tab_divergence, tab_vol_profile, tab_causal_graph, tab_chart, tab_transitions, tab_pro_trader, tab_iv_greeks, tab_sniper, tab_fo_pcr = st.tabs([
        "⚡ Institutional Divergence & Traps",
        "📊 Strike Volume & OI Profile",
        "🕸️ Strike Causal Graph & Shockwave",
        "📈 Price Action & Smart OI Overlay",
        "⏱️ Signal Transitions Timeline",
        "🏦 Pro Trader Analytics & Walls",
        "📊 IV Skew & Greeks",
        "🎯 Sniper Reversals",
        "🎲 F&O Universe PCR Radar"
    ])

    with tab_divergence:
        render_tab_divergence_radar(divergence_data, spot_price, max_pain_strike, pcr, call_wall, put_wall)

    with tab_vol_profile:
        render_tab_volume_profile(volume_profile_df, spot_price, max_pain_strike)

    with tab_causal_graph:
        render_tab_causal_graph(
            symbol=db_symbol,
            spot_price=spot_price,
            latest_oc=latest_oc,
            call_wall=call_wall,
            put_wall=put_wall,
            max_pain=max_pain_strike,
            strike_step=analyzer.strike_step,
        )

    with tab_chart:
        render_tab_chart(price_df, snapshots, max_pain_strike, analyzer, selected_symbol_label)

    with tab_transitions:
        render_tab_transitions(snapshots, price_df, analyzer)

    with tab_pro_trader:
        render_tab_pro_trader(latest_oc, snapshots, spot_price, analyzer, prev_oc, summary)
        st.markdown("---")
        render_institutional_walls(latest_oc, snapshots, summary, analyzer, signal_strikes)

    with tab_iv_greeks:
        pro_atm = pro_oc_analyzer.compute_atm_premium_analysis(
            latest_oc, spot_price, analyzer.strike_step
        )
        render_tab_iv_greeks(latest_oc, spot_price, analyzer, pro_atm)

    with tab_sniper:
        render_tab_sniper(snapshots, spot_price, analyzer)

    with tab_fo_pcr:
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


if __name__ == "__main__":
    run_dashboard()
else:
    # Ensure Streamlit navigation exec() context runs the dashboard
    run_dashboard()
