"""
BTST Institutional Scanner Dashboard.

Dedicated terminal to decode the 15:00 – 15:30 IST market microstructure window:
- Last 30-Minute Volume Shockwave (% of daily volume)
- Day High Proximity (Stocks closing at extreme highs)
- Open Interest Quadrant Analysis (Long Build-Up vs Short Covering Trap)
- Algorithmic BTST Conviction Scoring (0 - 100)
- Actionable Setups (Entry, Stop Loss, 09:20 AM Targets, Risk:Reward)
- Telegram Notification Dispatch
"""

from __future__ import annotations

import logging
from datetime import datetime, date, time as dtime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from trade_system.domains.analysis.application.analysis.btst_scanner import (
    BTSTInstitutionalScanner,
    BTSTCandidate,
    BTSTScanResult,
)
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.interfaces.dashboard.shared_broker import get_cached_broker

LOGGER = logging.getLogger(__name__)


def _clean_html(html_str: str) -> str:
    """Strip leading/trailing whitespace on each line to prevent markdown parser code-block induction."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines())


def inject_custom_css():
    st.markdown(_clean_html("""
    <style>
        .block-container { padding-top: 1.8rem !important; padding-bottom: 2rem !important; }
        
        /* Hero Banner */
        .btst-hero {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            padding: 18px 24px;
            border-radius: 12px;
            border: 1px solid #30363d;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        
        /* BTST Card */
        .btst-card {
            background: #1e2130;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #30363d;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }

        /* Trap Card */
        .trap-card {
            background: #2b1117;
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid #ff4d6d55;
            margin-bottom: 12px;
        }

        .metric-badge {
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
        }
    </style>
    """), unsafe_allow_html=True)


def get_market_window_status() -> Tuple[str, str, str]:
    """Evaluate current IST time relative to the 15:00-15:30 BTST execution window."""
    now = datetime.now()
    t = now.time()
    if dtime(15, 0) <= t <= dtime(15, 30):
        return (
            "🔥 ACTIVE INSTITUTIONAL WINDOW (15:00 – 15:30 IST)",
            "#00d084",
            "Institutions are actively executing closing VWAP orders and accumulating overnight delivery. Prime time to enter BTST!"
        )
    elif dtime(9, 15) <= t < dtime(15, 0):
        return (
            "⏱️ INTRADAY PRE-CLOSE PREVIEW (< 15:00 IST)",
            "#ffb703",
            "Scanning early day-gainers near highs. Official 15:00-15:30 institutional volume shockwave will activate in the final 30 minutes."
        )
    else:
        return (
            "🌙 POST-MARKET EOD REVIEW (> 15:30 IST)",
            "#58a6ff",
            "Session closed. Analyzing finalized 15:00-15:30 volume shockwave and delivery accumulation for tomorrow's market open."
        )


def render_btst_dashboard():
    inject_custom_css()

    st.title("🌅 BTST Institutional Scanner")
    st.caption("Decodes the 15:00 – 15:30 IST Market Microstructure Window: Volume Shockwave, Day High Absorption & OI Long Buildup.")

    # 1. Market Window Status Banner
    status_title, status_color, status_desc = get_market_window_status()
    now_str = datetime.now().strftime("%H:%M:%S IST • %d %b %Y")

    st.markdown(_clean_html(f"""
    <div class="btst-hero">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    Microstructure Window • {now_str}
                </div>
                <div style="font-size:1.3rem; font-weight:bold; color:{status_color}; margin-top:3px;">
                    {status_title}
                </div>
                <div style="font-size:0.85rem; color:#c9d1d9; margin-top:2px;">
                    {status_desc}
                </div>
            </div>
            <div style="display:flex; gap:12px; align-items:center;">
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;">F&O Universe Size</div>
                    <div style="font-size:1.2rem; font-weight:bold; color:#f0f6fc; font-family:monospace;">~180 Stocks</div>
                </div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # 2. Control Bar
    c_btn1, c_btn2, c_mode, c_min_score = st.columns([1.2, 1.4, 1.2, 1.2])

    with c_btn1:
        run_scan = st.button("🚀 Run Live BTST Scan", type="primary", use_container_width=True)
    with c_btn2:
        send_tg = st.button("📢 Send Top Picks to Telegram", use_container_width=True)
    with c_mode:
        scan_universe_mode = st.selectbox("Scan Scope", ["Top 35 Momentum Stocks", "Full F&O Universe (~180)"])
    with c_min_score:
        min_score_filter = st.slider("Min BTST Score", min_value=40, max_value=90, value=60, step=5)

    # Execution Logic
    scanner = BTSTInstitutionalScanner(broker=get_cached_broker())

    if run_scan or "btst_scan_result" not in st.session_state:
        max_symbols = 35 if "Top 35" in scan_universe_mode else None
        progress_bar = st.progress(0, text="Initializing two-tier F&O scan...")

        def _update_prog(ratio, txt):
            progress_bar.progress(ratio, text=txt)

        with st.spinner("Scanning 15:00-15:30 volume shockwaves & option chain OI..."):
            universe_list = get_fo_universe()
            if max_symbols:
                universe_list = universe_list[:max_symbols]
            res = scanner.scan_btst(symbols=universe_list, progress_callback=_update_prog)
            st.session_state["btst_scan_result"] = res
            progress_bar.empty()
            st.success(f"✅ Scan completed at {res.scan_time}! Identified {len(res.top_picks)} BTST picks and {len(res.traps)} short-covering traps.")

    scan_res: BTSTScanResult = st.session_state.get("btst_scan_result")
    if not scan_res:
        st.warning("No scan data available. Click 'Run Live BTST Scan' above.")
        return

    # Telegram Dispatch
    if send_tg:
        top_to_send = [c for c in scan_res.top_picks if c.btst_score >= min_score_filter][:3]
        if top_to_send:
            with st.spinner("Dispatching Telegram alerts..."):
                count = scanner.dispatch_telegram_alerts(top_to_send)
                st.success(f"🚀 Dispatched {count} high-conviction BTST alerts to Telegram!")
        else:
            st.info(f"No BTST candidates met the min score threshold ({min_score_filter}) for Telegram.")

    # 3. Top 3 BTST Conviction Hero Cards
    st.markdown("### 🏆 Top Institutional BTST Picks (Highest Conviction)")
    st.caption("Stocks closing near Day High with massive 15:00-15:30 volume concentration and genuine Long Buildup.")

    filtered_top = [c for c in scan_res.top_picks if c.btst_score >= min_score_filter]

    if filtered_top:
        card_cols = st.columns(min(3, len(filtered_top)))
        for i, cand in enumerate(filtered_top[:3]):
            with card_cols[i]:
                st.markdown(_clean_html(f"""
                <div class="btst-card" style="border-top: 5px solid {cand.verdict_badge_color};">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; font-weight:bold;">{cand.sector}</span>
                            <span class="metric-badge" style="background:#00d08422; color:#00d084; border:1px solid #00d084;">
                                SCORE: {cand.btst_score}/100 🔥
                            </span>
                        </div>
                        <div style="font-size:1.4rem; font-weight:800; color:#f0f6fc; margin-bottom:2px;">
                            {cand.clean_symbol}
                        </div>
                        <div style="font-size:1.1rem; font-weight:bold; color:#00d084; font-family:monospace; margin-bottom:10px;">
                            ₹{cand.spot_price:,.2f} <span style="font-size:0.85rem;">({cand.day_change_pct:+.2f}%)</span>
                        </div>
                        
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; background:#161b22; padding:8px 10px; border-radius:6px; margin-bottom:10px; font-size:0.75rem;">
                            <div><span style="color:#8b949e;">30m Vol Ratio:</span><br><b style="color:#00e6ff;">{cand.vol_30m_ratio_pct:.1f}% ({cand.vol_burst_multiplier:.1f}x)</b></div>
                            <div><span style="color:#8b949e;">Dist to High:</span><br><b style="color:#00d084;">{cand.dist_to_day_high_pct:.2f}%</b></div>
                            <div><span style="color:#8b949e;">OI Flow:</span><br><b style="color:#ffb703;">{cand.oi_quadrant.replace('_', ' ')}</b></div>
                            <div><span style="color:#8b949e;">Option PCR:</span><br><b style="color:#f0f6fc;">{cand.pcr_oi:.2f}</b></div>
                        </div>
                        
                        <div style="font-size:0.75rem; color:#c9d1d9; margin-bottom:10px; line-height:1.4;">
                            <b>Action:</b> <span style="color:#00e6ff;">{cand.recommended_entry}</span><br>
                            <b>Target 1:</b> <span style="color:#00d084;">{cand.target_1}</span> | <b>SL:</b> <span style="color:#ff4d6d;">{cand.stop_loss}</span>
                        </div>
                    </div>
                    <div style="font-size:0.7rem; color:#8b949e; border-top:1px solid #30363d; padding-top:6px;">
                        💡 <i>{cand.rationale}</i>
                    </div>
                </div>
                """), unsafe_allow_html=True)
    elif scan_res.stbt_picks:
        st.info(f"Market showed downward momentum: No BTST (Long) met threshold, but {len(scan_res.stbt_picks)} institutional STBT (Short Breakdown) setups were detected:")
        stbt_cols = st.columns(min(3, len(scan_res.stbt_picks)))
        for i, cand in enumerate(scan_res.stbt_picks[:3]):
            with stbt_cols[i]:
                st.markdown(_clean_html(f"""
                <div class="btst-card" style="border-top: 5px solid {cand.verdict_badge_color};">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; font-weight:bold;">{cand.sector}</span>
                            <span class="metric-badge" style="background:#f77f0022; color:#f77f00; border:1px solid #f77f00;">
                                STBT SCORE: {cand.btst_score}/100 ⚡
                            </span>
                        </div>
                        <div style="font-size:1.4rem; font-weight:800; color:#f0f6fc; margin-bottom:2px;">
                            {cand.clean_symbol}
                        </div>
                        <div style="font-size:1.1rem; font-weight:bold; color:#ff4d6d; font-family:monospace; margin-bottom:10px;">
                            ₹{cand.spot_price:,.2f} <span style="font-size:0.85rem;">({cand.day_change_pct:+.2f}%)</span>
                        </div>
                        
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; background:#161b22; padding:8px 10px; border-radius:6px; margin-bottom:10px; font-size:0.75rem;">
                            <div><span style="color:#8b949e;">30m Vol Ratio:</span><br><b style="color:#00e6ff;">{cand.vol_30m_ratio_pct:.1f}% ({cand.vol_burst_multiplier:.1f}x)</b></div>
                            <div><span style="color:#8b949e;">Proximity:</span><br><b style="color:#ff4d6d;">At Day Low</b></div>
                            <div><span style="color:#8b949e;">OI Flow:</span><br><b style="color:#ffb703;">{cand.oi_quadrant.replace('_', ' ')}</b></div>
                            <div><span style="color:#8b949e;">Option PCR:</span><br><b style="color:#f0f6fc;">{cand.pcr_oi:.2f}</b></div>
                        </div>
                        
                        <div style="font-size:0.75rem; color:#c9d1d9; margin-bottom:10px; line-height:1.4;">
                            <b>Action:</b> <span style="color:#f77f00;">{cand.recommended_entry}</span><br>
                            <b>Target 1:</b> <span style="color:#00d084;">{cand.target_1}</span> | <b>SL:</b> <span style="color:#ff4d6d;">{cand.stop_loss}</span>
                        </div>
                    </div>
                    <div style="font-size:0.7rem; color:#8b949e; border-top:1px solid #30363d; padding-top:6px;">
                        💡 <i>{cand.rationale}</i>
                    </div>
                </div>
                """), unsafe_allow_html=True)
    else:
        st.info(f"No BTST candidates met the minimum score threshold of {min_score_filter}. Lower the threshold or re-run scan during 15:15-15:30.")

    # 4. The Short-Covering Trap Radar
    if scan_res.traps:
        st.markdown("---")
        st.markdown("### ⚠️ The Short-Covering Trap Radar (DO NOT CARRY OVERNIGHT)")
        st.caption("These stocks rallied sharply into the close, but Call/Futures OI fell drastically. Late buying was driven by intraday bears covering, NOT fresh institutional accumulation.")

        for trap in scan_res.traps[:3]:
            st.markdown(_clean_html(f"""
            <div class="trap-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <b style="color:#ff4d6d; font-size:1.05rem;">{trap.clean_symbol}</b> 
                        <span style="color:#f0f6fc; font-family:monospace; margin-left:8px;">₹{trap.spot_price:,.2f} ({trap.day_change_pct:+.2f}%)</span>
                        <span class="metric-badge" style="background:#ff4d6d22; color:#ff4d6d; border:1px solid #ff4d6d; margin-left:8px;">
                            TRAP SCORE: {trap.btst_score}/100
                        </span>
                    </div>
                    <div style="font-size:0.8rem; color:#ffb703;">
                        30m Vol: {trap.vol_30m_ratio_pct:.1f}% | OI: {trap.oi_quadrant} (CE Unwinding)
                    </div>
                </div>
                <div style="font-size:0.8rem; color:#c9d1d9; margin-top:4px;">
                    {trap.trap_warning}
                </div>
            </div>
            """), unsafe_allow_html=True)

    # 5. Full Candidates Data Table
    st.markdown("---")
    st.markdown("### 📊 Universe BTST & STBT Candidates Table")

    table_data = []
    for c in scan_res.all_results:
        table_data.append({
            "Symbol": c.clean_symbol,
            "Sector": c.sector,
            "Spot LTP": c.spot_price,
            "Day Chg %": c.day_change_pct,
            "Dist High %": c.dist_to_day_high_pct,
            "30m Vol %": c.vol_30m_ratio_pct,
            "Burst": f"{c.vol_burst_multiplier:.1f}x",
            "OI Quadrant": c.oi_quadrant,
            "PCR": c.pcr_oi,
            "Score": c.btst_score,
            "Verdict": c.verdict,
            "Target 1": c.target_1,
            "Stop Loss": c.stop_loss,
        })

    if table_data:
        df_table = pd.DataFrame(table_data)
        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Spot LTP": st.column_config.NumberColumn(format="₹%.2f"),
                "Day Chg %": st.column_config.NumberColumn(format="%+.2f%%"),
                "Dist High %": st.column_config.NumberColumn(format="%.2f%%"),
                "30m Vol %": st.column_config.NumberColumn(format="%.1f%%"),
                "PCR": st.column_config.NumberColumn(format="%.2f"),
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            }
        )

    # 6. Deep-Dive Stock Inspector (5-Minute Candlesticks & 15:00-15:30 Shockwave)
    st.markdown("---")
    st.markdown("### 🔬 Deep-Dive Stock Inspector (5-Minute Volume Shockwave)")
    st.caption("Visually inspect the 15:00 – 15:30 IST volume surge and price action against session VWAP.")

    candidate_symbols = [c.clean_symbol for c in scan_res.all_results]
    if candidate_symbols:
        selected_clean_sym = st.selectbox("Select Candidate to Inspect", candidate_symbols, index=0)
        selected_cand = next((c for c in scan_res.all_results if c.clean_symbol == selected_clean_sym), None)

        if selected_cand:
            c_chart, c_forensics = st.columns([1.5, 1])

            with c_chart:
                # Fetch 5-minute candles
                candles_df = scanner.fetch_candidate_candles(selected_cand.symbol, resolution="5")
                if not candles_df.empty:
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{selected_cand.clean_symbol} — 5-Min Intraday Session", "Volume & 15:00-15:30 Shockwave")
                    )

                    # Candlesticks
                    fig.add_trace(
                        go.Candlestick(
                            x=candles_df["timestamp"],
                            open=candles_df["open"],
                            high=candles_df["high"],
                            low=candles_df["low"],
                            close=candles_df["close"],
                            name="Price",
                            increasing_line_color="#00d084",
                            decreasing_line_color="#ff4d6d",
                        ),
                        row=1, col=1
                    )

                    # Session VWAP Line
                    tp = (candles_df["high"] + candles_df["low"] + candles_df["close"]) / 3
                    cum_vol = candles_df["volume"].cumsum()
                    cum_tp_vol = (tp * candles_df["volume"]).cumsum()
                    vwap_series = cum_tp_vol / cum_vol.replace(0, 1)

                    fig.add_trace(
                        go.Scatter(
                            x=candles_df["timestamp"],
                            y=vwap_series,
                            name="Session VWAP",
                            line=dict(color="#ffb703", width=1.5, dash="dot"),
                        ),
                        row=1, col=1
                    )

                    # Highlight 15:00-15:30 Window
                    # Color volume bars: cyan for 15:00-15:30, gray otherwise
                    vol_colors = [
                        "#00e6ff" if ts.time() >= dtime(15, 0) else "#30363d"
                        for ts in candles_df["timestamp"]
                    ]

                    fig.add_trace(
                        go.Bar(
                            x=candles_df["timestamp"],
                            y=candles_df["volume"],
                            name="Volume",
                            marker_color=vol_colors,
                        ),
                        row=2, col=1
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=480,
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="#161b22",
                        plot_bgcolor="#161b22",
                        xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
                    )

                    st.plotly_chart(fig, use_container_width=True, key=f"btst_chart_{selected_clean_sym}")
                else:
                    st.info(f"5-minute intraday candle history currently unavailable for {selected_clean_sym}.")

            with c_forensics:
                st.markdown(_clean_html(f"""
                <div style="background:#1e2130; padding:18px; border-radius:12px; border:1px solid #30363d;">
                    <div style="font-size:1.1rem; font-weight:bold; color:#f0f6fc; margin-bottom:12px;">
                        🎯 4-Pillar BTST Scoring Matrix
                    </div>
                    
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e;">
                            <span>1. Day High Proximity</span>
                            <b style="color:#f0f6fc;">{selected_cand.price_action_score} / 25</b>
                        </div>
                        <div style="background:#21262d; height:6px; border-radius:3px; overflow:hidden; margin-top:3px;">
                            <div style="background:#00d084; width:{selected_cand.price_action_score / 25 * 100}%; height:100%;"></div>
                        </div>
                    </div>

                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e;">
                            <span>2. 30m Volume Shockwave</span>
                            <b style="color:#f0f6fc;">{selected_cand.volume_shockwave_score} / 25</b>
                        </div>
                        <div style="background:#21262d; height:6px; border-radius:3px; overflow:hidden; margin-top:3px;">
                            <div style="background:#00e6ff; width:{selected_cand.volume_shockwave_score / 25 * 100}%; height:100%;"></div>
                        </div>
                    </div>

                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e;">
                            <span>3. Institutional OI Alignment</span>
                            <b style="color:#f0f6fc;">{selected_cand.oi_flow_score} / 25</b>
                        </div>
                        <div style="background:#21262d; height:6px; border-radius:3px; overflow:hidden; margin-top:3px;">
                            <div style="background:#ffb703; width:{selected_cand.oi_flow_score / 25 * 100}%; height:100%;"></div>
                        </div>
                    </div>

                    <div style="margin-bottom:14px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e;">
                            <span>4. Momentum & Relative Strength</span>
                            <b style="color:#f0f6fc;">{selected_cand.trend_score} / 25</b>
                        </div>
                        <div style="background:#21262d; height:6px; border-radius:3px; overflow:hidden; margin-top:3px;">
                            <div style="background:#a855f7; width:{selected_cand.trend_score / 25 * 100}%; height:100%;"></div>
                        </div>
                    </div>

                    <div style="background:#161b22; padding:12px; border-radius:8px; border-left:4px solid {selected_cand.verdict_badge_color};">
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; font-weight:bold;">Conviction Verdict</div>
                        <div style="font-size:1.1rem; font-weight:bold; color:{selected_cand.verdict_badge_color}; margin:2px 0;">
                            {selected_cand.verdict.replace('_', ' ')}
                        </div>
                        <div style="font-size:0.8rem; color:#c9d1d9; margin-top:4px;">
                            {selected_cand.rationale}
                        </div>
                    </div>
                </div>
                """), unsafe_allow_html=True)


if __name__ == "__main__":
    render_btst_dashboard()
