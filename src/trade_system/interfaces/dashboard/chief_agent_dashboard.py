"""
Chief Agent Dashboard — Dedicated Mission Control & Sniper Trade Recommender.

Features:
- Enforces strict institutional daily discipline: Max 2-3 trades on Indices, Max 2 on Stocks.
- Visualizes the 5-Pillar Confluence Score (Price Action, Smart OI, Order Flow, Walls, Trap Filter).
- Generates high-conviction sniper trade cards with exact Strike, Entry, SL, and Targets.
- One-click live evaluation across Indices and F&O Stocks.
- Historical daily trade execution ledger.
"""
from __future__ import annotations

from datetime import datetime, date
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trade_system.domains.advisory.application.agent.chief_trading_agent import (
    ChiefTradingAgent,
    ChiefTradeSignal,
)
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe

LOGGER = logging.getLogger(__name__)


def inject_chief_agent_css() -> None:
    """Inject custom styles for Chief Trading Agent dashboard."""
    st.markdown("""
    <style>
        .chief-hero-card {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        .sniper-card {
            background: linear-gradient(135deg, #1c2333 0%, #161b22 100%);
            border-left: 6px solid #00d084;
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
            margin-bottom: 20px;
            box-shadow: 0 6px 12px rgba(0, 208, 132, 0.15);
        }
        .stand-aside-card {
            background: #161b22;
            border-left: 6px solid #8b949e;
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
            margin-bottom: 20px;
        }
        .budget-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


def render_confluence_gauge(score: int, direction: str) -> None:
    """Render Plotly gauge chart for confluence score."""
    color = "#00d084" if score >= 80 else ("#e0a96d" if score >= 60 else "#ff4d6d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confluence Score ({direction})", 'font': {'size': 18, 'color': '#c9d1d9'}},
        number={'suffix': "/100", 'font': {'size': 32, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#8b949e"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#161b22",
            'borderwidth': 2,
            'bordercolor': "#30363d",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(255, 77, 109, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(224, 169, 109, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(0, 208, 132, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#00d084", 'width': 4},
                'thickness': 0.8,
                'value': 80
            }
        }
    ))
    fig.update_layout(
        height=240,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pillar_breakdown_bars(pillars: Dict[str, int]) -> None:
    """Render horizontal bar chart for the 5 confluence pillars."""
    max_scores = {
        "Price Action & Momentum": 25,
        "Smart Money Flow": 25,
        "Order Flow Aggression": 20,
        "Structural Walls": 15,
        "Divergence & Traps": 15,
    }

    pillar_names = list(max_scores.keys())
    achieved = [pillars.get(p, 0) for p in pillar_names]
    max_vals = [max_scores[p] for p in pillar_names]
    pcts = [(achieved[i] / max_vals[i]) * 100 for i in range(len(pillar_names))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pillar_names,
        x=achieved,
        orientation='h',
        marker=dict(
            color=pcts,
            colorscale=[[0, '#ff4d6d'], [0.6, '#e0a96d'], [1, '#00d084']],
            cmin=0,
            cmax=100,
        ),
        text=[f"{achieved[i]} / {max_vals[i]} pts" for i in range(len(pillar_names))],
        textposition='inside',
    ))

    fig.update_layout(
        title="🏛️ Five-Pillar Confluence Breakdown",
        xaxis=dict(range=[0, 26], title="Points Achieved"),
        yaxis=dict(autorange="reversed"),
        height=240,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sniper_signal_card(signal: ChiefTradeSignal) -> None:
    """Render an actionable high-conviction sniper trade card."""
    is_call = "CALL" in signal.direction
    color = "#00d084" if is_call else "#ff4d6d"
    action_text = "🟢 BUY CALL (CE)" if is_call else "🔴 BUY PUT (PE)"

    st.markdown(f"""
    <div class="sniper-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0; color: {color};">🎯 {signal.symbol} — {action_text}</h3>
            <span style="background: rgba(0, 208, 132, 0.2); color: {color}; padding: 6px 14px; border-radius: 20px; font-weight: bold;">
                Confluence: {signal.confluence_score}/100 🔥
            </span>
        </div>
        <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 15px; color: #f0f6fc;">
            Contract: <code>{signal.strike_name}</code>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; background: #0d1117; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
            <div>
                <span style="color: #8b949e; font-size: 0.85rem;">Spot Entry Trigger</span><br>
                <b style="font-size: 1.15rem; color: #f0f6fc;">₹{signal.spot_entry_trigger:,.2f}</b>
            </div>
            <div>
                <span style="color: #8b949e; font-size: 0.85rem;">Spot Stop Loss</span><br>
                <b style="font-size: 1.15rem; color: #ff4d6d;">₹{signal.spot_stop_loss:,.2f}</b>
            </div>
            <div>
                <span style="color: #8b949e; font-size: 0.85rem;">Target 1 / Target 2</span><br>
                <b style="font-size: 1.15rem; color: #00d084;">₹{signal.target_1:,.2f} / ₹{signal.target_2:,.2f}</b>
            </div>
            <div>
                <span style="color: #8b949e; font-size: 0.85rem;">Risk : Reward</span><br>
                <b style="font-size: 1.15rem; color: #00b4d8;">{signal.risk_reward}</b>
            </div>
        </div>
        <div style="color: #c9d1d9; font-size: 0.95rem; line-height: 1.5;">
            <b>🧠 Smart Money Rationale:</b> {signal.primary_thesis}
        </div>
    </div>
    """, unsafe_allow_html=True)


def run_chief_agent_dashboard() -> None:
    """Main rendering entrypoint for Chief Trading Agent dashboard."""
    inject_chief_agent_css()

    st.title("🎯 Chief Trading Agent: Autonomous Sniper Decision Center")
    st.caption("Fuses Price Action, 3m/15m Supertrend, Smart OI 4-Quadrant Buildup, ATM Volume Delta, Liquidity Walls, and Trap Filters into maximum 2-3 sniper trades per day.")
    st.markdown("---")

    agent = ChiefTradingAgent()

    # 1. Continuous Daemon Pulse Banner
    live_state_path = agent.state_file.parent / "chief_agent_live_state.json"
    live_state = {}
    if live_state_path.exists():
        try:
            import json
            with open(live_state_path, "r") as f:
                live_state = json.load(f)
        except Exception:
            live_state = {}

    last_ts = live_state.get("timestamp", "")
    is_live = bool(last_ts)
    vix_data = live_state.get("vix", {})

    v_col1, v_col2 = st.columns([2, 1.5])
    with v_col1:
        st.markdown(f"""
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: {'#00d084' if is_live else '#e0a96d'}; font-weight: bold; font-size: 1.05rem;">
                        {'🟢 Continuous Autonomous Engine: ACTIVE' if is_live else '🟡 Continuous Engine: STANDBY'}
                    </span><br>
                    <span style="color: #8b949e; font-size: 0.85rem;">
                        Last Evaluation: <b>{last_ts[:19].replace('T', ' ') if last_ts else 'No cycles recorded yet'}</b> | Daemon Interval: <b>120s</b>
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with v_col2:
        v_lvl = vix_data.get("vix_level", 0.0)
        v_chp = vix_data.get("vix_change_pct", 0.0)
        v_reg = vix_data.get("vix_regime", "NORMAL")
        st.markdown(f"""
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-bottom: 15px;">
            <span style="color: #8b949e; font-size: 0.85rem;">India VIX Dynamics</span><br>
            <b style="font-size: 1.1rem; color: {'#00d084' if v_chp < 0 else '#ff4d6d'};">
                {v_lvl:.2f} ({v_chp:+.2f}%)
            </b>
            <span style="color: #8b949e; font-size: 0.8rem; margin-left: 8px;">{v_reg.split('—')[0]}</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. Daily Trade Governor Budget Bar
    idx_ok, idx_count, idx_max, idx_msg = agent.check_daily_trade_budget("INDEX")
    stk_ok, stk_count, stk_max, stk_msg = agent.check_daily_trade_budget("STOCK")

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.metric(
            "🛡️ Indices Budget Today",
            f"{idx_count} / {idx_max} Trades",
            delta="Hunting Active" if idx_ok else "Daily Cap Reached",
            delta_color="normal" if idx_ok else "inverse"
        )
    with col_b2:
        st.metric(
            "🏢 F&O Stocks Budget",
            f"{stk_count} / {stk_max} Trades",
            delta="Hunting Active" if stk_ok else "Daily Cap Reached",
            delta_color="normal" if stk_ok else "inverse"
        )
    with col_b3:
        st.metric("🎯 Entry Gate Threshold", ">= 80 / 100 Pts", help="Strict institutional bar: Only fires when >= 80% confluence and zero vetoes.")
    with col_b4:
        st.metric("⏳ Trade Cooldown", f"{agent.COOLDOWN_MINUTES} Mins", help="Enforces a minimum wait period between trades to prevent overtrading.")

    st.markdown("---")

    # 3. Controls Bar
    c_cat, c_sym, c_btn1, c_btn2, c_btn3 = st.columns([1.1, 1.6, 1.1, 1.4, 1.3])
    with c_cat:
        asset_class = st.selectbox("Asset Class", ["📊 Major Indices", "🏢 F&O Stock Universe"], index=0, key="chief_asset_class")

    with c_sym:
        if asset_class == "📊 Major Indices":
            index_options = {
                "NIFTY 50": "NSE:NIFTY50-INDEX",
                "BANK NIFTY": "NSE:NIFTYBANK-INDEX",
                "FINNIFTY": "NSE:FINNIFTY-INDEX",
                "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
                "SENSEX": "BSE:SENSEX-INDEX",
            }
            selected_label = st.selectbox("Select Underlying", list(index_options.keys()), index=0, key="chief_index_select")
            eval_symbol = index_options[selected_label]
        else:
            try:
                fo_list = get_fo_universe()
            except Exception:
                fo_list = ["NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:KEI-EQ", "NSE:TATAMOTORS-EQ"]
            stock_map = {s.split(":")[-1].replace("-EQ", ""): s for s in sorted(fo_list)}
            selected_label = st.selectbox("Select F&O Stock", list(stock_map.keys()), index=0, key="chief_stock_select")
            eval_symbol = stock_map[selected_label]

    with c_btn1:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        eval_btn = st.button("🎯 Evaluate Symbol", key="btn_chief_eval", use_container_width=True)

    with c_btn2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        scan_all_indices_btn = st.button("⚡ Scan All Indices", key="btn_chief_scan_indices", use_container_width=True)

    with c_btn3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        run_cycle_btn = st.button("🔄 Trigger Continuous Cycle", key="btn_chief_run_cycle", use_container_width=True)

    # 4. Execution & Evaluation Logic
    from trade_system.domains.advisory.application.agent.chief_trading_agent import ContinuousChiefAgent

    if run_cycle_btn:
        with st.spinner("Executing full autonomous continuous cycle..."):
            daemon = ContinuousChiefAgent(agent)
            cycle_res = daemon.run_cycle(symbols=["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"], dispatch_telegram=True)
            st.success("✅ Continuous cycle completed and live state updated!")
            st.rerun()
    if "chief_last_eval" not in st.session_state or eval_btn or scan_all_indices_btn:
        symbols_to_run = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"] if scan_all_indices_btn else [eval_symbol]
        results = []

        with st.spinner("Chief Agent evaluating live market confluence..."):
            for sym in symbols_to_run:
                try:
                    sig, scorecard = agent.evaluate_symbol(sym, strikecount=15, force_evaluation=True)
                    results.append((sym, sig, scorecard))
                except Exception as ex:
                    LOGGER.error("Chief Agent evaluation error for %s: %s", sym, ex)
                    results.append((sym, None, {"symbol": sym, "error": str(ex), "total_confluence_score": 0}))

        st.session_state["chief_last_eval"] = results

    eval_results = st.session_state.get("chief_last_eval", [])

    if not eval_results:
        st.info("Click '🎯 Evaluate Symbol' to run the Chief Agent.")
        return

    # 4. Display Results
    for sym, signal, scorecard in eval_results:
        score = scorecard.get("total_confluence_score", 0)
        direction = scorecard.get("direction", "NEUTRAL")
        pillars = scorecard.get("pillar_scores", {})
        vetoes = scorecard.get("veto_reasons", [])
        spot = scorecard.get("spot_price", 0.0)

        st.markdown(f"### 📍 Evaluation Scorecard: `{sym}` (Spot: ₹{spot:,.2f})")

        # Top Gauge and Pillar Breakdown
        col_g, col_p = st.columns([1, 1.5])
        with col_g:
            render_confluence_gauge(score, direction)
        with col_p:
            render_pillar_breakdown_bars(pillars)

        # Market Structure & Volatility Dynamics Row
        ms = scorecard.get("market_structure", {})
        if ms:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"**Market Structure State:**<br>`{ms.get('structure_state', 'N/A')}`", unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"**Wall Migration:**<br>`{ms.get('wall_migration', 'STABLE')}`", unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"**Taker Aggression:**<br>`{ms.get('taker_aggression', 'BALANCED')}`", unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"**ATM Volume Delta:**<br>`{ms.get('atm_volume_delta', 0):,}` contracts", unsafe_allow_html=True)

            # Expiry Day Dynamics & Gamma Radar Row
            if ms.get("is_expiry_day") or ms.get("is_expiry_eve") or ms.get("days_to_expiry", 5) <= 2:
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                with col_e1:
                    is_exp = ms.get("is_expiry_day", False)
                    dte_val = ms.get("days_to_expiry", 0)
                    badge_txt = "🔥 0DTE: TODAY IS EXPIRY DAY" if is_exp else (f"⏳ 1DTE (Expiry Eve)" if ms.get("is_expiry_eve") else f"{dte_val} Days to Expiry")
                    st.markdown(f"**Expiry Countdown:**<br><b style='color: {'#ff4d6d' if is_exp else '#00d084'};'>{badge_txt}</b>", unsafe_allow_html=True)
                with col_e2:
                    mp = ms.get("max_pain_strike", 0.0)
                    dist_mp = ms.get("dist_to_max_pain", 0.0)
                    st.markdown(f"**Max Pain Strike:**<br>₹{mp:,.0f} (Dist: {dist_mp:+.1f} pts)", unsafe_allow_html=True)
                with col_e3:
                    phase_txt = ms.get("expiry_phase", "NORMAL").split('(')[0].strip()
                    st.markdown(f"**Expiry Phase:**<br>`{phase_txt}`", unsafe_allow_html=True)
                with col_e4:
                    rec_c = ms.get("recommended_contract_type", "CURRENT_EXPIRY_ATM")
                    st.markdown(f"**Contract Strategy:**<br><b>{rec_c}</b>", unsafe_allow_html=True)

                risk_al = ms.get("expiry_risk_alert", "")
                if risk_al and "⚠️" in risk_al:
                    st.warning(risk_al)
                elif risk_al and "🚀" in risk_al:
                    st.info(risk_al)

            # AMD (Wyckoff / PO3) Market Cycle Row
            amd_p = ms.get("amd_phase", "CONSOLIDATION")
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                if "SPRING" in amd_p:
                    badge = "⚡ MANIPULATION (Spring Sweep Reclaim!)"
                    p_col = "#00f5d4"
                elif "UTAD" in amd_p:
                    badge = "⚡ MANIPULATION (Upthrust / UTAD Reject!)"
                    p_col = "#ff4d6d"
                elif "DISTRIBUTION" in amd_p:
                    badge = "🚀 DISTRIBUTION (Expansion Trend)"
                    p_col = "#70d6ff"
                else:
                    badge = "📦 PHASE 1: ACCUMULATION (Compression)"
                    p_col = "#ffd166"
                st.markdown(f"**AMD Cycle (Wyckoff/PO3):**<br><b style='color: {p_col};'>{badge}</b>", unsafe_allow_html=True)
            with col_a2:
                r_low = ms.get("amd_range_low", 0.0)
                r_high = ms.get("amd_range_high", 0.0)
                st.markdown(f"**Accumulation Range:**<br>₹{r_low:,.1f} — ₹{r_high:,.1f}", unsafe_allow_html=True)
            with col_a3:
                m_level = ms.get("amd_manipulation_level", 0.0)
                st.markdown(f"**Sweep / Liquidity Level:**<br>₹{m_level:,.1f}", unsafe_allow_html=True)
            with col_a4:
                rec_act = ms.get("amd_action", "STAND_ASIDE_ACCUMULATION")
                st.markdown(f"**Cycle Action:**<br>`{rec_act}`", unsafe_allow_html=True)

            amd_desc = ms.get("amd_description", "")
            if amd_desc:
                if "⚡" in amd_desc:
                    st.success(amd_desc)
                elif "📦" in amd_desc:
                    st.info(amd_desc)
                elif "🚀" in amd_desc:
                    st.info(amd_desc)

        # Signal or Stand Aside Notice
        if signal:
            render_sniper_signal_card(signal)

            # Option to dispatch Telegram alert
            col_t1, col_t2 = st.columns([2, 1])
            with col_t2:
                if st.button(f"📢 Dispatch {sym} Signal to Telegram", key=f"tg_{sym}", use_container_width=True):
                    sent = agent.dispatch_telegram_alert(signal)
                    if sent:
                        st.success("✅ Telegram alert dispatched successfully!")
                    else:
                        st.error("❌ Failed to dispatch Telegram alert.")
        else:
            st.markdown(f"""
            <div class="stand-aside-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #8b949e;">🛑 Status: STAND ASIDE / NO TRADE</h4>
                    <span style="color: #8b949e;">Score: {score}/100 (Threshold: 80)</span>
                </div>
                <p style="color: #c9d1d9; margin-top: 8px; margin-bottom: 0;">
                    <b>Gatekeeping Reason:</b> {vetoes[0] if vetoes else 'Overall confluence score is below the institutional 80% threshold.'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        if vetoes:
            with st.expander(f"🔍 View Active Gatekeeping Vetoes for {sym}"):
                for v in vetoes:
                    st.write(f"• {v}")

        st.markdown("---")

    # 5. Today's Executed Signal Ledger
    st.subheader("📋 Today's Trade Execution Ledger")
    state_data = agent._load_state()
    executed_trades = state_data.get("executed_trades", [])

    if executed_trades:
        ledger_df = pd.DataFrame(executed_trades)
        st.dataframe(ledger_df[[
            "timestamp", "symbol", "direction", "strike_name", "spot_entry_trigger", "spot_stop_loss", "target_1", "risk_reward", "confluence_score"
        ]], use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed yet today. The Chief Agent will record executed trades here to enforce the daily 2-3 trade limit.")


run_chief_agent_dashboard()

