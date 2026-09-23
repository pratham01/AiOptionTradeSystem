"""
Autonomous Execution Cockpit — Institutional Trade Management Dashboard
Provides real-time visibility into autonomous strategy executions (SMC, Smart OI Divergence,
Midday Breakout), live active positions, dynamic 2-stage trailing stops, PnL analytics,
emergency panic controls, and an interactive simulation test rig.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trade_system.shared.config import Settings
from trade_system.domains.trading.application.execution.position_manager import PositionManager, OpenPosition
from trade_system.domains.trading.application.execution.execution_engine import ExecutionEngine
from trade_system.domains.trading.application.execution.autonomous_router import AutonomousExecutionRouter
from trade_system.domains.advisory.application.agent.risk_manager import RiskManager
from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBrokerClient

LOGGER = logging.getLogger(__name__)

# --- CLEAN HTML HELPER ---
def _clean_html(html_str: str) -> str:
    """Strips leading whitespace from every line to avoid markdown code block interpretation."""
    return "\n".join(line.lstrip() for line in html_str.strip().splitlines())


# --- STATE LOADER ---
def _load_autonomous_state() -> Dict[str, Any]:
    """Loads active positions and trade history from data/autonomous_trades.json."""
    settings = Settings.load()
    state_file = settings.data_dir / "autonomous_trades.json"
    if not state_file.exists():
        return {"active_positions": [], "trade_history": [], "dry_run": True, "last_updated": ""}
    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception as exc:
        LOGGER.warning("Could not read autonomous trades state: %s", exc)
        return {"active_positions": [], "trade_history": [], "dry_run": True, "last_updated": ""}


class LightweightSimBroker:
    """Fallback simulated broker for dashboard and paper trading."""
    def place_order(self, symbol, side, quantity, order_type, product_type, price=0.0, stoploss=0.0):
        return {
            "s": "ok",
            "id": f"sim_{datetime.now().strftime('%H%M%S')}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price
        }

    def get_funds(self):
        return {"fund_limit": [{"equityAmount": 1000000.0}]}


def _get_live_position_manager() -> PositionManager:
    """Instantiates a lightweight PositionManager wired to persistent state."""
    from trade_system.interfaces.dashboard.shared_broker import get_cached_broker
    settings = Settings.load()
    risk_manager = RiskManager()
    broker = None
    try:
        broker = get_cached_broker()
    except Exception as exc:
        LOGGER.debug("Could not get cached broker: %s", exc)
    if not broker:
        broker = LightweightSimBroker()
    return PositionManager(broker=broker, risk_manager=risk_manager, settings=settings)


# --- MAIN RENDER FUNCTION ---
def render_autonomous_execution_dashboard() -> None:
    st.markdown(
        """
        <style>
            .stApp { background-color: #0b0e14; color: #f0f6fc; }
            .metric-card {
                background: linear-gradient(135deg, rgba(22, 27, 34, 0.95), rgba(13, 17, 23, 0.95));
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 16px 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            }
            .position-card {
                background: rgba(22, 27, 34, 0.85);
                border-left: 4px solid #00d2ff;
                border-radius: 10px;
                padding: 16px 20px;
                margin-bottom: 14px;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
                border-right: 1px solid rgba(255, 255, 255, 0.06);
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }
            .position-card.long { border-left-color: #00e676; }
            .position-card.short { border-left-color: #ff3b30; }
            .badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            .badge-smc { background: rgba(157, 78, 221, 0.25); color: #d084ff; border: 1px solid rgba(157, 78, 221, 0.5); }
            .badge-smartoi { background: rgba(0, 210, 255, 0.2); color: #00d2ff; border: 1px solid rgba(0, 210, 255, 0.4); }
            .badge-breakout { background: rgba(255, 170, 0, 0.2); color: #ffb703; border: 1px solid rgba(255, 170, 0, 0.4); }
            .badge-locked { background: rgba(0, 230, 118, 0.2); color: #00e676; border: 1px solid rgba(0, 230, 118, 0.4); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Header & Live Mode Banner
    state = _load_autonomous_state()
    active_positions = state.get("active_positions", [])
    trade_history = state.get("trade_history", [])
    is_dry_run = state.get("dry_run", True)

    col_title, col_actions = st.columns([3, 1.2])
    with col_title:
        st.markdown(
            _clean_html("""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #f0f6fc;">
                    ⚡ Autonomous Execution Cockpit
                </h1>
                <span class="badge" style="background: rgba(0, 230, 118, 0.15); color: #00e676; border: 1px solid #00e676;">
                    LIVE AGENTS ACTIVE
                </span>
            </div>
            <p style="margin: 0 0 16px 0; font-size: 13px; color: #8b949e;">
                Autonomous Strategy Wiring: Photon SMC, StockMojo Smart OI Divergence & Midday Breakouts with 2-Stage Trailing Stops.
            </p>
            """),
            unsafe_allow_html=True,
        )

    with col_actions:
        mode_label = "🟢 PAPER TRADING (DRY RUN)" if is_dry_run else "🔴 LIVE BROKER MODE"
        mode_color = "#00e676" if is_dry_run else "#ff3b30"
        st.markdown(
            _clean_html(f"""
            <div style="text-align: right; margin-top: 4px;">
                <div style="font-size: 12px; font-weight: 700; color: {mode_color}; letter-spacing: 0.5px;">
                    {mode_label}
                </div>
                <div style="font-size: 11px; color: #6e7681;">Updated: {state.get('last_updated', 'Just now')[:19].replace('T', ' ')}</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    # 2. Executive KPI Deck
    total_realized_pnl = sum(t.get("pnl", 0.0) for t in trade_history)
    total_unrealized_pnl = sum(p.get("unrealized_pnl", 0.0) for p in active_positions)
    net_pnl = total_realized_pnl + total_unrealized_pnl
    win_trades = [t for t in trade_history if t.get("is_win", False)]
    win_rate = (len(win_trades) / len(trade_history) * 100.0) if trade_history else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        pnl_color = "#00e676" if net_pnl >= 0 else "#ff3b30"
        st.markdown(
            _clean_html(f"""
            <div class="metric-card">
                <div style="font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase;">Total Net PnL</div>
                <div style="font-size: 22px; font-weight: 800; color: {pnl_color}; margin-top: 4px;">₹{net_pnl:+,.2f}</div>
                <div style="font-size: 11px; color: #8b949e; margin-top: 2px;">Realized + Unrealized</div>
            </div>
            """),
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            _clean_html(f"""
            <div class="metric-card">
                <div style="font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase;">Active Positions</div>
                <div style="font-size: 22px; font-weight: 800; color: #00d2ff; margin-top: 4px;">{len(active_positions)} / 5</div>
                <div style="font-size: 11px; color: #8b949e; margin-top: 2px;">Max 5 concurrent</div>
            </div>
            """),
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            _clean_html(f"""
            <div class="metric-card">
                <div style="font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase;">Realized PnL</div>
                <div style="font-size: 22px; font-weight: 800; color: {'#00e676' if total_realized_pnl >= 0 else '#ff3b30'}; margin-top: 4px;">₹{total_realized_pnl:+,.2f}</div>
                <div style="font-size: 11px; color: #8b949e; margin-top: 2px;">{len(trade_history)} trades closed</div>
            </div>
            """),
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            _clean_html(f"""
            <div class="metric-card">
                <div style="font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase;">Win Rate</div>
                <div style="font-size: 22px; font-weight: 800; color: #ffb703; margin-top: 4px;">{win_rate:.1f}%</div>
                <div style="font-size: 11px; color: #8b949e; margin-top: 2px;">{len(win_trades)}W - {len(trade_history) - len(win_trades)}L</div>
            </div>
            """),
            unsafe_allow_html=True,
        )
    with kpi5:
        st.markdown(
            _clean_html(f"""
            <div class="metric-card">
                <div style="font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase;">Unrealized PnL</div>
                <div style="font-size: 22px; font-weight: 800; color: {'#00e676' if total_unrealized_pnl >= 0 else '#ff3b30'}; margin-top: 4px;">₹{total_unrealized_pnl:+,.2f}</div>
                <div style="font-size: 11px; color: #8b949e; margin-top: 2px;">Mark-to-Market</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. Emergency Controls Ribbon
    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1.5, 3])
    with c_btn1:
        if st.button("🚨 PANIC FLATTEN ALL POSITIONS", use_container_width=True, type="secondary"):
            pm = _get_live_position_manager()
            flushed = pm.flatten_all_positions(reason="DASHBOARD_PANIC_FLATTEN")
            st.toast(f"🚨 Flattened {len(flushed)} active positions!", icon="🚨")
            st.rerun()
    with c_btn2:
        if st.button("🔄 Refresh State & PnL", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # 4. Main Tabs
    tab_positions, tab_sandbox, tab_history = st.tabs([
        f"📊 Active Positions ({len(active_positions)})",
        "🧪 1-Click Simulation Sandbox",
        f"📜 Trade History & Audit ({len(trade_history)})"
    ])

    # -----------------------------------------------------------------------
    # TAB 1: ACTIVE POSITIONS MATRIX
    # -----------------------------------------------------------------------
    with tab_positions:
        if not active_positions:
            st.markdown(
                _clean_html("""
                <div style="background: rgba(22, 27, 34, 0.6); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 40px 20px; text-align: center; margin: 20px 0;">
                    <div style="font-size: 32px; margin-bottom: 8px;">🛡️</div>
                    <div style="font-size: 16px; font-weight: 700; color: #f0f6fc;">Zero Open Exposure — All Capital Protected</div>
                    <p style="font-size: 13px; color: #8b949e; max-width: 500px; margin: 6px auto 16px auto;">
                        Autonomous strategy routers are continuously scanning real-time order flow and option chain snapshots for high-conviction SMC structure realignments and Smart OI divergences.
                    </p>
                    <div style="font-size: 12px; color: #00d2ff;">
                        💡 You can use the <b>1-Click Simulation Sandbox</b> tab to simulate an execution and verify trailing stops!
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )
        else:
            for pos in active_positions:
                sym = pos.get("symbol", "")
                strat = pos.get("strategy_name", "AUTONOMOUS")
                direction = pos.get("direction", "LONG")
                side_class = "long" if direction == "LONG" else "short"
                side_color = "#00e676" if direction == "LONG" else "#ff3b30"
                entry = pos.get("entry_price", 0.0)
                ltp = pos.get("current_price", entry)
                sl = pos.get("current_stop_loss", 0.0)
                tp1 = pos.get("target_1", 0.0)
                tp2 = pos.get("target_2", 0.0)
                qty = pos.get("quantity", 0)
                unrealized = pos.get("unrealized_pnl", 0.0)
                unrealized_pct = ((ltp - entry) / entry * 100.0) if direction == "LONG" else ((entry - ltp) / entry * 100.0)
                is_be_locked = pos.get("is_breakeven_locked", False)

                strat_badge = "badge-smc" if "SMC" in strat else ("badge-smartoi" if "OI" in strat else "badge-breakout")

                card_col1, card_col2 = st.columns([4, 1])
                with card_col1:
                    lock_html = '<span class="badge badge-locked">🛡️ BREAKEVEN LOCKED</span>' if is_be_locked else '<span class="badge" style="background: rgba(255, 255, 255, 0.08); color: #8b949e;">TARGET 1 HUNTING</span>'
                    pnl_disp = f"₹{unrealized:+,.2f} ({unrealized_pct:+.2f}%)"
                    pnl_style_color = "#00e676" if unrealized >= 0 else "#ff3b30"

                    st.markdown(
                        _clean_html(f"""
                        <div class="position-card {side_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 16px; font-weight: 800; color: #f0f6fc;">{sym}</span>
                                    <span class="badge" style="background: rgba({ '0, 230, 118, 0.2' if direction == 'LONG' else '255, 59, 48, 0.2' }); color: {side_color};">
                                        {direction} {qty} Qty
                                    </span>
                                    <span class="badge {strat_badge}">{strat}</span>
                                    {lock_html}
                                </div>
                                <div style="font-size: 18px; font-weight: 800; color: {pnl_style_color};">
                                    {pnl_disp}
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; background: rgba(0, 0, 0, 0.2); padding: 10px 14px; border-radius: 8px; font-size: 12px;">
                                <div><span style="color: #8b949e;">Entry:</span> <b style="color: #f0f6fc;">₹{entry:,.2f}</b></div>
                                <div><span style="color: #8b949e;">LTP:</span> <b style="color: {side_color};">₹{ltp:,.2f}</b></div>
                                <div><span style="color: #8b949e;">Trailed SL:</span> <b style="color: #ff3b30;">₹{sl:,.2f}</b></div>
                                <div><span style="color: #8b949e;">TP1 (Range):</span> <b style="color: #ffb703;">₹{tp1:,.2f}</b></div>
                                <div><span style="color: #8b949e;">TP2 (Target):</span> <b style="color: #00e676;">₹{tp2:,.2f}</b></div>
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True,
                    )

                with card_col2:
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    if st.button(f"🛑 Close {sym.split(':')[-1]}", key=f"close_{sym}", use_container_width=True):
                        pm = _get_live_position_manager()
                        pm.manual_close_position(sym, exit_price=ltp, reason="MANUAL_DASHBOARD_EXIT")
                        st.toast(f"Closed {sym} at ₹{ltp:.2f}!", icon="✅")
                        st.rerun()

                    # Quick simulation tick button
                    if st.button(f"⚡ Tick to TP1", key=f"tick_tp1_{sym}", use_container_width=True, help="Simulate tick price touching Target 1 to verify Breakeven lock"):
                        pm = _get_live_position_manager()
                        pm.process_tick(sym, tp1 + 2.0)
                        st.toast(f"Simulated price tick ₹{tp1+2.0:.2f} -> TP1 reached!", icon="🎯")
                        st.rerun()

                    if st.button(f"⚡ Tick to TP2", key=f"tick_tp2_{sym}", use_container_width=True, help="Simulate tick price touching Target 2 to verify profit exit"):
                        pm = _get_live_position_manager()
                        pm.process_tick(sym, tp2 + 2.0)
                        st.toast(f"Simulated price tick ₹{tp2+2.0:.2f} -> TP2 reached!", icon="🚀")
                        st.rerun()

    # -----------------------------------------------------------------------
    # TAB 2: 1-CLICK SIMULATION SANDBOX
    # -----------------------------------------------------------------------
    with tab_sandbox:
        st.markdown(
            _clean_html("""
            <div style="margin-bottom: 16px;">
                <h3 style="margin: 0; font-size: 16px; color: #f0f6fc;">🧪 1-Click Strategy Execution Sandbox</h3>
                <p style="margin: 4px 0 0 0; font-size: 12px; color: #8b949e;">
                    Instantly simulate order execution for SMC setups, Smart OI Divergences, or Midday Breakouts.
                    Verify position creation, risk checks, dynamic trailing stops, and exit mechanics end-to-end.
                </p>
            </div>
            """),
            unsafe_allow_html=True,
        )

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            test_symbol = st.selectbox(
                "Symbol",
                ["NSE:NIFTY50-INDEX", "NSE:BANKNIFTY-INDEX", "BSE:SENSEX-INDEX", "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ"],
                index=0,
            )
            test_strategy = st.selectbox(
                "Strategy Attribution",
                ["SMC_INSTITUTIONAL", "SMART_OI_DIVERGENCE", "MIDDAY_BREAKOUT"],
                index=0,
            )
        with sim_col2:
            test_direction = st.radio("Trade Direction", ["LONG / BUY", "SHORT / SELL"], horizontal=True)
            dir_val = 1 if "LONG" in test_direction else -1
            default_entry = 25150.0 if "NIFTY" in test_symbol else (54200.0 if "BANKNIFTY" in test_symbol else 2950.0)
            test_entry = st.number_input("Simulated Entry Price (₹)", value=float(default_entry), step=10.0)
        with sim_col3:
            default_sl = round(test_entry - 50.0, 2) if dir_val == 1 else round(test_entry + 50.0, 2)
            default_tp1 = round(test_entry + 75.0, 2) if dir_val == 1 else round(test_entry - 75.0, 2)
            default_tp2 = round(test_entry + 150.0, 2) if dir_val == 1 else round(test_entry - 150.0, 2)
            test_sl = st.number_input("Stop Loss (₹)", value=float(default_sl), step=5.0)
            test_tp1 = st.number_input("Target 1 (Range / TP1) (₹)", value=float(default_tp1), step=5.0)
            test_tp2 = st.number_input("Target 2 (Swing / TP2) (₹)", value=float(default_tp2), step=5.0)

        if st.button("🚀 Execute Simulated Test Trade", type="primary", use_container_width=True):
            pm = _get_live_position_manager()
            engine = ExecutionEngine(broker=pm.broker, risk_manager=pm.risk_manager)
            router = AutonomousExecutionRouter(execution_engine=engine, position_manager=pm, risk_manager=pm.risk_manager)
            qty = 50 if "NIFTY" in test_symbol else 15
            res = router.simulate_test_trade(
                symbol=test_symbol,
                strategy_name=test_strategy,
                direction=dir_val,
                entry_price=test_entry,
                stop_loss=test_sl,
                target_1=test_tp1,
                target_2=test_tp2,
                quantity=qty,
            )
            st.success(f"✅ Successfully executed test trade for {test_symbol} via {test_strategy}!")
            st.rerun()

    # -----------------------------------------------------------------------
    # TAB 3: TRADE HISTORY & AUDIT TRAIL
    # -----------------------------------------------------------------------
    with tab_history:
        if not trade_history:
            st.info("No trades closed yet in this session.")
        else:
            hist_df = pd.DataFrame(trade_history)
            
            # Cumulative PnL chart
            if "pnl" in hist_df.columns:
                hist_df["cum_pnl"] = hist_df["pnl"].cumsum()
                fig_pnl = px.line(
                    hist_df,
                    x=range(len(hist_df)),
                    y="cum_pnl",
                    title="📈 Cumulative Autonomous Strategy Equity Curve (₹)",
                    labels={"x": "Trade #", "cum_pnl": "Net PnL (₹)"},
                )
                fig_pnl.update_traces(line_color="#00e676" if hist_df["cum_pnl"].iloc[-1] >= 0 else "#ff3b30", line_width=2.5)
                fig_pnl.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

            # Historical Trades Table
            display_cols = ["exit_time", "symbol", "strategy", "side", "quantity", "entry_price", "exit_price", "pnl", "pnl_pct", "exit_reason"]
            avail_cols = [c for c in display_cols if c in hist_df.columns]
            st.dataframe(
                hist_df[avail_cols].iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    render_autonomous_execution_dashboard()
