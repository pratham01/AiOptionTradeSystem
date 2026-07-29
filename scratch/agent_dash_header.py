"""
Agent Dashboard — Streamlit page for the agentic trading system.

Shows:
  - Today's SessionPlan with all suggestions
  - Live market context (VIX, PCR, regime)
  - Evolution metrics (win rate trend, weight changes)
  - Pending/closed trade outcomes
  - Feature importance chart

Run with: streamlit run -m trade_system.dashboard.agent_dashboard
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Page Config (handled by main.py)

# --- THEME & CSS ---
st.markdown("""
<style>
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(90deg, #1e2130 0%, #2b3044 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 8px solid #00b4d8;
    }
    .status-active { color: #00d084; font-weight: bold; }
    
    /* Modern Trade Cards */
    .trade-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.2s;
    }
    .trade-card:hover { transform: translateY(-2px); border-color: rgba(0, 180, 216, 0.3); }
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .call-tag { border-left: 5px solid #00d084; }
    .put-tag { border-left: 5px solid #ff4d6d; }
    
    /* Utility */
    .metric-sub { font-size: 0.85rem; color: #888; }
    .price-tag { font-family: monospace; font-size: 1.1rem; color: #00b4d8; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _render_trade_cards(trades):
    if not trades:
        st.info("No active setups identified by the Swarm yet.")
        return
    
    for t in trades:
        tag_cls = "call-tag" if t.direction == "CALL" else "put-tag"
        icon = "🟢" if t.direction == "CALL" else "🔴"
        
        sym_short = t.symbol.split(":")[-1].replace("-EQ", "").replace("-INDEX", "")
        
        st.markdown(f"""
        <div class="trade-card {tag_cls}">
            <div class="card-header">
                <span style="font-size:1.2rem; font-weight:bold;">{icon} {sym_short} — BUY {t.direction}</span>
                <span class="metric-sub">{t.horizon}</span>
            </div>
            <div style="display:flex; gap:20px; margin: 10px 0;">
                <div><span class="metric-sub">Entry Zone</span><br/><span class="price-tag">₹{t.entry_zone_low:.0f}-{t.entry_zone_high:.0f}</span></div>
                <div><span class="metric-sub">Target</span><br/><span class="price-tag" style="color:#00d084">₹{t.target:.0f}</span></div>
                <div><span class="metric-sub">Stop Loss</span><br/><span class="price-tag" style="color:#ff4d6d">₹{t.stop_loss:.0f}</span></div>
            </div>
            <div style="font-size:0.9rem; opacity:0.8; margin-top:10px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
                <b>Reasoning:</b> {t.narrative}
                <br/>
                <span class="metric-sub">Confidence: {t.confidence:.0%} | Strike: ₹{t.option_strike or 0:.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def load_session_plan():
    """Try to load today's session plan from DB or file."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from trade_system.domains.market_data.infrastructure.database.models import SuggestedTrade
        from sqlalchemy.orm import Session
        engine = get_engine()
        today = date.today().isoformat()
        with Session(engine) as session:
            trades = (
                session.query(SuggestedTrade)
                .filter(SuggestedTrade.date == today)
                .order_by(SuggestedTrade.generated_at.desc())
                .all()
            )
        return trades
    except Exception:
        return []


def load_closed_trades(lookback_days: int = 30):
    """Load closed trades for performance metrics."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from trade_system.domains.market_data.infrastructure.database.models import SuggestedTrade
        from sqlalchemy.orm import Session
        engine = get_engine()
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        with Session(engine) as session:
            return (
                session.query(SuggestedTrade)
                .filter(SuggestedTrade.outcome != "PENDING")
                .filter(SuggestedTrade.date >= cutoff)
                .order_by(SuggestedTrade.date.desc())
                .all()
            )
    except Exception:
        return []


def load_agent_weights():
    """Load current agent weights."""
    try:
        from trade_system.domains.analysis.application.evolution.weight_evolver import WeightEvolver
        evolver = WeightEvolver()
        return {
            "candidate_screener": evolver.load_weights("candidate_screener"),
            "setup_validator": evolver.load_weights("setup_validator"),
        }
    except Exception:
        return {}


def load_evolution_reports(limit: int = 10):
    """Load recent evolution reports."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from trade_system.domains.market_data.infrastructure.database.models import EvolutionReport
        from sqlalchemy.orm import Session
        engine = get_engine()
        with Session(engine) as session:
            return (
                session.query(EvolutionReport)
                .order_by(EvolutionReport.run_at.desc())
                .limit(limit)
                .all()
            )
    except Exception:
        return []


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 AI Trade Agent")
    st.markdown("---")
    lookback_days = st.slider("Performance Lookback (days)", 7, 90, 30)

    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("🔄 Run Orchestrator Now", type="primary"):
        with st.spinner("Running agentic pipeline..."):
            try:
                from trade_system.domains.advisory.application.agent.orchestrator import TradeOrchestrator
                orc = TradeOrchestrator()
                plan = orc.run_session_sync()
                st.success(f"✅ Generated {len(plan.all_suggestions())} suggestions!")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("📊 Run Evolution Loop"):
        with st.spinner("Running evolution loop..."):
            try:
                from trade_system.domains.analysis.application.evolution.evolution_loop import EvolutionLoop
                loop = EvolutionLoop()
                result = loop.run()
                st.success(f"✅ Win Rate: {result['win_rate']:.1%}")
            except Exception as e:
                st.error(f"Failed: {e}")


def run_main():
    """Entry point for trade-dashboard script."""
    import subprocess
    import sys
    from pathlib import Path
    file_path = Path(__file__).resolve()
