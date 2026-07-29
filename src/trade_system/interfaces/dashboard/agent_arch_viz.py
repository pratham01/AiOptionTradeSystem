import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# from trade_system.knowledge_graph.engine import MarketKnowledgeGraph (Module missing)
class MockGraph:
    def nodes(self, data=False): return []
    def edges(self, data=False): return []

class MarketKnowledgeGraph:
    """Mock for missing Knowledge Graph engine."""
    def __init__(self):
        self.graph = MockGraph()
    def add_news_event(self, event, sectors): pass
    def get_impact_path(self, event): return []
    def get_all_sectors(self): return ["IT", "BANKING", "AUTO", "METALS", "ENERGY", "PHARMA", "CONSUMER"]

from trade_system.domains.market_data.infrastructure.data.fo_universe import FO_METADATA

# Page Config (handled by main.py)

# Custom CSS for a "Neural" Look
st.markdown("""
    <style>
    .agent-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #00b4d8;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .agent-card h3 {
        margin-top: 0;
        color: #00b4d8;
        font-size: 1.1rem;
    }
    .observation {
        color: #888;
        font-size: 0.9rem;
    }
    .decision {
        color: #00f5d4;
        font-weight: bold;
    }
    .thought-terminal {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #30363d;
        height: 400px;
        overflow-y: scroll;
        margin-bottom: 20px;
    }
    .thought-line {
        margin-bottom: 5px;
        font-size: 0.85rem;
    }
    .timestamp { color: #8b949e; }
    .agent-name { color: #58a6ff; font-weight: bold; }
    .action { color: #aff5b4; }
    .msg { color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

def load_agent_stats():
    """Load neural weights and latest feedback."""
    weights = {}
    try:
        from trade_system.domains.analysis.application.evolution.weight_evolver import WeightEvolver
        evolver = WeightEvolver()
        weights = evolver.load_weights("setup_validator")
    except Exception:
        pass
    
    feedback = []
    if Path("data/agent_feedback.json").exists():
        try:
            feedback = json.loads(Path("data/agent_feedback.json").read_text())
        except Exception: pass
    
    return weights, feedback

def load_thought_stream(limit=100):
    """Load latest deliberations from the database."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from trade_system.domains.market_data.infrastructure.database.repository import get_latest_thoughts
        from sqlalchemy.orm import Session
        engine = get_engine()
        with Session(engine) as session:
            return get_latest_thoughts(session, limit=limit)
    except Exception:
        return []

st.title("🤖 Swarm Intelligence & Thought Stream")
st.subheader("Real-time transparent deliberation of the AI Agents")

# --- DATA LOAD ---
weights, feedback = load_agent_stats()
thoughts = load_thought_stream()

# Tabs for Organization
thought_tab, arch_tab, memory_tab, feedback_tab = st.tabs([
    "🧠 Live Thought Stream",
    "🏛️ Swarm Architecture", 
    "💾 Memory & Skills",
    "🔄 Feedback Loop"
])

with thought_tab:
    st.markdown("### 📡 Central Working Memory (Real-time)")
    st.caption("This log captures the raw internal logic as agents process news, OI, and price action.")
    
    if not thoughts:
        st.info("Swarm is currently in idle state. Trigger the orchestrator to see deliberation.")
    else:
        # Build terminal UI
        thought_html = '<div class="thought-terminal">'
        for t in reversed(thoughts):
            time_str = t.timestamp.strftime("%H:%M:%S")
            action_str = f"[{t.action}]" if t.action else ""
            sym_str = f"({t.symbol})" if t.symbol else ""
            thought_html += f"""
            <div class="thought-line">
                <span class="timestamp">[{time_str}]</span> 
                <span class="agent-name">{t.agent_name}</span> 
                <span class="action">{action_str}</span> 
                {sym_str} <span class="msg">{t.message}</span>
            </div>
            """
        thought_html += '</div>'
        st.markdown(thought_html, unsafe_allow_html=True)

    if st.button("🔄 Refresh Brain"):
        st.rerun()

with arch_tab:
    # --- ARCHITECTURE FLOW ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("#### Agent Hierarchy & Institutional Intelligence")
        # Visualizing the Flow including new agents
        fig = go.Figure(go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = ["Macro Data", "Derivatives Forensics (GEX)", "Regime Scientist (Hurst)", "Market Strategist", "Nifty Analyst", "F&O Scanner", "Decision Hub"],
              color = ["#8b949e", "#ffb703", "#9b5de5", "#00b4d8", "#00b4d8", "#00b4d8", "#00d084"]
            ),
            link = dict(
              source = [0, 1, 2, 3, 4, 5], 
              target = [3, 6, 3, 6, 6, 6],
              value = [1, 2, 2, 2, 1, 1]
          )))
        fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.write("#### 💎 Institutional Intelligence significance")
        c_a, c_b = st.columns(2)
        with c_a:
            st.info("""
            **Gamma Exposure (GEX):**
            Identifies 'Market Maker Walls'. If Nifty is at 24,000 and GEX shows a massive wall, the Swarm will warn against buying Calls there, predicting a high-probability institutional rejection.
            """)
        with c_b:
            st.success("""
            **Hurst Exponent:**
            Detects 'Price Memory'. If Hurst < 0.45, the Swarm detects a ranging market and automatically ignores breakout signals, switching to mean-reversion logic to avoid premium decay.
            """)

    with col2:
        st.write("#### Active Swarm Components")
        st.markdown("""
        <div class="agent-card">
            <h3>1. Derivatives Forensics</h3>
            <p class="observation"><b>GEX Radar:</b> Tracking Gamma Walls & Zero-Gamma levels. Identifying institutional magnets.</p>
        </div>
        <div class="agent-card" style="border-left-color: #9b5de5;">
            <h3>2. Regime Scientist</h3>
            <p class="observation"><b>Regime Logic:</b> Hurst Exponent persistence analysis. Separating trending moves from random walk.</p>
        </div>
        <div class="agent-card" style="border-left-color: #f72585;">
            <h3>3. Market Strategist</h3>
            <p class="observation"><b>Market Breadth:</b> Advance-Decline verification. Ensuring Nifty moves have broad participation.</p>
        </div>
        <div class="agent-card" style="border-left-color: #00d084;">
            <h3>4. Decision Hub</h3>
            <p class="observation"><b>Confluence:</b> Synthesizing News, OI, GEX, and Technicals into a high-conviction verdict.</p>
        </div>
        """, unsafe_allow_html=True)

with memory_tab:
    st.write("### 💾 Long-Term Memory & Learned Skills")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### 🧠 Neural Weights (Synapses)")
        st.caption("These weights evolve daily based on profit/loss feedback.")
        if weights:
            fig_weights = go.Figure(go.Bar(
                x=list(weights.values()),
                y=list(weights.keys()),
                orientation='h',
                marker_color='#00d084'
            ))
            fig_weights.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_weights, use_container_width=True)
        else:
            st.info("Weights not yet initialized.")
            
    with c2:
        st.write("#### 📜 Learned Skills (Behavioral Rules)")
        st.caption("AI-generated rules stored in `data/skills/` to avoid repeating mistakes.")
        try:
            from trade_system.domains.advisory.application.agent.skill_registry import SkillRegistry
            registry = SkillRegistry()
            skills = registry.list_skills()
            if skills:
                for skill in skills:
                    st.success(f"✅ **{skill.replace('-', ' ').title()}**")
                    with st.expander("Show Rule Logic"):
                        st.code(registry.get_skill_text(skill))
            else:
                st.info("No skills learned yet. Post-market analysis generates skills.")
        except Exception:
            st.info("Skill Registry unavailable.")

with feedback_tab:
    st.write("#### 🔄 Self-Improvement Loop")
    st.markdown("""
    The swarm uses a **Reinforcement Learning** loop:
    1. **Execute**: Suggestions are logged.
    2. **Observe**: Market outcomes are updated at 4:00 PM.
    3. **Analyze**: `PostMarketImproverAgent` compares "Thoughts" vs "Results".
    4. **Adapt**: Updates `agent_weights.json` and writes new `.md` skills.
    """)
    
    if feedback:
        f_df = pd.DataFrame(feedback)
        st.write("#### Agent Performance History")
        st.dataframe(f_df, use_container_width=True)
    else:
        st.info("Genesis phase active. No historical feedback data yet.")

st.success("✨ **Swarm Matrix Online.** All agents transparent and communicating.")
