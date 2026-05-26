import json
from datetime import date, timedelta, datetime
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
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #00b4d8;
    }
    .status-active { color: #00d084; font-weight: bold; font-size: 0.9rem; }
    
    /* Agent Consensus Gauge */
    .consensus-box {
        background: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .consensus-val { font-size: 1.8rem; font-weight: bold; color: #00b4d8; }
    
    /* Agent Thought Cards */
    .thought-card {
        background: #161b22;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 4px solid #30363d;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .agent-name { color: #58a6ff; font-weight: bold; margin-bottom: 2px; }
    .thought-msg { color: #c9d1d9; }
    .thought-time { color: #8b949e; font-size: 0.75rem; float: right; }
    
    /* Trade Cards */
    .trade-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
        position: relative;
    }
    .trade-card:hover { transform: translateY(-2px); border-color: rgba(0, 180, 216, 0.3); }
    .call-tag { border-left: 5px solid #00d084; }
    .put-tag { border-left: 5px solid #ff4d6d; }
    .confluence-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(0, 180, 216, 0.2);
        color: #00b4d8;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: bold;
    }
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
        
        # Identify Confluence Level
        conf_label = "STANDARD"
        if t.confidence >= 0.85: conf_label = "🔥 HIGH CONVICTION"
        elif t.confidence >= 0.70: conf_label = "💎 CONFLUENCE"
        
        # Check for RSI Divergence in narrative
        rsi_icon = "📈" if "RSI Divergence" in (t.narrative or "") else ""

        st.markdown(f"""
        <div class="trade-card {tag_cls}">
            <div class="confluence-badge">{conf_label}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size:1.2rem; font-weight:bold;">{icon} {sym_short} — BUY {t.direction} {rsi_icon}</span>
                <span class="metric-sub">{t.horizon}</span>
            </div>
            <div style="display:flex; gap:20px; margin: 10px 0;">
                <div><span class="metric-sub">Entry Zone</span><br/><span class="price-tag">₹{t.entry_zone_low:.0f}-{t.entry_zone_high:.0f}</span></div>
                <div><span class="metric-sub">Target</span><br/><span class="price-tag" style="color:#00d084">₹{t.target:.0f}</span></div>
                <div><span class="metric-sub">Stop Loss</span><br/><span class="price-tag" style="color:#ff4d6d">₹{t.stop_loss:.0f}</span></div>
            </div>
            <div style="font-size:0.9rem; opacity:0.8; margin-top:10px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
                <b>Swarm Reasoning:</b> {t.narrative}
                <br/>
                <span class="metric-sub">Confidence Score: {t.confidence:.0%} | {t.sector or 'Market'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def load_session_plan():
    """Try to load today's session plan from DB."""
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from trade_system.infrastructure.database.models import SuggestedTrade
        from sqlalchemy.orm import Session
        engine = get_engine()
        today = date.today().isoformat()
        with Session(engine) as session:
            trades = (
                session.query(SuggestedTrade)
                .filter(SuggestedTrade.date == today)
                .all()
            )
            return trades
    except Exception:
        return []

def load_thought_stream(limit=15):
    """Fetch live thoughts from the DB."""
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        query = text("SELECT timestamp, agent_name, symbol, action, message FROM agent_thought_stream ORDER BY timestamp DESC LIMIT :limit")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
            return df
    except Exception:
        return pd.DataFrame()

def load_closed_trades(limit_days=30):
    """Load recently closed trades for stats."""
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from trade_system.infrastructure.database.models import SuggestedTrade
        from sqlalchemy.orm import Session
        engine = get_engine()
        cutoff = (date.today() - timedelta(days=limit_days)).isoformat()
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
    """Load active scoring weights."""
    try:
        from trade_system.application.evolution.weight_evolver import WeightEvolver
        evolver = WeightEvolver()
        return {
            "candidate_screener": evolver.load_weights("candidate_screener"),
            "setup_validator": evolver.load_weights("setup_validator")
        }
    except Exception:
        return {}

@st.cache_data(ttl=601)
def load_sector_performance():
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        import pandas as pd
        import json
        from pathlib import Path
        
        engine = get_engine()
        config_path = Path("config/fo_universe.json")
        with open(config_path, "r") as f:
            fo_metadata = json.load(f)
            
        query = text("""
            SELECT symbol, close 
            FROM ohlcv_15m 
            WHERE timestamp >= date('now', '-3 days')
            ORDER BY timestamp DESC
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            return pd.DataFrame()
            
        latest_closes = df.groupby('symbol').first().reset_index()
        sectors = []
        for sym in latest_closes['symbol']:
            sectors.append(fo_metadata.get(sym, "UNKNOWN"))
        latest_closes['sector'] = sectors
        
        grouped = df.groupby('symbol')
        changes = []
        for sym, group in grouped:
            if len(group) > 5:
                last = group.iloc[0]['close']
                prev = group.iloc[-1]['close']
                pct = ((last - prev) / prev) * 100
            else:
                pct = 0
            changes.append({'symbol': sym, 'pChange': pct})
            
        changes_df = pd.DataFrame(changes)
        merged = pd.merge(latest_closes, changes_df, on='symbol', suffixes=('_dummy', ''))
        
        sector_df = merged.groupby('sector').agg({'pChange': 'mean', 'close': 'mean'}).reset_index()
        sector_df = sector_df.rename(columns={'close': 'lastPrice'})
        sector_df = sector_df[sector_df['sector'] != 'UNKNOWN']
        return sector_df.sort_values('pChange', ascending=False)
    except Exception as e:
        import logging
        logging.error(f"Sector load error: {e}")
        pass
    return pd.DataFrame()

def load_agent_status():
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        import pandas as pd
        engine = get_engine()
        query = text("""
            SELECT agent_name, MAX(timestamp) as last_seen, action, message 
            FROM agent_thought_stream 
            WHERE timestamp >= date('now', '-1 day')
            GROUP BY agent_name 
            ORDER BY last_seen DESC
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame()

def load_radar_alerts(limit=10):
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        import pandas as pd
        engine = get_engine()
        query = text("""
            SELECT timestamp, agent_name, symbol, action, message 
            FROM agent_thought_stream 
            WHERE (action IN ('VOL', 'SUDDEN_OI', 'MWPL_SQUEEZE', 'GAMMA_BLAST', 'SMC_Alert') 
               OR action LIKE 'RETEST_%' 
               OR message LIKE '%breakout%' 
               OR message LIKE '%surge%')
               AND timestamp >= datetime('now', '-4 hours', 'localtime')
            ORDER BY timestamp DESC LIMIT :limit
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"limit": limit})
    except Exception:
        return pd.DataFrame()
        
def load_latest_news():
    try:
        from trade_system.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        import pandas as pd
        engine = get_engine()
        query = text("""
            SELECT timestamp, message 
            FROM agent_thought_stream 
            WHERE agent_name = 'PreMarketNewsAgent' 
            ORDER BY timestamp DESC LIMIT 1
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            if not df.empty:
                return df.iloc[0]['message']
    except Exception:
        pass
    return "No recent macro news synthesis available."

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🤖 AI Trade Agent")
    st.markdown("---")
    lookback_days = st.slider("Performance Lookback (days)", 7, 90, 30)

    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("🔄 Run Orchestrator Now", type="primary"):
        with st.spinner("Running agentic pipeline..."):
            try:
                from trade_system.application.agent.factory import build_orchestrator
                from trade_system.config.settings import Settings
                
                settings = Settings.load()
                settings.ensure_directories()
                orc = build_orchestrator(settings)
                # Run sync wrapper
                import asyncio
                plan = asyncio.run(orc.run_session())
                st.success(f"✅ Generated {len(plan.all_suggestions())} suggestions!")
            except Exception as e:
                st.error(f"Failed: {e}")

# --- MAIN UI ---
st.markdown(f"""
<div class="hero-banner">
    <h1 style='margin:0; color:white;'>🚀 Swarm Mission Control</h1>
    <p style='margin:0; opacity:0.8;'>Autonomous Multi-Agent Parallel Intelligence — Session: <b>{date.today().strftime('%d %b %Y')}</b></p>
    <div style='margin-top:10px;'>
        <span class='status-active'>● Swarm Operational</span> &nbsp;&nbsp;
        <span style='opacity:0.6'>|</span> &nbsp;&nbsp;
        <span style='color:#00b4d8'>Active Agents: 7</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- AUTO REFRESH SCRIPT ---
import streamlit_javascript as st_js
st_js.st_javascript("setTimeout(function(){ window.parent.location.reload(); }, 60000);")

# --- TOP: MACRO & SECTORS ---
macro_c1, macro_c2 = st.columns([2, 1])
with macro_c1:
    st.markdown("### 🌐 Macro & News Flow")
    news_msg = load_latest_news()
    st.info(f"**Latest Synthesis:** {news_msg}")

with macro_c2:
    st.markdown("### 📊 Top Sectors")
    sector_df = load_sector_performance()
    if not sector_df.empty:
        st.dataframe(sector_df.head(4), use_container_width=True, hide_index=True)
    else:
        st.caption("Waiting for sector data...")

st.markdown("---")

# --- MIDDLE: RADAR & AGENT STATUS ---
mid_c1, mid_c2 = st.columns([1.5, 1.5])
with mid_c1:
    st.markdown("### 🎯 High-Conviction Radar")
    st.caption("Live breakouts, volume spikes, and anomalies")
    radar_df = load_radar_alerts(limit=6)
    if not radar_df.empty:
        for _, row in radar_df.iterrows():
            ts = pd.to_datetime(row['timestamp']).strftime('%H:%M:%S')
            action = row['action']
            sym = row['symbol'].split(':')[-1].replace('-EQ', '').replace('-INDEX', '') if row['symbol'] else ''
            icon = "⚡" if "VOL" in action else "🚀" if "BREAKOUT" in action else "🔥" if "GAMMA" in action else "🎯"
            
            st.markdown(f"""
            <div style='background:#2b3044; padding:10px; border-radius:8px; margin-bottom:8px; border-left:4px solid #ffb703'>
                <span style='color:#8b949e; font-size:0.8rem'>{ts}</span> &nbsp;
                <b style='color:#00b4d8'>{icon} {sym}</b> &nbsp;
                <span style='font-size:0.9rem'>{row['message']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No anomalies detected recently.")

with mid_c2:
    st.markdown("### 🤖 Agent Grid Status")
    status_df = load_agent_status()
    if not status_df.empty:
        for _, row in status_df.iterrows():
            agent = row['agent_name']
            msg = row['message'][:60] + "..." if len(row['message']) > 60 else row['message']
            st.markdown(f"""
            <div style='background:#161b22; padding:8px; border-radius:6px; margin-bottom:6px; border:1px solid #30363d'>
                <div style='color:#58a6ff; font-weight:bold; font-size:0.85rem'>@{agent} <span style='float:right; color:#00d084'>● Active</span></div>
                <div style='color:#c9d1d9; font-size:0.8rem'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# --- QUICK STATS & SUGGESTIONS ---
today_trades = load_session_plan()
nifty_trades = [t for t in today_trades if t.is_nifty]
fo_trades = [t for t in today_trades if not t.is_nifty]

# --- SWARM CONSENSUS ---
with st.container():
    c1, c2, c3, c4, c5 = st.columns(4 + 1)

    # Dynamic Bias Calculation from Agent Deliberations
    thoughts_bias = load_thought_stream(limit=10)
    bias_score = 0.5 # Neutral base
    if not thoughts_bias.empty:
        bullish_c = thoughts_bias['message'].str.contains('Bullish|BULL|Positive', case=False).sum()
        bearish_c = thoughts_bias['message'].str.contains('Bearish|BEAR|Negative', case=False).sum()
        total = bullish_c + bearish_c
        if total > 0:
            bias_score = bullish_c / total

    bias_text = "NEUTRAL"
    if bias_score > 0.6: bias_text = "BULLISH"
    elif bias_score < 0.4: bias_text = "BEARISH"

    with c1:
        st.markdown(f"""<div class="consensus-box"><span class="metric-sub">Swarm Consensus</span><br/><span class="consensus-val" style="color:{'#00d084' if bias_text=='BULLISH' else '#ff4d6d' if bias_text=='BEARISH' else '#00b4d8'}">{bias_text}</span></div>""", unsafe_allow_html=True)
    with c2:
        st.metric("Agents Deliberating", "7", help="News, OI, Technical, Correlation, Prediction, Retracement, Skill")
    with c3:
        st.metric("Swarm Conviction", f"{bias_score:.0%}")
    with c4:
        st.metric("Confluence Depth", "HIGH" if (bullish_c + bearish_c) > 5 else "MEDIUM")
    with c5:
        st.metric("Self-Healing", "ACTIVE")

# --- LAYOUT: Left (Suggestions) | Right (Meta-Analysis) ---
left_col, right_col = st.columns([1.8, 1.2])

with left_col:
    st.subheader("📡 Multi-Agent Parallel Verdicts")
    tab_nifty, tab_fo, tab_chat = st.tabs(["📊 NIFTY Master", "💎 Stock Momentum", "💬 Swarm Chat"])

    with tab_nifty:
        _render_trade_cards(nifty_trades)

    with tab_fo:
        _render_trade_cards(fo_trades)

    with tab_chat:
        st.markdown("### 💬 Ask Swarm Data Agent")
        
        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display chat messages from history on rerun
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask about market data (e.g., 'What is the average volume in Reliance for the last 1 month?'):"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Agent is analyzing database and synthesizing answer..."):
                    import asyncio
                    from trade_system.application.agent.data_query_agent import DataQueryAgent
                    try:
                        agent = DataQueryAgent()
                        response = asyncio.run(agent.process_query(prompt))
                        st.markdown(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error querying agent: {e}")

with right_col:
    st.subheader("🧠 Swarm Thought Stream")

    # Live Thought Stream with Color Coding
    thoughts = load_thought_stream()
    if not thoughts.empty:
        for _, row in thoughts.iterrows():
            ts = pd.to_datetime(row['timestamp']).strftime('%H:%M:%S')
            agent = row['agent_name']
            msg = row['message']

            # Highlight key keywords
            msg = msg.replace("Bullish", "<b><span style='color:#00d084'>Bullish</span></b>")
            msg = msg.replace("Bearish", "<b><span style='color:#ff4d6d'>Bearish</span></b>")

            st.markdown(f"""
            <div class="thought-card" style="border-left-color: {'#00b4d8' if 'News' in agent else '#ffb703' if 'OI' in agent else '#9b5de5'}">
                <span class="thought-time">{ts}</span>
                <div class="agent-name">@{agent} {f'[{row["symbol"]}]' if row['symbol'] else ''}</div>
                <div class="thought-msg">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Waiting for agent deliberations...")


    st.divider()
    st.markdown("### 🧬 Swarm Architecture Context")
    st.info("""
    **Significance of the Swarm:**
    - **NewsAgent**: Processes macro sentiment (avoiding 'bad news' traps).
    - **OptionChainAgent**: Tracks the 'Large Institutional Footprint' via OI.
    - **ScreenerAgent**: Ensures technical structure (VWAP/ST) is confirmed.
    *A trade is only suggested when these independent minds reach Confluence.*
    """)

    st.divider()
    st.markdown("### ⚙️ Evolved Weights")
    weights = load_agent_weights()
    if weights:
        for agent_name, agent_weights in weights.items():
            with st.expander(f"Weights: {agent_name.replace('_', ' ').title()}"):
                for k, v in sorted(agent_weights.items(), key=lambda x: x[1], reverse=True):
                    st.progress(v, text=f"{k}: {v:.2f}")

st.markdown("---")
st.caption("🤖 AI Trading Agent | Agentic Options Trading System | Advisory Only — Not Financial Advice")
