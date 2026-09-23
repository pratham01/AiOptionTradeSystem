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
        direction = t['direction'] if isinstance(t, dict) else t.direction
        symbol = t['symbol'] if isinstance(t, dict) else t.symbol
        confidence = t['confidence'] if isinstance(t, dict) else t.confidence
        narrative = t['narrative'] if isinstance(t, dict) else t.narrative
        horizon = t['horizon'] if isinstance(t, dict) else t.horizon
        entry_zone_low = t['entry_zone_low'] if isinstance(t, dict) else t.entry_zone_low
        entry_zone_high = t['entry_zone_high'] if isinstance(t, dict) else t.entry_zone_high
        target = t['target'] if isinstance(t, dict) else t.target
        stop_loss = t['stop_loss'] if isinstance(t, dict) else t.stop_loss
        sector = t['sector'] if isinstance(t, dict) else t.sector
        
        tag_cls = "call-tag" if direction == "CALL" else "put-tag"
        icon = "🟢" if direction == "CALL" else "🔴"
        
        sym_short = symbol.split(":")[-1].replace("-EQ", "").replace("-INDEX", "")
        
        # Identify Confluence Level
        conf_label = "STANDARD"
        if confidence >= 0.85: conf_label = "🔥 HIGH CONVICTION"
        elif confidence >= 0.70: conf_label = "💎 CONFLUENCE"
        
        # Check for RSI Divergence in narrative
        rsi_icon = "📈" if "RSI Divergence" in (narrative or "") else ""

        st.markdown(f"""
        <div class="trade-card {tag_cls}">
            <div class="confluence-badge">{conf_label}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size:1.2rem; font-weight:bold;">{icon} {sym_short} — BUY {direction} {rsi_icon}</span>
                <span class="metric-sub">{horizon}</span>
            </div>
            <div style="display:flex; gap:20px; margin: 10px 0;">
                <div><span class="metric-sub">Entry Zone</span><br/><span class="price-tag">₹{entry_zone_low:.0f}-{entry_zone_high:.0f}</span></div>
                <div><span class="metric-sub">Target</span><br/><span class="price-tag" style="color:#00d084">₹{target:.0f}</span></div>
                <div><span class="metric-sub">Stop Loss</span><br/><span class="price-tag" style="color:#ff4d6d">₹{stop_loss:.0f}</span></div>
            </div>
            <div style="font-size:0.9rem; opacity:0.8; margin-top:10px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
                <b>Swarm Reasoning:</b> {narrative}
                <br/>
                <span class="metric-sub">Confidence Score: {confidence:.0%} | {sector or 'Market'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_session_plan():
    """Try to load today's session plan from DB (returns serializable dicts)."""
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
                .all()
            )
            return [
                {
                    "symbol": t.symbol, "direction": t.direction, "horizon": t.horizon,
                    "entry_zone_low": t.entry_zone_low, "entry_zone_high": t.entry_zone_high,
                    "target": t.target, "stop_loss": t.stop_loss, "confidence": t.confidence,
                    "narrative": t.narrative, "sector": t.sector, "is_nifty": t.is_nifty,
                    "outcome": t.outcome, "actual_pnl_pct": t.actual_pnl_pct,
                }
                for t in trades
            ]
    except Exception:
        return []

@st.cache_data(ttl=15)
def load_thought_stream(limit=15):
    """Fetch live thoughts from the DB."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        query = text("SELECT timestamp, agent_name, symbol, action, message FROM agent_thought_stream ORDER BY timestamp DESC LIMIT :limit")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
            return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def load_latest_synthesis():
    """Fetch the latest AI Swarm Synthesis Report."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        query = text("""
            SELECT timestamp, message 
            FROM agent_thought_stream 
            WHERE agent_name = 'MarketSynthesizer' AND action = 'SYNTHESIS'
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = conn.execute(query).first()
            if row:
                ts_str = pd.to_datetime(row[0]).strftime('%d %b %Y %H:%M')
                return ts_str, row[1]
    except Exception:
        pass
    return None

@st.cache_data(ttl=120)
def load_closed_trades(limit_days=30):
    """Load recently closed trades for stats (returns serializable dicts)."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from trade_system.domains.market_data.infrastructure.database.models import SuggestedTrade
        from sqlalchemy.orm import Session
        engine = get_engine()
        cutoff = (date.today() - timedelta(days=limit_days)).isoformat()
        with Session(engine) as session:
            trades = (
                session.query(SuggestedTrade)
                .filter(SuggestedTrade.outcome != "PENDING")
                .filter(SuggestedTrade.date >= cutoff)
                .order_by(SuggestedTrade.date.desc())
                .all()
            )
            return [
                {
                    "symbol": t.symbol, "direction": t.direction,
                    "outcome": t.outcome, "actual_pnl_pct": t.actual_pnl_pct,
                    "date": t.date, "confidence": t.confidence,
                }
                for t in trades
            ]
    except Exception:
        return []

@st.cache_data(ttl=300)
def load_agent_weights():
    """Load active scoring weights."""
    try:
        from trade_system.domains.analysis.application.evolution.weight_evolver import WeightEvolver
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
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
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

@st.cache_data(ttl=30)
def load_agent_status():
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
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

@st.cache_data(ttl=15)
def load_radar_alerts(limit=10):
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
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
        
@st.cache_data(ttl=120)
def load_latest_news():
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
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

@st.cache_data(ttl=300)
def fetch_live_headlines():
    try:
        from trade_system.shared.utils.news_scraper import BusinessNewsScraper
        scraper = BusinessNewsScraper()
        return scraper.get_all_headlines()
    except Exception as e:
        return [f"Failed to load headlines: {e}"]

async def analyze_news_impact_async(headlines: list[str]) -> str:
    from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
    llm = LlmAdvisorClient()
    if not llm.configured():
        return "LLM not configured. Please check your GOOGLE_API_KEY."
    
    prompt = (
        "You are an expert Indian derivatives strategist and advisor specializing in options trading.\n"
        "Analyze the following macro news headlines to assess their direct impact on options buyers:\n\n"
        "Headlines:\n" + "\n".join(f"- {h}" for h in headlines[:25]) + "\n\n"
        "Provide a concise, premium Option Buyer's Macro Impact Report. Structure it with:\n"
        "1. **Macro Sentiment Verdict**: (BULLISH, BEARISH, NEUTRAL, or VOLATILE) and short rationale.\n"
        "2. **Sectoral Predictions**: Which sectors stand to benefit or suffer from momentum today.\n"
        "3. **Option Buyer Guidance**: Clear warnings (IV crush, theta decay risk, momentum setups to target).\n"
        "Keep it highly actionable, professional, and visually structured with emojis."
    )
    return await llm.complete(prompt)

@st.cache_data(ttl=60)
def load_detailed_suggested_trades(limit_days=30):
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from trade_system.domains.market_data.infrastructure.database.models import SuggestedTrade
        from sqlalchemy.orm import Session
        from datetime import date, timedelta
        engine = get_engine()
        cutoff = (date.today() - timedelta(days=limit_days)).isoformat()
        with Session(engine) as session:
            trades = (
                session.query(SuggestedTrade)
                .filter(SuggestedTrade.date >= cutoff)
                .order_by(SuggestedTrade.date.desc())
                .all()
            )
            
            rows = []
            for t in trades:
                f = t.features
                rows.append({
                    "id": t.id,
                    "date": t.date,
                    "symbol": t.symbol.split(":")[-1].replace("-EQ", "").replace("-INDEX", ""),
                    "direction": t.direction,
                    "entry_zone": f"₹{t.entry_zone_low:.0f}-{t.entry_zone_high:.0f}",
                    "target": t.target,
                    "stop_loss": t.stop_loss,
                    "confidence": t.confidence,
                    "outcome": t.outcome,
                    "actual_pnl_pct": t.actual_pnl_pct or 0.0,
                    "volume_surge": f.volume_surge if f else 1.0,
                    "near_support": f.near_support if f else 0,
                    "near_resistance": f.near_resistance if f else 0,
                    "above_vwap": f.above_vwap if f else 0,
                    "above_poc": f.above_poc if f else 0,
                })
            return rows
    except Exception as e:
        import logging
        logging.error(f"Error loading detailed suggested trades: {e}")
        return []

def compute_option_buyer_metrics(trade_row):
    vol_surge = trade_row.get("volume_surge") or 1.0
    near_support = trade_row.get("near_support") or 0
    near_resistance = trade_row.get("near_resistance") or 0
    direction = trade_row.get("direction")
    outcome = trade_row.get("outcome")
    
    # 1. Momentum Score (up to 40 points)
    mom_score = min(40.0, vol_surge * 15.0)
    
    # 2. Entry Zone Quality (up to 30 points)
    zone_score = 10.0
    if direction == "CALL" and near_support == 1:
        zone_score = 30.0
    elif direction == "PUT" and near_resistance == 1:
        zone_score = 30.0
        
    # 3. Efficiency Score (up to 30 points)
    eff_score = 15.0
    if outcome == "WIN":
        eff_score = 30.0
    elif outcome == "LOSS":
        eff_score = 5.0
        
    return round(mom_score + zone_score + eff_score, 1)

async def generate_option_buyer_eod_report(trades_data: list) -> str:
    from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient
    from trade_system.domains.market_data.infrastructure.database import log_agent_thought, get_db_session
    llm = LlmAdvisorClient()
    if not llm.configured():
        return "LLM not configured. Please check your GOOGLE_API_KEY."
        
    trade_summary = ""
    for t in trades_data[:15]:
        trade_summary += (
            f"- {t['date']} | {t['symbol']} | {t['direction']} | Outcome: {t['outcome']} (PnL: {t['actual_pnl_pct']:.1f}%) | "
            f"Vol Surge: {t['volume_surge']:.1f}x | Near Support: {t['near_support']} | Near Resistance: {t['near_resistance']}\n"
        )
        
    prompt = (
        "You are an expert Options Trading System Optimizer specializing in option buying strategy execution.\n"
        "Review the following record of suggested trades generated by our automated multi-agent swarm:\n\n"
        f"Trades Record:\n{trade_summary}\n"
        "Option buyers need explosive momentum and perfect entry timing to combat theta decay and transaction friction.\n\n"
        "Generate a comprehensive Post-Market Option Buyer System Optimization Report including:\n"
        "1. **Trade Entry Quality Audit**: Critique the entries. Did we buy near key structures? Was volume surge present?\n"
        "2. **Theta Decay Warning & Performance**: Assess holding durations and decay risk factors.\n"
        "3. **Tuning Recommendations**: Suggest exact indicator modifications (e.g. increase ST multipliers, raise volume surge thresholds, adjust evolution weights).\n"
        "Format your output in clean, premium Markdown with emojis."
    )
    
    report = await llm.complete(prompt)
    
    try:
        session = get_db_session()
        log_agent_thought(
            session,
            agent_name="OptionBuyerOptimizer",
            action="EOD_OPTIMIZATION",
            message=report
        )
        session.close()
    except Exception as e:
        import logging
        logging.error(f"Failed to log optimizer report: {e}")
        
    return report

@st.cache_data(ttl=300)
def load_historical_optimizations(limit=5):
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        query = text("""
            SELECT timestamp, message 
            FROM agent_thought_stream 
            WHERE agent_name = 'OptionBuyerOptimizer' AND action = 'EOD_OPTIMIZATION'
            ORDER BY timestamp DESC LIMIT :limit
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"limit": limit})
    except Exception:
        return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🤖 AI Trade Agent")
    st.markdown("---")
    lookback_days = st.slider("Performance Lookback (days)", 7, 90, 30)

    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("🔄 Run Orchestrator Now", type="primary", use_container_width=True):
        with st.spinner("Running agentic pipeline..."):
            try:
                from trade_system.domains.advisory.application.agent.factory import build_orchestrator
                from trade_system.shared.config.settings import Settings
                
                settings = Settings.load()
                settings.ensure_directories()
                orc = build_orchestrator(settings)
                import asyncio
                plan = asyncio.run(orc.run_session())
                st.success(f"✅ Generated {len(plan.all_suggestions())} suggestions!")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown("---")
    st.markdown("### 🧪 Swarm Strategy Lab")
    st.caption("Customize indicators and write natural language instructions for the AI Swarm.")
    
    custom_strategy = st.selectbox(
        "Active Strategy Family",
        options=["Breakout/Momentum", "SMC (Smart Money)", "ICT / Liquidity Sweep", "SMA Cross", "RVOL Trend"],
        key="custom_strat_family"
    )
    
    col_st1, col_st2 = st.columns(2)
    custom_st_period = col_st1.number_input("ST Period", 5, 20, 7, key="custom_st_period")
    custom_st_mult = col_st2.number_input("ST Multiplier", 1, 5, 3, key="custom_st_multiplier")
    
    col_rsi1, col_rsi2 = st.columns(2)
    custom_rsi_ob = col_rsi1.slider("RSI Overbought", 60, 85, 70, key="custom_rsi_ob")
    custom_rsi_os = col_rsi2.slider("RSI Oversold", 15, 40, 30, key="custom_rsi_os")
    
    custom_directive = st.text_area(
        "Swarm Prompt Directive",
        placeholder="e.g. Focus on high volume breakouts, CALL options only",
        value="Focus on high-conviction momentum setups, matching leading sectors.",
        key="custom_directive"
    )
    
    if st.button("⚡ Run Custom Swarm", type="primary", use_container_width=True):
        with st.spinner("Instantiating custom swarm..."):
            from trade_system.shared.config import Settings
            from dataclasses import replace
            from trade_system.domains.advisory.application.agent.factory import build_orchestrator
            import asyncio
            
            try:
                base_settings = Settings.load()
                new_ind = replace(
                    base_settings.indicator_config,
                    supertrend_period=int(custom_st_period),
                    supertrend_multiplier=int(custom_st_mult),
                    rsi_overbought=float(custom_rsi_ob),
                    rsi_oversold=float(custom_rsi_os)
                )
                custom_settings = replace(base_settings, indicator_config=new_ind)
                
                orc = build_orchestrator(custom_settings)
                plan = asyncio.run(orc.run_session(user_directive=custom_directive))
                st.session_state.custom_suggestions = plan.all_suggestions()
                st.success(f"Generated {len(st.session_state.custom_suggestions)} recommendations!")
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

# --- AUTO REFRESH ---
# Auto-refresh is handled globally by st_autorefresh in main.py to prevent full page reloads.


# --- TOP: MACRO & SECTORS ---
@st.fragment
def _render_macro_and_sectors():
    macro_c1, macro_c2 = st.columns([2, 1])
    with macro_c1:
        st.markdown("### 🌐 Macro & News Flow")
        news_msg = load_latest_news()
        st.info(f"**Latest Synthesis:** {news_msg}")
        
        # Scraped headlines
        headlines = fetch_live_headlines()
        with st.expander(f"📰 View Scraped Business Headlines ({len(headlines)})"):
            for h in headlines[:10]:
                st.markdown(f"- {h}")
                
        # News analysis trigger
        if st.button("🤖 Analyze Macro News Impact", key="analyze_news_btn", use_container_width=True):
            with st.spinner("Analyzing macro news impact for option buyers..."):
                import asyncio
                try:
                    verdict = asyncio.run(analyze_news_impact_async(headlines))
                    st.markdown(f"""
                    <div style='background:#1e2130; padding:15px; border-radius:10px; border:1px solid #00b4d8; margin-top:10px;'>
                        <h4 style='margin-top:0; color:#00b4d8;'>🧠 AI Macro Impact Analysis</h4>
                        <div style='font-size:0.9rem; line-height:1.4;'>{verdict.replace(chr(10), '<br/>')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to analyze news: {e}")

    with macro_c2:
        st.markdown("### 📊 Top Sectors")
        sector_df = load_sector_performance()
        if not sector_df.empty:
            st.dataframe(sector_df.head(4), use_container_width=True, hide_index=True)
        else:
            st.caption("Waiting for sector data...")

_render_macro_and_sectors()
st.markdown("---")

# --- MIDDLE: RADAR & AGENT STATUS ---
@st.fragment
def _render_radar_and_status():
    mid_c1, mid_c2 = st.columns([1.5, 1.5])
    with mid_c1:
        st.markdown("### 🎯 High-Conviction Radar")
        st.caption("Live breakouts, volume spikes, and anomalies")
        radar_df = load_radar_alerts(limit=6)
        if not radar_df.empty:
            for _, row in radar_df.iterrows():
                ts = pd.to_datetime(row['timestamp'], format="mixed").strftime('%H:%M:%S')
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

_render_radar_and_status()
st.markdown("---")

# --- QUICK STATS & SUGGESTIONS ---
today_trades = load_session_plan()
nifty_trades = [t for t in today_trades if t.get('is_nifty')]
fo_trades = [t for t in today_trades if not t.get('is_nifty')]

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
    # --- CUSTOM RECOMMENDATIONS PANEL ---
    if "custom_suggestions" in st.session_state and st.session_state.custom_suggestions:
        st.markdown("### ✨ Custom Swarm Recommendations")
        st.info("These suggestions were generated with your Strategy Lab overrides and custom directives.")
        _render_trade_cards(st.session_state.custom_suggestions)
        if st.button("🗑️ Clear Custom Recommendations"):
            st.session_state.custom_suggestions = []
            st.rerun()
        st.markdown("---")

    st.subheader("📡 Multi-Agent Parallel Verdicts")
    tab_nifty, tab_fo, tab_smc, tab_chat, tab_synthesis, tab_optimizer = st.tabs(["📊 NIFTY Master", "💎 Stock Momentum", "🏛️ Smart Money (SMC)", "💬 Swarm Chat", "🧠 AI Swarm Synthesis", "📉 Option Buyer EOD Optimizer"])

    with tab_nifty:
        _render_trade_cards(nifty_trades)

    with tab_fo:
        _render_trade_cards(fo_trades)

    with tab_smc:
        st.markdown("### 🏛️ Institutional Smart Money & Fair Value Gap (FVG) Studio")
        st.caption("Post-Market Institutional Fair Value Gaps (BISI/SIBI), Consequent Encroachment (50% CE), Order Blocks, and Wyckoff Schematics on Daily timeframe.")

        scan_mode = st.radio("Institutional Strategy Engine", ["⚡ Fair Value Gap (FVG & 50% CE)", "🏛️ Full Smart Money Concepts (SMC)", "📈 Wyckoff & VSA Accumulation"], horizontal=True)

        smc_col1, smc_col2, smc_col3 = st.columns([1, 1, 1.5])
        with smc_col1:
            smc_dir = st.selectbox("Direction Filter", ["both", "bullish", "bearish"], key="smc_dash_dir")
        with smc_col2:
            smc_score = st.slider("Min Confluence Score", 50, 95, 65, key="smc_dash_score")
        with smc_col3:
            smc_top_n = st.number_input("Max Setups to Show", 5, 50, 15, key="smc_dash_top_n")

        if st.button(f"🔍 Scan F&O Universe ({scan_mode.split()[1]})", key="run_inst_scan_dash_btn", type="primary", use_container_width=True):
            with st.spinner(f"Scanning 212 F&O universe stocks and indices with {scan_mode}..."):
                try:
                    if "Fair Value Gap" in scan_mode:
                        from trade_system.domains.advisory.application.agent.fvg_agent import FvgDailyScannerAgent
                        fvg_agent = FvgDailyScannerAgent(min_score=float(smc_score))
                        st.session_state.inst_setups = fvg_agent.scan_universe(direction=smc_dir, min_score=float(smc_score))
                        st.session_state.inst_scan_type = "FVG"
                    elif "Wyckoff" in scan_mode:
                        from trade_system.domains.advisory.application.agent.wyckoff_agent import WyckoffDailyScannerAgent
                        wyck_agent = WyckoffDailyScannerAgent(min_score=float(smc_score))
                        st.session_state.inst_setups = wyck_agent.scan_universe(direction=smc_dir, min_score=float(smc_score))
                        st.session_state.inst_scan_type = "WYCKOFF"
                    else:
                        from trade_system.domains.advisory.application.agent.smc_daily_agent import SmcDailyScannerAgent
                        smc_agent = SmcDailyScannerAgent(min_score=float(smc_score))
                        st.session_state.inst_setups = smc_agent.scan_universe(direction=smc_dir, min_score=float(smc_score))
                        st.session_state.inst_scan_type = "SMC"
                except Exception as e:
                    st.error(f"Scan failed: {e}")

        if "inst_setups" in st.session_state and st.session_state.inst_setups:
            setups = st.session_state.inst_setups
            scan_type = st.session_state.get("inst_scan_type", "SMC")
            st.success(f"🎯 Found {len(setups)} High-Conviction {scan_type} Setups on Daily Timeframe")

            # Table Overview
            table_rows = []
            for s in setups[:int(smc_top_n)]:
                zone_type = getattr(s, "zone_classification", "")
                row_dict = {
                    "Symbol": s.symbol.replace("NSE:", "").replace("-EQ", ""),
                    "Action": "🟢 BUY CALL" if s.direction == 1 else "🔴 BUY PUT",
                    "Score": s.confluence_score,
                    "Spot/Entry": s.entry_price,
                    "Protected SL": s.stop_loss,
                    "T1 (Range)": s.target_1,
                    "T2 (Target)": getattr(s, "target_2", s.target_1),
                    "RRR": f"1:{s.risk_reward_ratio:.1f}",
                    "Zone Type": zone_type if zone_type and zone_type != "NONE" else getattr(s, "setup_type", getattr(s, "pattern_type", "SETUP")),
                }
                if hasattr(s, "consequent_encroachment"):
                    row_dict["50% CE"] = s.consequent_encroachment
                table_rows.append(row_dict)

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                    "Spot/Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "Protected SL": st.column_config.NumberColumn("Protected SL", format="₹%.2f"),
                    "T1 (Range)": st.column_config.NumberColumn("T1 (Range)", format="₹%.2f"),
                    "T2 (Target)": st.column_config.NumberColumn("T2 (Target)", format="₹%.2f"),
                    "50% CE": st.column_config.NumberColumn("50% CE", format="₹%.2f"),
                }
            )

            st.markdown("---")
            st.markdown("#### 🔍 Setup Deep-Dives (Photon Structure & JeaFx S&D)")
            for idx, s in enumerate(setups[:int(smc_top_n)], 1):
                sym_clean = s.symbol.replace("NSE:", "").replace("-EQ", "")
                setup_title = getattr(s, "setup_type", getattr(s, "pattern_type", "SETUP"))
                z_name = getattr(s, "zone_classification", "")
                z_badge = f" • [{z_name}]" if z_name and z_name != "NONE" else ""
                realigned_badge = " ⚡ [REALIGNED]" if getattr(s, "is_internal_realigned", False) else ""
                badge = f"🟢 BUY CALL [{setup_title}]{z_badge}{realigned_badge}" if s.direction == 1 else f"🔴 BUY PUT [{setup_title}]{z_badge}{realigned_badge}"
                
                with st.expander(f"#{idx} | {sym_clean} — {badge} (Score: {s.confluence_score:.0f}/100)"):
                    c_a, c_b, c_c, c_d, c_e = st.columns(5)
                    c_a.metric("Entry Zone", f"₹{s.entry_price:.2f}")
                    c_b.metric("Protected SL", f"₹{s.stop_loss:.2f}", delta=f"-{abs(s.entry_price - s.stop_loss):.2f}", delta_color="inverse")
                    c_c.metric("T1 (Range-to-Range)", f"₹{s.target_1:.2f}", delta=f"+{abs(s.target_1 - s.entry_price):.2f}")
                    t2_val = getattr(s, "target_2", s.target_1)
                    c_d.metric("T2 (Weak Target)", f"₹{t2_val:.2f}", delta=f"+{abs(t2_val - s.entry_price):.2f}")
                    c_e.metric("Risk:Reward", f"1:{s.risk_reward_ratio:.1f}")

                    if hasattr(s, "market_structure") and s.market_structure:
                        st.markdown(f"🏛️ **Market Structure Context:** `{s.market_structure}` | **Dealing Range:** `{getattr(s, 'equilibrium_status', 'EQUILIBRIUM')}`")

                    if hasattr(s, "fvg_bottom") and hasattr(s, "fvg_top"):
                        st.markdown(f"**Imbalance Zone:** ₹{s.fvg_bottom:.2f} ➔ ₹{s.fvg_top:.2f} | **50% CE Midpoint:** ₹{s.consequent_encroachment:.2f}")

                    st.markdown("**Institutional Confluence Factors:**")
                    for r in getattr(s, "reasons", []):
                        st.markdown(f"- {r}")

        else:
            st.info("Select an institutional engine above and click scan to view active setups on the daily chart.")

    with tab_synthesis:
        synthesis_data = load_latest_synthesis()
        if synthesis_data:
            ts_str, report_content = synthesis_data
            st.caption(f"📅 Generated at: {ts_str} (IST)")
            st.markdown(report_content)
        else:
            st.info("No AI Swarm Synthesis reports generated yet. Run the EOD post-market scan to generate the report.")

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
                    from trade_system.domains.advisory.application.agent.data_query_agent import DataQueryAgent
                    try:
                        agent = DataQueryAgent()
                        response = asyncio.run(agent.process_query(prompt))
                        st.markdown(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error querying agent: {e}")

    with tab_optimizer:
        st.markdown("### 📉 Option Buyer EOD Performance Analyzer & Optimizer")
        st.caption("Auditing trade entry conditions, theta decay indicators, and volume node breakouts to optimize option buyer edge.")
        
        detailed_trades = load_detailed_suggested_trades(lookback_days)
        
        if not detailed_trades:
            st.info("No suggested trades found in the lookback period to analyze.")
        else:
            trades_df_opt = pd.DataFrame(detailed_trades)
            trades_df_opt["Buyer Edge Score"] = trades_df_opt.apply(compute_option_buyer_metrics, axis=1)
            
            avg_score = trades_df_opt["Buyer Edge Score"].mean()
            win_rate_opt = (trades_df_opt["outcome"] == "WIN").sum() / max(1, len(trades_df_opt)) * 100
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Average Buyer Edge Score", f"{avg_score:.1f}/100", help="Based on volume surge, support proximity, and hold efficiency.")
            mc2.metric("Win Rate", f"{win_rate_opt:.1f}%")
            mc3.metric("Total Trades Audited", f"{len(trades_df_opt)}")
            
            st.dataframe(
                trades_df_opt[["date", "symbol", "direction", "entry_zone", "outcome", "actual_pnl_pct", "volume_surge", "Buyer Edge Score"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "actual_pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f%%"),
                    "volume_surge": st.column_config.NumberColumn("Vol Surge", format="%.2fx"),
                    "Buyer Edge Score": st.column_config.ProgressColumn("Buyer Edge Score", min_value=0, max_value=100)
                }
            )
            
            st.markdown("---")
            if st.button("🧠 Run AI EOD Optimization Audit", key="run_eod_opt_audit_btn", use_container_width=True):
                with st.spinner("Analyzing trade metrics and generating system tuning advice..."):
                    import asyncio
                    try:
                        opt_report = asyncio.run(generate_option_buyer_eod_report(detailed_trades))
                        st.markdown(f"""
                        <div style='background:#1e2130; padding:20px; border-radius:12px; border:1px solid #ff4d6d; margin-top:15px; margin-bottom:15px;'>
                            <h3 style='margin-top:0; color:#ff4d6d;'>🧠 AI System Optimization Report</h3>
                            <div style='font-size:0.9rem; line-height:1.5;'>{opt_report.replace(chr(10), '<br/>')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Failed to generate optimization report: {e}")
                        
        st.markdown("---")
        st.subheader("📚 Historical EOD Optimization Reports")
        hist_opt = load_historical_optimizations()
        if not hist_opt.empty:
            for _, r in hist_opt.iterrows():
                ts_formatted = pd.to_datetime(r['timestamp'], format="mixed").strftime('%d %b %Y %H:%M')
                with st.expander(f"Report — {ts_formatted}"):
                    st.markdown(r['message'])
        else:
            st.caption("No historical optimization reports found.")

with right_col:
    @st.fragment
    def _render_thought_stream():
        st.subheader("🧠 Swarm Thought Stream")

        # Live Thought Stream with Color Coding
        thoughts = load_thought_stream()
        if not thoughts.empty:
            for _, row in thoughts.iterrows():
                ts = pd.to_datetime(row['timestamp'], format="mixed").strftime('%H:%M:%S')
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
    
    _render_thought_stream()

st.markdown("---")
st.caption("🤖 AI Trading Agent | Agentic Options Trading System | Advisory Only — Not Financial Advice")
