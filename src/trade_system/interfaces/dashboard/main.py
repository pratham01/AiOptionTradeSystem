import streamlit as st
from pathlib import Path

# Set a unified page config for the entire app
# Note: Individual pages might still try to set their own, 
# but st.navigation handles the main app shell.
st.set_page_config(
    page_title="AI Trade System - Mission Control",
    page_icon="🚀",
    layout="wide",
)

# Define the pages
HERE = Path(__file__).parent
pages = {
    "Live Operations": [
        st.Page(HERE / "chief_agent_dashboard.py", title="Chief Trading Agent", icon="🎯", default=True),
        st.Page(HERE / "agent_dashboard.py", title="Agent Dashboard", icon="🤖"),
        st.Page(HERE / "smart_oi_dashboard.py", title="Smart OI", icon="🎯"),
        st.Page(HERE / "sector_scope_dashboard.py", title="Sector Scope", icon="🧭"),
        st.Page(HERE / "broker_health_dashboard.py", title="Broker Health", icon="🔌"),
        st.Page(HERE / "market_mood_dashboard.py", title="Market Mood", icon="📊"),
        st.Page(HERE / "option_edge_dashboard.py", title="Option Edge", icon="⚡"),
        st.Page(HERE / "btst_scanner_dashboard.py", title="BTST Scanner", icon="🌅"),
    ],
    "Research & Strategy": [
        st.Page(HERE / "greeks_exposure_dashboard.py", title="GEX & DEX Engine", icon="🧲"),
        st.Page(HERE / "nifty_volatility_dashboard.py", title="Nifty Volatility Lab", icon="🧭"),
        st.Page(HERE / "option_research_dashboard.py", title="Option Research Lab", icon="🔬"),
        st.Page(HERE / "trade_book_dashboard.py", title="Detailed Trade Book", icon="📖"),
        st.Page(HERE / "bollinger_breakout_dashboard.py", title="Bollinger Breakout Lab", icon="⚡"),
        st.Page(HERE / "candlestick_pattern_dashboard.py", title="Candlestick Pattern Lab", icon="🕯️"),
        st.Page(HERE / "strategy_dashboard.py", title="Strategy Lab", icon="🧪"),
        st.Page(HERE / "backtest_studio_dashboard.py", title="Backtest Studio", icon="🔬"),
        st.Page(HERE / "volumetric_order_flow_dashboard.py", title="Volumetric Order Flow", icon="📊"),
        st.Page(HERE / "sr_channels_dashboard.py", title="S/R Channels", icon="🏗️"),
        st.Page(HERE / "gamma_blast_dashboard.py", title="Gamma Blast Lab", icon="⚡"),
        st.Page(HERE / "agent_arch_viz.py", title="Swarm Architecture", icon="🏛️"),
    ],
}

# Initialize Navigation
pg = st.navigation(pages)

# Custom Sidebar Header
st.sidebar.title("🚀 Trade System V2")
st.sidebar.caption("Autonomous Multi-Agent Swarm")
st.sidebar.markdown("---")

# ── ST Flip Live Bot & Data Sanity Supervisor ───────────────────────────
def _get_live_bot_health():
    import urllib.request, json
    try:
        req = urllib.request.urlopen("http://localhost:9090/health", timeout=1.5)
        return json.loads(req.read().decode())
    except Exception:
        return None

def _start_live_bot():
    import subprocess, sys
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    script_path = root_dir / "scripts" / "run_live_trading.py"
    subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(root_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

# Auto-start on application initial load
if "live_bot_auto_checked" not in st.session_state:
    st.session_state["live_bot_auto_checked"] = True
    bot_health = _get_live_bot_health()
    if not bot_health:
        _start_live_bot()

bot_health = _get_live_bot_health()
if bot_health and bot_health.get("status") == "healthy":
    st.sidebar.success("🟢 ST Flip Live Bot: **Running**")
    ws_ok = bot_health.get("components", {}).get("websocket", {}).get("connected", False)
    db_ok = bot_health.get("components", {}).get("database", {}).get("status", "") == "healthy"
    st.sidebar.caption(f"⚡ WS: {'Connected' if ws_ok else 'Reconnecting'} | 🛡️ DB: {'Sanity OK' if db_ok else 'Checking'}")
    st.sidebar.caption("📢 Alerts: **STFlip Channel**")
else:
    st.sidebar.warning("🔴 ST Flip Live Bot: **Offline**")
    st.sidebar.caption("ST Flip alerts & tick stream inactive")
    if st.sidebar.button("🚀 Start ST Flip Bot", key="start_bot_btn", use_container_width=True):
        _start_live_bot()
        st.rerun()

# Global Auto-Refresh
refresh_rate = st.sidebar.select_slider(
    "🔄 Auto-Refresh (seconds)",
    options=[0, 5, 10, 30, 60, 300],
    value=10,
    help="Set to 0 to disable implicit updates."
)

if refresh_rate > 0:
    st.sidebar.caption(f"⏱️ Implicit updates every {refresh_rate}s")

# Run the selected page
pg.run()

# Implicit Update Logic
if refresh_rate > 0:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh_rate * 1000, key="global_dashboard_refresh")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.info("System Status: **Operational**")

