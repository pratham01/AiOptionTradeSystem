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
        st.Page(HERE / "agent_dashboard.py", title="Agent Dashboard", icon="🤖", default=True),
        st.Page(HERE / "market_mood_dashboard.py", title="Market Mood", icon="📊"),
        st.Page(HERE / "smart_oi_dashboard.py", title="Smart OI", icon="🎯"),
        st.Page(HERE / "sector_scope_dashboard.py", title="Sector Scope", icon="🧭"),
        st.Page(HERE / "option_edge_dashboard.py", title="Option Edge", icon="⚡"),
    ],
    "Research & Strategy": [
        st.Page(HERE / "nifty_volatility_dashboard.py", title="Nifty Volatility Lab", icon="🧭"),
        st.Page(HERE / "option_research_dashboard.py", title="Option Research Lab", icon="🔬"),
        st.Page(HERE / "candlestick_pattern_dashboard.py", title="Candlestick Pattern Lab", icon="🕯️"),
        st.Page(HERE / "strategy_dashboard.py", title="Strategy Lab", icon="🧪"),
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

# Global Auto-Refresh
refresh_rate = st.sidebar.select_slider(
    "🔄 Auto-Refresh (seconds)",
    options=[0, 10, 30, 60, 300],
    value=60,
    help="Set to 0 to disable implicit updates."
)

if refresh_rate > 0:
    st.sidebar.caption(f"⏱️ Implicit updates every {refresh_rate}s")

# Run the selected page
pg.run()

# Implicit Update Logic
# We use st_autorefresh to trigger a clean server-side script rerun at the selected refresh rate,
# preventing the browser window from performing a full reload and resetting DOM/connections.
if refresh_rate > 0:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh_rate * 1000, key="global_dashboard_refresh")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.info("System Status: **Operational**")
