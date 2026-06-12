import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from trade_system.interfaces.dashboard.data import (
    load_strategy_summaries,
    load_strategy_trades,
    load_agent_lab_runs,
    load_agent_lab_detail
)

# Page Config (handled by main.py)
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# --- THEME & STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; }
    .stMetric { background: #1e2130; padding: 15px; border-radius: 12px; border: 1px solid #30363d; }
    .best-badge { background: #00d084; color: #0d1117; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .card { background: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px; }
    h1, h2, h3 { color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADING ---
@st.cache_data(ttl=600)
def cached_load_strategy_summaries(root_path):
    return load_strategy_summaries(root_path)

@st.cache_data(ttl=600)
def cached_load_strategy_trades(root_path, strategy):
    return load_strategy_trades(root_path, strategy)

@st.cache_data(ttl=600)
def cached_load_agent_lab_runs(root_path):
    return load_agent_lab_runs(root_path)

@st.cache_data(ttl=600)
def cached_load_agent_lab_detail(root_path, run_name):
    return load_agent_lab_detail(root_path, run_name)


def main():
    st.title("📈 Strategy Analytics Studio")
    st.caption("Quantum performance analysis and AI-driven strategy optimizations.")

    # 1. Load Data (Cached)
    summaries = cached_load_strategy_summaries(ROOT)
    if summaries.empty:
        st.warning("No backtest results found. Run a backtest ritual first.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🎯 Strategy Filters")
    
    # Year Filter
    available_years = sorted([int(y) for y in summaries['year'].dropna().unique()])
    year_options = ["All Years"] + [str(y) for y in available_years]
    selected_year = st.sidebar.selectbox("Filter by Year", year_options)
    
    # Family Filter
    available_families = sorted(summaries['family'].unique())
    family_options = ["All Families"] + list(available_families)
    selected_family = st.sidebar.selectbox("Filter by Strategy Family", family_options)
    
    # Apply Filters
    filtered_summaries = summaries.copy()
    if selected_year != "All Years":
        filtered_summaries = filtered_summaries[filtered_summaries['year'] == int(selected_year)]
    if selected_family != "All Families":
        filtered_summaries = filtered_summaries[filtered_summaries['family'] == selected_family]
        
    if filtered_summaries.empty:
        st.warning("No strategy matches the selected filters. Please select different options.")
        return

    # 2. Hero Overview (Tournament Leaderboard)
    st.markdown("### 🏆 Strategy Leaderboard")
    
    # Calculate global ranking
    filtered_summaries['Score'] = (filtered_summaries['win_rate'] / 100 * 0.4) + (filtered_summaries['profit_factor'] * 0.6)
    max_score = filtered_summaries['Score'].max()
    if pd.isna(max_score) or max_score <= 0:
        max_score = 1.0
        
    ranked = filtered_summaries.sort_values("Score", ascending=False).reset_index(drop=True)
    
    top_strat = ranked.iloc[0]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Performer", top_strat['strategy'], delta=f"PF: {top_strat['profit_factor']:.2f}")
    c2.metric("Best Net PnL", f"{filtered_summaries['net_points'].max():.2f} pts")
    c3.metric("Peak Win Rate", f"{filtered_summaries['win_rate'].max():.1f}%")
    c4.metric("Active Strategies", len(filtered_summaries['strategy'].unique()))

    # Tournament Table
    st.dataframe(
        ranked[['strategy', 'family', 'year', 'trades', 'win_rate', 'net_points', 'profit_factor', 'Score']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
            "net_points": st.column_config.NumberColumn("Net PnL", format="%.2f"),
            "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
            "Score": st.column_config.ProgressColumn("Rank Score", min_value=0, max_value=float(max_score))
        }
    )

    # 3. Visual Comparison Matrix
    st.markdown("---")
    st.subheader("📊 Strategy Comparison Matrix")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Scatter: Risk vs Reward
        fig_scatter = px.scatter(
            filtered_summaries, x="win_rate", y="profit_factor", size="trades", color="family",
            hover_name="strategy", title="Win Rate vs. Profit Factor (Bubble = Trade Vol)",
            labels={"win_rate": "Win Rate %", "profit_factor": "Profit Factor"},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_scatter.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_v2:
        # Bar: Total Points by Strategy
        fig_bar = px.bar(
            filtered_summaries.sort_values("net_points"), x="net_points", y="strategy", color="family",
            orientation='h', title="Total Net Points Captured",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    # 4. Deep-Dive Explorer
    st.markdown("---")
    st.subheader("🔍 Individual Strategy Deep-Dive")
    
    selected_strat = st.selectbox("Select Strategy to Audit", filtered_summaries['strategy'].unique(), index=0)
    
    if selected_strat:
        trades_df = cached_load_strategy_trades(ROOT, selected_strat)
        if not trades_df.empty:
            trades_df = trades_df.sort_values('entry_time')
            trades_df['cum_pnl'] = trades_df['points_captured'].cumsum()
            
            # --- Equity Curve & Drawdown ---
            st.markdown(f"#### Performance Curve: {selected_strat}")
            fig_equity = go.Figure()
            
            # Main Equity Line
            fig_equity.add_trace(go.Scatter(
                x=trades_df['entry_time'], y=trades_df['cum_pnl'],
                name="Cumulative PnL", line=dict(color="#00b4d8", width=3),
                fill='tozeroy', fillcolor='rgba(0, 180, 216, 0.1)'
            ))
            
            # Max Drawdown overlay
            cum_max = trades_df['cum_pnl'].cummax()
            drawdown = trades_df['cum_pnl'] - cum_max
            fig_equity.add_trace(go.Scatter(
                x=trades_df['entry_time'], y=drawdown,
                name="Drawdown", line=dict(color="#ff4d6d", width=1),
                fill='tozeroy', fillcolor='rgba(255, 77, 109, 0.1)', yaxis="y2"
            ))
            
            fig_equity.update_layout(
                height=450, margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(title="Points"),
                yaxis2=dict(title="Drawdown", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            # --- Stats Grid ---
            win_df = trades_df[trades_df['points_captured'] > 0]
            loss_df = trades_df[trades_df['points_captured'] <= 0]
            
            s1, s2, s3, s4 = st.columns(4)
            s1.write(f"📊 **Win Rate:** {len(win_df)/len(trades_df):.1%}")
            s1.write(f"📈 **Profit Factor:** {abs(win_df['points_captured'].sum() / (loss_df['points_captured'].sum() or -1)):.2f}")
            
            s2.write(f"🟢 **Avg Win:** {win_df['points_captured'].mean():.2f} pts")
            s2.write(f"🔴 **Avg Loss:** {loss_df['points_captured'].mean():.2f} pts")
            
            s3.write(f"⏱️ **Avg Holding:** {trades_df['holding_minutes'].mean():.1f} mins")
            s3.write(f"⚠️ **Max Loss:** {trades_df['points_captured'].min():.2f} pts")
            
            s4.write(f"🌊 **Max Drawdown:** {drawdown.min():.2f} pts")
            s4.write(f"📅 **Total Trades:** {len(trades_df)}")

            # --- Trade Log ---
            with st.expander("📖 View Full Trade Execution Log"):
                cols = ['entry_time', 'exit_time', 'direction', 'points_captured', 'peak_profit_pts', 'reason']
                available = [c for c in cols if c in trades_df.columns]
                st.dataframe(
                    trades_df[available].sort_values('entry_time', ascending=False),
                    use_container_width=True
                )

    # 5. AI Research Lab
    st.markdown("---")
    st.subheader("🧪 AI Optimization Lab")
    lab_runs = cached_load_agent_lab_runs(ROOT)
    
    if not lab_runs.empty:
        run_col1, run_col2 = st.columns([1, 2])
        selected_run = run_col1.selectbox("Select Optimization Run", lab_runs['run_name'].tolist())
        
        if selected_run:
            detail = cached_load_agent_lab_detail(ROOT, selected_run)
            with run_col1:
                st.markdown("#### Baseline vs. Upgraded")
                st.dataframe(detail['comparison'], hide_index=True, use_container_width=True)
            with run_col2:
                st.markdown("#### 🧠 Agent Research Report")
                st.markdown(f"""<div class="card">{detail['report']}</div>""", unsafe_allow_html=True)
    else:
        st.info("No AI research cycles (Agent Lab) recorded yet.")

if __name__ == "__main__":
    main()
