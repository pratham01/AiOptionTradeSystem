    subprocess.run([sys.executable, "-m", "streamlit", "run", str(file_path)])



# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card { background: #1e2329; border-radius: 12px; padding: 16px; margin: 4px; }
    .suggestion-call { border-left: 4px solid #00d084; }
    .suggestion-put { border-left: 4px solid #ff4757; }
    .tag-pill { background: #2d3748; border-radius: 20px; padding: 2px 10px; font-size: 12px; margin: 2px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Trading Agent Dashboard")
st.caption(f"Today: {date.today().strftime('%A, %d %B %Y')} | Nifty Options Buyer System")

# ------------------------------------------------------------------
# Today's Session Plan
# ------------------------------------------------------------------
st.markdown("## 📋 Today's Session Plan")

today_trades = load_session_plan()

if not today_trades:
    st.info("No trade suggestions yet for today. Run the orchestrator from the sidebar or wait for pre-market to start.")
else:
    nifty_trades = [t for t in today_trades if t.is_nifty]
    fo_trades = [t for t in today_trades if not t.is_nifty]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Nifty Trades", f"{len(nifty_trades)} / 2")
    with col2:
        st.metric("🎯 F&O Trades", f"{len(fo_trades)} / 3")
    with col3:
        avg_conf = sum(t.confidence for t in today_trades) / len(today_trades)
        st.metric("📊 Avg Confidence", f"{avg_conf:.0%}")
    with col4:
        pending = sum(1 for t in today_trades if t.outcome == "PENDING")
        st.metric("⏳ Pending", pending)

    st.markdown("### Nifty Index Trades")
    _render_trade_cards(nifty_trades)
    st.markdown("### F&O Stock Trades")
    _render_trade_cards(fo_trades)

# ------------------------------------------------------------------
# Performance Analytics
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(f"## 📈 Performance Analytics ({lookback_days}d)")

closed_trades = load_closed_trades(lookback_days)


def _render_trade_cards(trades):
    if not trades:
        st.caption("No suggestions in this category.")
        return
    for t in trades:
        direction_cls = "suggestion-call" if t.direction == "CALL" else "suggestion-put"
        dir_emoji = "📈" if t.direction == "CALL" else "📉"
        outcome_badge = {
            "PENDING": "⏳ Pending",
            "WIN": "✅ Win",
            "LOSS": "❌ Loss",
            "NEUTRAL": "➡️ Neutral",
            "EXPIRED": "💨 Expired",
        }.get(t.outcome, t.outcome)

        sym_short = t.symbol.split(":")[-1].replace("-EQ", "").replace("-INDEX", "")

        with st.container():
            st.markdown(f"""
            <div class='metric-card {direction_cls}'>
                <b>{dir_emoji} BUY {t.direction} — {sym_short}</b>
                &nbsp;&nbsp;<span style='opacity:0.7'>{t.horizon}</span>
                &nbsp;&nbsp;<span style='float:right'>{outcome_badge}</span>
                <br/>
                <span style='font-size:13px'>
                Entry: ₹{t.entry_zone_low:.0f}–{t.entry_zone_high:.0f} &nbsp;|&nbsp;
                Target: ₹{t.target:.0f} &nbsp;|&nbsp;
                SL: ₹{t.stop_loss:.0f} &nbsp;|&nbsp;
                Strike: ₹{t.option_strike or 'N/A':.0f} {t.option_expiry_type or ''} {t.direction}
                </span>
                <br/>
                <span style='font-size:12px; opacity:0.8'>Conf: {t.confidence:.0%} &nbsp;|&nbsp; {t.narrative[:180] if t.narrative else ''}</span>
            </div>
            """, unsafe_allow_html=True)
            st.write("")


# Rerender removed
# ------------------------------------------------------------------
# Performance Analytics
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(f"## 📈 Performance Analytics ({lookback_days}d)")

closed_trades = load_closed_trades(lookback_days)

if closed_trades:
    import pandas as pd

    df = pd.DataFrame([{
        "date": t.date,
        "symbol": t.symbol.split(":")[-1].replace("-EQ", ""),
        "direction": t.direction,
        "horizon": t.horizon,
        "outcome": t.outcome,
        "confidence": t.confidence,
        "pnl_pct": t.actual_pnl_pct or 0.0,
        "is_nifty": bool(t.is_nifty),
    } for t in closed_trades])

    total = len(df)
    wins = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()
    win_rate = wins / total if total > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades", total)
    c2.metric("Wins", wins)
    c3.metric("Losses", losses)
    c4.metric("Win Rate", f"{win_rate:.1%}")
    c5.metric("Avg P&L", f"{df['pnl_pct'].mean():.2f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Win Rate by Horizon")
        for horizon in df["horizon"].unique():
            sub = df[df["horizon"] == horizon]
            wr = (sub["outcome"] == "WIN").mean()
            st.write(f"**{horizon}**: {wr:.1%} ({len(sub)} trades)")

    with col_right:
        st.markdown("#### Win Rate by Direction")
        for direction in df["direction"].unique():
            sub = df[df["direction"] == direction]
            wr = (sub["outcome"] == "WIN").mean()
            st.write(f"**{direction}**: {wr:.1%} ({len(sub)} trades)")

    st.markdown("#### Recent Trades")
    display_df = df[["date", "symbol", "direction", "horizon", "outcome", "confidence", "pnl_pct"]].head(20)
    display_df = df.copy()
    st.dataframe(
        display_df,
        width=None, # use_container_width=True is actually still preferred in 1.57 for dataframes
        column_config={
            "Conf": st.column_config.ProgressColumn("Conf", min_value=0, max_value=1),
            "PnL%": st.column_config.NumberColumn("PnL%", format="%.2f%%"),
        }
    )

else:
    st.info("No closed trades in the lookback window yet.")

# ------------------------------------------------------------------
# Agent Weights & Evolution
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("## ⚙️ Agent Weights (Self-Evolved)")

weights = load_agent_weights()
if weights:
    w_col1, w_col2 = st.columns(2)
    for i, (agent_name, agent_weights) in enumerate(weights.items()):
        col = w_col1 if i == 0 else w_col2
        with col:
            st.markdown(f"#### {agent_name.replace('_', ' ').title()}")
            for k, v in sorted(agent_weights.items(), key=lambda x: x[1], reverse=True):
                st.progress(v, text=f"{k}: {v:.3f}")
else:
    st.info("No evolved weights found yet. Run the evolution loop after some closed trades.")

# ------------------------------------------------------------------
# Swarm Intelligence & Skills
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧠 Swarm Intelligence")

col_skills, col_directive = st.columns(2)

with col_skills:
    st.markdown("### 📜 Learned Skills")
    from trade_system.application.agent.skill_registry import SkillRegistry
    registry = SkillRegistry()
    skills = registry.list_skills()
    if skills:
        for skill in skills:
            st.markdown(f"• **{skill.replace('-', ' ').title()}**")
    else:
        st.info("No skills learned yet. Run the post-market loop to generate them.")

with col_directive:
    st.markdown("### ✍️ User Directive")
    user_input = st.text_area("Give instructions to the AI Swarm:", placeholder="e.g. 'Prioritize IT sector today' or 'Be more conservative with BankNifty'")
    if st.button("Apply Directive"):
        st.success("Directive applied to next session run!")

st.markdown("### 📡 Agent Activity Stream")
# Simulated activity log
activities = [
    {"time": "09:00", "agent": "PreMarketNewsAgent", "msg": "Crude oil is up 2%. Sentiment: VOLATILE."},
    {"time": "09:05", "agent": "OptionChainAgent", "msg": "Nifty PCR at 0.85. Heavy call writing at 24500."},
    {"time": "09:15", "agent": "IndexDecisionAgent", "msg": "Market Open. Standing aside due to VIX spike."},
    {"time": "10:30", "agent": "FoStockSuggesterAgent", "msg": "RELIANCE showing Volume Delta breakout. Suggesting BUY CALL."},
]
for act in reversed(activities):
    st.markdown(f"`{act['time']}` **{act['agent']}**: {act['msg']}")

st.markdown("---")
st.caption("🤖 AI Trading Agent | Agentic Options Trading System | Advisory Only — Not Financial Advice")
