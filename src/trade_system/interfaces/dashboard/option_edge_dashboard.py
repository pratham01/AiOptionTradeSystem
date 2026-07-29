"""
Option Edge Dashboard — Streamlit page for the Intraday Option Edge pipeline.

Shows:
  1. Market Regime banner (TRENDING/NEUTRAL/CHOPPY)
  2. Shortlisted Universe table (~25 stocks)
  3. Edge-scored alerts with entry/option details
  4. Historical backtest performance (if available)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime
from pathlib import Path

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from sqlalchemy import text

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem !important; }

    .oe-hero {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1a2e 100%);
        padding: 0.7rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border-left: 5px solid #f59e0b;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .oe-hero h3 { margin: 0; color: #fbbf24; font-size: 1.15rem; }
    .oe-hero .hero-sub { color: #94a3b8; font-size: 0.78rem; }

    .regime-card {
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .regime-trending {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .regime-neutral {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15) 0%, rgba(234, 179, 8, 0.05) 100%);
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .regime-choppy {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .regime-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .regime-trending .regime-label { color: #22c55e; }
    .regime-neutral .regime-label { color: #eab308; }
    .regime-choppy .regime-label { color: #ef4444; }
    .regime-sub { color: #94a3b8; font-size: 0.75rem; margin-top: 4px; }

    .signal-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 2px;
    }
    .chip-green { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .chip-yellow { background: rgba(234, 179, 8, 0.15); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.3); }
    .chip-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .chip-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }

    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(245, 158, 11, 0.4) 50%, transparent 100%);
        margin: 0.6rem 0;
        border: none;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0.5rem 0 0.4rem;
    }
    .section-header h3 { margin: 0; font-size: 1rem; color: #e2e8f0; }
    .section-header .badge {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 600;
    }

    .alert-card {
        background: linear-gradient(180deg, #13132b 0%, #1a1a2e 100%);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .alert-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .alert-symbol { font-size: 1.1rem; font-weight: 700; }
    .alert-call { color: #22c55e; }
    .alert-put { color: #ef4444; }
    .score-badge {
        font-size: 0.85rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 16px;
    }
    .score-high { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
    .score-mid { background: rgba(234, 179, 8, 0.2); color: #eab308; }
    .score-low { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="oe-hero">
    <h3>🎯 Intraday Option Edge</h3>
    <span class="hero-sub">Universe Filter • Regime Detection • Confluence Scoring • Smart Entry • Strike Selection</span>
</div>
""", unsafe_allow_html=True)


# ── Date Selector ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_available_dates():
    engine = get_engine()
    query = text("SELECT DISTINCT date(timestamp) as d FROM ohlcv_15m WHERE symbol NOT LIKE '%INDEX%' ORDER BY d DESC")
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return [row[0] for row in result if row[0] is not None]
    except Exception:
        return []

dates = fetch_available_dates()

from datetime import datetime
today_str = date.today().strftime("%Y-%m-%d")
is_weekday = datetime.today().weekday() < 5
is_after_nine = datetime.now().time() >= datetime.strptime("09:00:00", "%H:%M:%S").time()

if is_weekday and is_after_nine:
    if today_str not in dates:
        dates.insert(0, today_str)

col_date, col_refresh = st.columns([3, 1])
with col_date:
    selected = st.selectbox("📅 Trading Date", dates[:30] if dates else ["No data"], index=0)
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if not dates or selected == "No data":
    st.warning("No 15m data available in the database.")
    st.stop()

target_date = date.fromisoformat(selected) if isinstance(selected, str) else selected


# ── Run Pipeline ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Running Option Edge pipeline...")
def run_pipeline(target_date_str):
    td = date.fromisoformat(target_date_str)

    # Stage 1: Shortlist
    from trade_system.domains.analysis.application.analysis.fo_intraday_shortlist import FOIntradayShortlist
    shortlist_engine = FOIntradayShortlist()
    candidates = shortlist_engine.shortlist(td)

    # Stage 2: Regime
    from trade_system.domains.analysis.application.analysis.regime_detector import RegimeDetector
    regime_engine = RegimeDetector()
    regime = regime_engine.detect(td)

    # Stage 3+4+5: Full pipeline
    from trade_system.domains.analysis.application.analysis.intraday_option_edge import IntradayOptionEdgePipeline
    pipeline = IntradayOptionEdgePipeline(max_alerts_per_cycle=10, min_risk_reward=1.0)
    alerts = pipeline.scan(target_date=td)

    return candidates, regime, alerts

candidates, regime, alerts = run_pipeline(target_date.isoformat())


# ── Section 1: Regime Banner ──────────────────────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

regime_class = {
    "TRENDING": "regime-trending",
    "NEUTRAL": "regime-neutral",
    "CHOPPY": "regime-choppy",
}.get(regime.regime.value, "regime-neutral")

regime_icon = {"TRENDING": "🚀", "NEUTRAL": "⚖️", "CHOPPY": "🌀"}.get(regime.regime.value, "⚖️")

col_regime, col_signals = st.columns([1, 3])

with col_regime:
    st.markdown(f"""
    <div class="regime-card {regime_class}">
        <p class="regime-label">{regime_icon} {regime.regime.value}</p>
        <p class="regime-sub">Min Edge Score: {regime.score_threshold}</p>
    </div>
    """, unsafe_allow_html=True)

with col_signals:
    cols = st.columns(4)
    metrics = [
        ("Gap %", f"{regime.gap_pct:+.2f}%"),
        ("VIX", f"{regime.vix:.1f}"),
        ("30m/ADR", f"{regime.first_30m_range_pct:.0f}%"),
        ("Body/Wick", f"{regime.body_wick_ratio:.0%}"),
    ]
    for i, (label, val) in enumerate(metrics):
        cols[i].metric(label, val)

    # Signal chips
    chips_html = ""
    for sig in regime.signals:
        if "Trending" in sig or "optimal" in sig or "room" in sig or "conviction" in sig:
            chip_class = "chip-green"
        elif "Choppy" in sig or "expensive" in sig or "extended" in sig or "doji" in sig:
            chip_class = "chip-red"
        else:
            chip_class = "chip-yellow"
        chips_html += f'<span class="signal-chip {chip_class}">{sig}</span>'
    st.markdown(chips_html, unsafe_allow_html=True)


# ── Section 2: Shortlisted Universe ───────────────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="section-header">
    <h3>📋 Shortlisted Universe</h3>
    <span class="badge">{len(candidates)} stocks</span>
</div>
""", unsafe_allow_html=True)

if candidates:
    rows = []
    
    is_today = (target_date == date.today())
    live_quotes = {}
    if is_today:
        from trade_system.interfaces.dashboard.shared_broker import fetch_live_quotes
        symbols_to_fetch = [c.symbol for c in candidates]
        if alerts:
            symbols_to_fetch.extend([a.symbol for a in alerts])
        live_quotes = fetch_live_quotes(list(set(symbols_to_fetch)))

    for c in candidates:
        clean_sym = c.symbol.replace("NSE:", "").replace("-EQ", "")
        bias_emoji = {"STRONG_CALL": "🟢🟢", "CALL": "🟢", "STRONG_PUT": "🔴🔴", "PUT": "🔴", "NEUTRAL": "⚪"}.get(c.bias, "⚪")
        
        display_ltp = c.ltp
        display_chg = c.change_pct
        if is_today and c.symbol in live_quotes:
            q = live_quotes[c.symbol]
            display_ltp = q.last_price or q.close or q.open or c.ltp
            if q.change_percent is not None:
                display_chg = q.change_percent
                
        rows.append({
            "Symbol": clean_sym,
            "Sector": c.sector,
            "LTP": f"₹{display_ltp:,.2f}",
            "Chg%": f"{display_chg:+.1f}%",
            "Bias": f"{bias_emoji} {c.bias}",
            "ST Dir": "🟢 Bull" if c.daily_supertrend_dir == 1 else "🔴 Bear" if c.daily_supertrend_dir == -1 else "⚪",
            "Compressed": "🔥 Yes" if c.is_compressed else "",
            "Vol Surge": f"{c.volume_surge:.1f}x" if c.volume_surge >= 1.5 else "",
            "52W High": "📈" if c.is_near_52w_high else "",
            "52W Low": "📉" if c.is_near_52w_low else "",
            "Near ATH": "👑 Yes" if getattr(c, "is_near_ath", False) else "",
            "Near Swing": "🎯 High" if getattr(c, "is_near_swing_high", False) else "🎯 Low" if getattr(c, "is_near_swing_low", False) else "",
            "Reasons": len(c.reasons),
        })

    df_shortlist = pd.DataFrame(rows).sort_values("Reasons", ascending=False).reset_index(drop=True)
    st.dataframe(df_shortlist, use_container_width=True, hide_index=True, height=350)
else:
    st.info("No stocks passed the universe filter for this date.")


# ── Section 3: Edge-Scored Alerts ─────────────────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="section-header">
    <h3>🎯 Actionable Alerts</h3>
    <span class="badge">{len(alerts)} signals</span>
</div>
""", unsafe_allow_html=True)

if alerts:
    for alert in alerts:
        clean_sym = alert.symbol.replace("NSE:", "").replace("-EQ", "")
        dir_class = "alert-call" if alert.direction == "CALL" else "alert-put"
        dir_icon = "📈" if alert.direction == "CALL" else "📉"
        dir_label = "CALL" if alert.direction == "CALL" else "PUT"
        
        # Override with live quote if available
        display_ltp = alert.ltp
        if 'is_today' in locals() and is_today and alert.symbol in live_quotes:
            q = live_quotes[alert.symbol]
            display_ltp = q.last_price or q.close or q.open or alert.ltp

        score_class = "score-high" if alert.edge_score >= 65 else "score-mid" if alert.edge_score >= 55 else "score-low"

        # Alert card
        st.markdown(f"""
        <div class="alert-card">
            <div class="alert-header">
                <span class="alert-symbol {dir_class}">{dir_icon} {clean_sym} — {dir_label}</span>
                <span class="score-badge {score_class}">{alert.edge_score:.0f}/100</span>
            </div>
            
            <div class="alert-details">
                <div class="detail-item" style="color: #94a3b8; font-size: 0.78rem;">
                    <strong>LTP:</strong> ₹{display_ltp:,.2f} &nbsp;&bull;&nbsp; {alert.sector} &nbsp;&bull;&nbsp; {alert.entry_type.replace('_', ' ').title()} &nbsp;&bull;&nbsp; {alert.direction_confidence:.0%} layers agree
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Details in columns
        col_entry, col_option, col_signals_col = st.columns([1, 1, 1])

        with col_entry:
            st.markdown("**📍 Entry Levels**")
            st.markdown(f"""
            - Entry: ₹{alert.entry_price:,.2f}
            - Stop Loss: ₹{alert.stop_loss:,.2f}
            - Target 1: ₹{alert.target_1:,.2f}
            - Target 2: ₹{alert.target_2:,.2f}
            - Stock R:R: **{alert.risk_reward_stock:.1f}:1**
            """)

        with col_option:
            if alert.option:
                st.markdown("**🔔 Option Suggestion**")
                st.markdown(f"""
                - Strike: **{alert.option.display_strike}**
                - Premium: ₹{alert.option.est_premium_low:.0f}–{alert.option.est_premium_high:.0f}
                - Delta: {alert.option.est_delta:.2f}
                - Option R:R: {alert.option.risk_reward:.1f}:1
                - Lot: {alert.option.lot_size} | Cost: ~₹{alert.option.lot_value:,.0f}
                """)
            else:
                st.markdown("*No suitable option strike found*")

        with col_signals_col:
            st.markdown("**📊 Key Signals**")
            for sig in alert.key_signals[:5]:
                st.markdown(f"- {sig}")

            structural = [r for r in alert.shortlist_reasons if "Sector" not in r]
            if structural:
                st.markdown("**🔍 Structure**")
                for reason in structural:
                    st.markdown(f"- {reason}")

        st.markdown("---")
else:
    st.info("No alerts generated for this date. This could mean the regime is too choppy, or no stocks passed the confluence + entry filters.")


# ── Section 4: Backtest Performance Summary ───────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

backtest_path = Path("reports/option_edge/option_edge_summary.csv")
trades_path = Path("reports/option_edge/option_edge_trades.csv")

if backtest_path.exists():
    st.markdown("""
    <div class="section-header">
        <h3>📊 Backtest Performance</h3>
        <span class="badge">Historical</span>
    </div>
    """, unsafe_allow_html=True)

    summary = pd.read_csv(backtest_path)
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"]) if trades_path.exists() else pd.DataFrame()

    if not summary.empty:
        # Key metrics
        total_trades = int(summary["total_trades"].sum()) if "total_trades" in summary.columns else 0
        win_rate = summary["win_rate"].mean() if "win_rate" in summary.columns else 0
        avg_pnl = summary["avg_pnl_pct"].mean() if "avg_pnl_pct" in summary.columns else 0
        total_pnl = summary["total_pnl_pct"].sum() if "total_pnl_pct" in summary.columns else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", total_trades)
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Avg P&L / Trade", f"{avg_pnl:+.2f}%")
        m4.metric("Cumulative P&L", f"{total_pnl:+.1f}%")

        # Summary table
        st.dataframe(summary, use_container_width=True, hide_index=True)

    if not trades.empty:
        # Equity curve
        trades_sorted = trades.sort_values("exit_time")
        trades_sorted["cum_pnl"] = trades_sorted["pnl_pct"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_sorted["exit_time"],
            y=trades_sorted["cum_pnl"],
            mode="lines",
            line=dict(color="#f59e0b", width=2),
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.1)",
            name="Cumulative P&L %",
        ))
        fig.update_layout(
            title="Equity Curve (Stock Points %)",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis_title="",
            yaxis_title="Cumulative P&L %",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown("""
    <div class="section-header">
        <h3>📊 Backtest Performance</h3>
        <span class="badge">Not yet run</span>
    </div>
    """, unsafe_allow_html=True)
    st.info("Run the backtester to see historical performance: `python -m trade_system.domains.analysis.application.backtesting.option_edge_backtest`")
