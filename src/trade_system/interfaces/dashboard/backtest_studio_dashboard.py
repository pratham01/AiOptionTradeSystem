"""
Modular Quantitative Backtest Studio Dashboard.

Interactive Streamlit interface enabling traders to:
1. Fetch data across timeframes (1m, 3m, 5m, 15m, 30m, 60m, Daily).
2. Configure multi-indicator strategy rules.
3. Define dynamic Stop Loss & Target models (ATR, Fixed %, Swing, Trailing, Breakeven).
4. Enforce trade duration constraints (Intraday 15:15 cutoff, max bars).
5. Generate institutional statistics (Profit factor, win rate, Sharpe, Sortino, Drawdown, MFE/MAE).
6. Visualize trades on interactive candlestick charts, equity curves, and export reports.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

from trade_system.domains.analysis.application.backtesting.modular_backtester import (
    BacktestConfig,
    DurationConfig,
    ModularBacktester,
    StopLossConfig,
    StopLossType,
    TargetConfig,
    TargetType,
    TradeSide,
    TimeframeDataService,
)
from trade_system.domains.analysis.application.backtesting.backtest_reporter import (
    BacktestReporter,
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme & Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem !important; }
    .kpi-card {
        background: #1e2130;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .kpi-title { font-size: 0.85rem; color: #8b949e; font-weight: 500; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
    .kpi-sub { font-size: 0.75rem; color: #6e7681; margin-top: 2px; }
    .pos-val { color: #00d084 !important; }
    .neg-val { color: #ff4b4b !important; }
    .neutral-val { color: #58a6ff !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Logic Library
# ─────────────────────────────────────────────────────────────────────────────

def strat_supertrend_ema(df: pd.DataFrame, i: int):
    if i < 2:
        return None
    curr_st = df["supertrend_dir"].iloc[i]
    prev_st = df["supertrend_dir"].iloc[i - 1]
    close = df["close"].iloc[i]
    ema_200 = df["ema_200"].iloc[i]

    if curr_st == 1 and prev_st == -1 and close > ema_200:
        return TradeSide.LONG
    elif curr_st == -1 and prev_st == 1 and close < ema_200:
        return TradeSide.SHORT
    return None


def strat_rsi_bb_reversion(df: pd.DataFrame, i: int):
    if i < 2:
        return None
    rsi = df["rsi_14"].iloc[i]
    low = df["low"].iloc[i]
    bb_lower = df["bb_lower"].iloc[i]
    high = df["high"].iloc[i]
    bb_upper = df["bb_upper"].iloc[i]

    if rsi < 32 and low <= bb_lower:
        return TradeSide.LONG
    elif rsi > 68 and high >= bb_upper:
        return TradeSide.SHORT
    return None


def strat_dual_ema(df: pd.DataFrame, i: int):
    if i < 2:
        return None
    fast_curr = df["ema_9"].iloc[i]
    fast_prev = df["ema_9"].iloc[i - 1]
    slow_curr = df["ema_21"].iloc[i]
    slow_prev = df["ema_21"].iloc[i - 1]

    if fast_prev <= slow_prev and fast_curr > slow_curr:
        return TradeSide.LONG
    elif fast_prev >= slow_prev and fast_curr < slow_curr:
        return TradeSide.SHORT
    return None


def strat_macd_trend(df: pd.DataFrame, i: int):
    if i < 2:
        return None
    macd_curr = df["macd"].iloc[i]
    macd_prev = df["macd"].iloc[i - 1]
    sig_curr = df["macd_signal"].iloc[i]
    sig_prev = df["macd_signal"].iloc[i - 1]

    if macd_prev <= sig_prev and macd_curr > sig_curr:
        return TradeSide.LONG
    elif macd_prev >= sig_prev and macd_curr < sig_curr:
        return TradeSide.SHORT
    return None


def strat_smc_liquidity_sweep(df: pd.DataFrame, i: int):
    """SMC Liquidity Sweep & Reclaim."""
    if i < 8:
        return None
    bull_sweep = df["bull_sweep"].iloc[i]
    bear_sweep = df["bear_sweep"].iloc[i]
    close = df["close"].iloc[i]
    ema_50 = df["ema_50"].iloc[i]

    if bull_sweep and close > ema_50:
        return TradeSide.LONG
    elif bear_sweep and close < ema_50:
        return TradeSide.SHORT
    return None


def strat_ict_fvg_mitigation(df: pd.DataFrame, i: int):
    """ICT Fair Value Gap (FVG) Mitigation Entry."""
    if i < 6:
        return None
    close = df["close"].iloc[i]
    open_p = df["open"].iloc[i]
    low = df["low"].iloc[i]
    high = df["high"].iloc[i]
    ema_50 = df["ema_50"].iloc[i]

    recent_bull_fvg = df["bull_fvg"].iloc[max(0, i - 5) : i]
    if recent_bull_fvg.any() and close > ema_50:
        fvg_idx = recent_bull_fvg[recent_bull_fvg].index[-1]
        fvg_top = df.loc[fvg_idx, "fvg_top"]
        fvg_mid = df.loc[fvg_idx, "fvg_mid"]
        if low <= fvg_top and close >= fvg_mid and close > open_p:
            return TradeSide.LONG

    recent_bear_fvg = df["bear_fvg"].iloc[max(0, i - 5) : i]
    if recent_bear_fvg.any() and close < ema_50:
        fvg_idx = recent_bear_fvg[recent_bear_fvg].index[-1]
        fvg_bottom = df.loc[fvg_idx, "fvg_bottom"]
        fvg_mid = df.loc[fvg_idx, "fvg_mid"]
        if high >= fvg_bottom and close <= fvg_mid and close < open_p:
            return TradeSide.SHORT

    return None


def strat_order_block_retest(df: pd.DataFrame, i: int):
    """Institutional Order Block (OB) Re-test."""
    if i < 8:
        return None
    close = df["close"].iloc[i]
    open_p = df["open"].iloc[i]
    ema_50 = df["ema_50"].iloc[i]

    recent_bull_ob = df["bull_ob"].iloc[max(0, i - 8) : i]
    if recent_bull_ob.any() and close > ema_50 and close > open_p:
        return TradeSide.LONG

    recent_bear_ob = df["bear_ob"].iloc[max(0, i - 8) : i]
    if recent_bear_ob.any() and close < ema_50 and close < open_p:
        return TradeSide.SHORT

    return None


def strat_wyckoff_amd(df: pd.DataFrame, i: int):
    """Wyckoff / Power of 3 (AMD) Cycle."""
    if i < 16:
        return None
    range_high = df["high"].iloc[i - 16 : i - 2].max()
    range_low = df["low"].iloc[i - 16 : i - 2].min()
    range_span = range_high - range_low
    atr = df["atr"].iloc[i]

    if range_span < (2.0 * atr):
        close = df["close"].iloc[i]
        low = df["low"].iloc[i]
        high = df["high"].iloc[i]
        open_p = df["open"].iloc[i]

        if low < range_low and close > range_low and close > open_p:
            return TradeSide.LONG
        elif high > range_high and close < range_high and close < open_p:
            return TradeSide.SHORT

    return None


from trade_system.domains.analysis.application.backtesting.option_buyer_engine import (
    strategy_option_buyer_momentum_burst,
    strategy_smc_liquidity_sweep_opt,
    strategy_ict_fvg_mitigation_opt,
    strategy_supertrend_sr_opt,
    strategy_wyckoff_amd_opt,
    strategy_connors_rsi_mean_reversion,
    strategy_opening_range_breakout,
    strategy_momentum_pinball,
    strategy_vcp_breakout,
    strategy_expiry_day_gamma_edge,
    strategy_elephant_bar_continuation,
)

STRATEGIES = {
    # ── World-Class Option Buyer Protocol ──
    "🚀 Option Buyer: Momentum Burst": strategy_option_buyer_momentum_burst,
    "🧘 Larry Connors: RSI(2) Mean Reversion": strategy_connors_rsi_mean_reversion,
    "⚡ Oliver Velez: Opening Range Breakout (ORB)": strategy_opening_range_breakout,
    "🐘 Oliver Velez: Elephant Bar Continuation (87%)": strategy_elephant_bar_continuation,
    "🎯 Linda Raschke: Momentum Pinball": strategy_momentum_pinball,
    "📈 Mark Minervini: VCP Range Contraction": strategy_vcp_breakout,
    "💥 Jeff Augen: Expiry Day Gamma Scalp": strategy_expiry_day_gamma_edge,
    "🌀 Wyckoff: AMD (Option Buyer Optimized)": strategy_wyckoff_amd_opt,
    "🏛️ SMC: Liquidity Sweep (Option Buyer Optimized)": strategy_smc_liquidity_sweep_opt,
    "⚡ ICT: Fair Value Gap (Option Buyer Optimized)": strategy_ict_fvg_mitigation_opt,
    "⚡ SuperTrend + S/R & Time Protocol": strategy_supertrend_sr_opt,
    # ── Core Baseline Strategies ──
    "🌀 Wyckoff: AMD (Baseline)": strat_wyckoff_amd,
    "⚡ SuperTrend + 200 EMA (Baseline)": strat_supertrend_ema,
    "🏛️ SMC: Liquidity Sweep (Baseline)": strat_smc_liquidity_sweep,
    "⚡ ICT: Fair Value Gap (Baseline)": strat_ict_fvg_mitigation,
    "🧱 Order Block (OB) Re-test": strat_order_block_retest,
    "🔄 RSI + Bollinger Bands Reversion": strat_rsi_bb_reversion,
    "📈 Fast 9 EMA / 21 EMA Cross": strat_dual_ema,
    "📊 MACD Signal Line Cross": strat_macd_trend,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🔬 Modular Backtest Studio")
    st.caption("Institutional multi-timeframe quantitative simulation, risk modeling & attribution engine.")

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Backtest Configuration")

    # 1. Symbol & Timeframe
    POPULAR_SYMBOLS = [
        "NSE:NIFTY50-INDEX",
        "NSE:NIFTYBANK-INDEX",
        "BSE:SENSEX-INDEX",
        "NSE:FINNIFTY-INDEX",
        "NSE:RELIANCE-EQ",
        "NSE:HDFCBANK-EQ",
        "NSE:TCS-EQ",
        "NSE:INFY-EQ",
        "NSE:ICICIBANK-EQ",
        "NSE:TATAMOTORS-EQ",
    ]
    sym_choice = st.sidebar.selectbox("Symbol", POPULAR_SYMBOLS, index=0)
    custom_sym = st.sidebar.text_input("Or Custom Symbol", "")
    active_symbol = custom_sym.strip() if custom_sym.strip() else sym_choice

    timeframe_label = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "3m", "5m", "15m", "30m", "60m", "Daily"],
        index=3,
    )
    TF_MAP = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "60m": "60", "Daily": "D"}
    active_tf = TF_MAP[timeframe_label]

    # Date Range
    col_d1, col_d2 = st.sidebar.columns(2)
    default_from = date.today() - timedelta(days=90)
    from_date = col_d1.date_input("From Date", default_from)
    to_date = col_d2.date_input("To Date", date.today())

    # Initial Capital
    capital = st.sidebar.number_input("Initial Capital (₹)", min_value=10000.0, value=100000.0, step=10000.0)

    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Strategy & Indicators")
    strat_name = st.sidebar.selectbox("Strategy Rule", list(STRATEGIES.keys()), index=0)

    with st.sidebar.expander("🛡️ Option Buyer Protocol Rules", expanded=True):
        st.markdown(
            """
            * ⏰ **Prime Windows**: 09:30-11:15 & 13:15-15:00 IST
            * 🛑 **Midday Trap Veto**: 11:15-13:15 IST (Theta bleed)
            * 🧱 **S/R Runway**: Vetoes Call under Res / Put above Sup (<0.35%)
            * 🌪️ **Market Regime**: Vetoes breakout in Choppy range
            * ⚡ **India VIX**: Guards against IV crush (falling VIX)
            """
        )

    st.sidebar.markdown("---")
    st.sidebar.header("🛡️ Stop Loss & Target Rules")

    # Stop Loss
    sl_type_str = st.sidebar.selectbox("Stop Loss Mode", ["ATR", "PERCENT", "SWING", "TRAILING_ATR"], index=0)
    sl_val = st.sidebar.number_input("Stop Loss Value (ATR Mult or %)", min_value=0.2, value=1.5, step=0.1)
    be_rr = st.sidebar.number_input("Move SL to Breakeven at R:R (0 = Off)", min_value=0.0, value=1.0, step=0.5)

    # Target
    tp_type_str = st.sidebar.selectbox("Target Mode", ["RR_RATIO", "PERCENT", "ATR"], index=0)

    tp_val = st.sidebar.number_input("Target Value (R:R Ratio or %)", min_value=0.5, value=2.0, step=0.25)

    st.sidebar.markdown("---")
    st.sidebar.header("⏱️ Trade Duration Constraints")
    max_bars = st.sidebar.number_input("Max Holding Bars Timeout", min_value=0, value=50, step=5)
    enable_intraday = st.sidebar.checkbox("Intraday Auto Square-off (15:15 IST)", value=(active_tf != "D"))

    # ── Execute Backtest Buttons ──────────────────────────────────────────────
    col_btn1, col_btn2 = st.sidebar.columns(2)
    run_btn = col_btn1.button("⚡ Run Selected", type="primary", use_container_width=True)
    tourney_btn = col_btn2.button("🏆 Compare All", use_container_width=True)

    cfg = BacktestConfig(
        symbol=active_symbol,
        timeframe=active_tf,
        from_date=from_date,
        to_date=to_date,
        initial_capital=capital,
    )

    sl_cfg = StopLossConfig(
        sl_type=StopLossType(sl_type_str),
        value=sl_val,
        breakeven_at_rr=be_rr if be_rr > 0 else None,
    )

    tp_cfg = TargetConfig(
        target_type=TargetType(tp_type_str),
        value=tp_val,
    )

    dur_cfg = DurationConfig(
        max_bars=max_bars if max_bars > 0 else None,
        intraday_cutoff_time=dtime(15, 15) if (enable_intraday and active_tf != "D") else None,
    )

    # 1. Handle Tournament Execution (Run All Strategies)
    if tourney_btn or ("tournament_df" not in st.session_state and "last_backtest_result" not in st.session_state):
        with st.spinner(f"Running Tournament across all {len(STRATEGIES)} strategies on {active_symbol} ({timeframe_label})..."):
            ds = TimeframeDataService()
            candles = ds.get_data(active_symbol, active_tf, from_date, to_date)
            vix_df = ds.get_data("NSE:INDIAVIX-INDEX", active_tf, from_date, to_date)
            from trade_system.domains.analysis.application.backtesting.indicators_library import apply_indicator_suite
            df_ind = apply_indicator_suite(candles, vix_df=vix_df) if not candles.empty else pd.DataFrame()

            tournament_rows = []
            tournament_results = {}
            for name, s_fn in STRATEGIES.items():
                bt = ModularBacktester(config=cfg, sl_config=sl_cfg, target_config=tp_cfg, duration_config=dur_cfg)
                res = bt.run(entry_signal_fn=s_fn, custom_candles=df_ind)
                tournament_results[name] = res
                met = res.metrics
                tournament_rows.append({
                    "Strategy": name,
                    "Net Profit (₹)": met["net_profit"],
                    "Return %": met["total_return_pct"],
                    "Win Rate %": met["win_rate_pct"],
                    "Profit Factor": met["profit_factor"],
                    "Payoff Ratio": met["payoff_ratio"],
                    "Max DD %": met["max_drawdown_pct"],
                    "Sharpe": met["sharpe_ratio"],
                    "Trades": met["total_trades"],
                    "Avg Duration (Bars)": met["avg_bars_held"],
                })

            tournament_df = pd.DataFrame(tournament_rows).sort_values("Net Profit (₹)", ascending=False).reset_index(drop=True)
            st.session_state["tournament_df"] = tournament_df
            st.session_state["tournament_results"] = tournament_results
            st.session_state["last_backtest_result"] = tournament_results.get(strat_name, tournament_results[tournament_df["Strategy"].iloc[0]])
            st.session_state["last_strat_name"] = strat_name

    # 2. Handle Single Strategy Run
    elif run_btn:
        with st.spinner(f"Running simulation for {strat_name} on {active_symbol} ({timeframe_label})..."):
            ds = TimeframeDataService()
            candles = ds.get_data(active_symbol, active_tf, from_date, to_date)
            vix_df = ds.get_data("NSE:INDIAVIX-INDEX", active_tf, from_date, to_date)
            from trade_system.domains.analysis.application.backtesting.indicators_library import apply_indicator_suite
            df_ind = apply_indicator_suite(candles, vix_df=vix_df) if not candles.empty else pd.DataFrame()
            backtester = ModularBacktester(
                config=cfg,
                sl_config=sl_cfg,
                target_config=tp_cfg,
                duration_config=dur_cfg,
            )
            selected_fn = STRATEGIES[strat_name]
            result = backtester.run(entry_signal_fn=selected_fn, custom_candles=df_ind)
            st.session_state["last_backtest_result"] = result
            st.session_state["last_strat_name"] = strat_name


    result = st.session_state.get("last_backtest_result")
    if not result:
        st.info("Configure parameters on the sidebar and click **'⚡ Run Selected'** or **'🏆 Compare All'**.")
        return

    m = result.metrics
    c = result.config

    # ── Top KPI Metric Cards ──────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    pnl_class = "pos-val" if m["net_profit"] >= 0 else "neg-val"
    col1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">NET PROFIT</div>
            <div class="kpi-value {pnl_class}">₹{m['net_profit']:,.2f}</div>
            <div class="kpi-sub">{m['total_return_pct']:+.2f}% Return</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    win_class = "pos-val" if m["win_rate_pct"] >= 50 else "neutral-val"
    col2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">WIN RATE %</div>
            <div class="kpi-value {win_class}">{m['win_rate_pct']:.1f}%</div>
            <div class="kpi-sub">{m['winning_trades']}W / {m['losing_trades']}L ({m['total_trades']} Total)</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    pf_class = "pos-val" if m["profit_factor"] >= 1.5 else ("neutral-val" if m["profit_factor"] >= 1.0 else "neg-val")
    col3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">PROFIT FACTOR</div>
            <div class="kpi-value {pf_class}">{m['profit_factor']:.2f}</div>
            <div class="kpi-sub">Payoff: {m['payoff_ratio']:.2f}x</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    exp_class = "pos-val" if m["expectancy_inr"] >= 0 else "neg-val"
    col4.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">EXPECTANCY / TRADE</div>
            <div class="kpi-value {exp_class}">₹{m['expectancy_inr']:,.2f}</div>
            <div class="kpi-sub">Avg R: {m['avg_r_multiple']:+.2f}R</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    dd_class = "pos-val" if m["max_drawdown_pct"] <= 10 else "neg-val"
    col5.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">MAX DRAWDOWN</div>
            <div class="kpi-value {dd_class}">{m['max_drawdown_pct']:.2f}%</div>
            <div class="kpi-sub">₹{m['max_drawdown_inr']:,.2f} ({m['max_drawdown_bars']} bars)</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col6.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">AVG DURATION</div>
            <div class="kpi-value neutral-val">{m['avg_bars_held']} bars</div>
            <div class="kpi-sub">{m['avg_duration_mins']:.0f} mins / trade</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive Tabs ──────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏆 Strategy Tournament & Comparison",
            "🕯️ Candlestick Chart & Trade Overlays",
            "📈 Equity Curve & Drawdown",
            "⏱️ Duration & MFE/MAE Excursions",
            "📋 Detailed Trade Ledger & Export",
        ]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 0: Strategy Tournament & Comparative Matrix
    # ─────────────────────────────────────────────────────────────────────────
    with tab0:
        st.subheader("🏆 Multi-Strategy Quantitative Tournament")
        tourney_df = st.session_state.get("tournament_df")
        tourney_results = st.session_state.get("tournament_results", {})

        if tourney_df is None or tourney_df.empty:
            st.info("Click **'🏆 Compare All'** in the sidebar to run all strategies simultaneously.")
        else:
            best_strat = tourney_df.iloc[0]
            st.success(
                f"🥇 **Tournament Leader: {best_strat['Strategy']}** | Net PnL: **₹{best_strat['Net Profit (₹)']:,.2f}** "
                f"({best_strat['Return %']:+.2f}%) | Win Rate: **{best_strat['Win Rate %']:.1f}%** | "
                f"Profit Factor: **{best_strat['Profit Factor']:.2f}** | Max DD: **{best_strat['Max DD %']:.2f}%**"
            )

            # Leaderboard Table with Rank Badges
            df_display = tourney_df.copy()
            badges = ["🥇 1st", "🥈 2nd", "🥉 3rd"] + [f"{k}th" for k in range(4, len(df_display) + 1)]
            df_display.insert(0, "Rank", badges[: len(df_display)])
            st.dataframe(
                df_display.style.format(
                    {
                        "Net Profit (₹)": "₹{:,.2f}",
                        "Return %": "{:+.2f}%",
                        "Win Rate %": "{:.1f}%",
                        "Profit Factor": "{:.2f}",
                        "Payoff Ratio": "{:.2f}x",
                        "Max DD %": "{:.2f}%",
                        "Sharpe": "{:.2f}",
                        "Avg Duration (Bars)": "{:.1f}",
                    }
                ),
                use_container_width=True,
                height=320,
            )

            st.markdown("---")
            st.subheader("📈 Multi-Strategy Overlaid Equity Curves")

            # Plot Overlaid Equity Curves
            fig_multi = go.Figure()
            colors = ["#00d084", "#58a6ff", "#a371f7", "#ffa500", "#ff7b72", "#39d353", "#f0883e", "#79c0ff"]
            for idx, (s_name, s_res) in enumerate(tourney_results.items()):
                if s_res and not s_res.equity_curve.empty:
                    color = colors[idx % len(colors)]
                    fig_multi.add_trace(
                        go.Scatter(
                            x=s_res.equity_curve["timestamp"],
                            y=s_res.equity_curve["equity"],
                            mode="lines",
                            name=s_name,
                            line=dict(width=2, color=color),
                        )
                    )

            fig_multi.update_layout(
                height=480,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                yaxis_title="Portfolio Equity (₹)",
                xaxis_title="Date & Time",
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_multi, use_container_width=True)

            # Side-by-side Comparative Charts
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.markdown("**Net Profit (₹) Comparison by Strategy**")
                fig_bar = go.Figure(
                    go.Bar(
                        x=tourney_df["Strategy"],
                        y=tourney_df["Net Profit (₹)"],
                        marker_color=np.where(tourney_df["Net Profit (₹)"] >= 0, "#00d084", "#ff4b4b"),
                        text=[f"₹{v:,.0f}" for v in tourney_df["Net Profit (₹)"]],
                        textposition="outside",
                    )
                )
                fig_bar.update_layout(
                    height=380,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#161b22",
                    yaxis_title="Net Profit (₹)",
                    margin=dict(l=20, r=20, t=30, b=30),
                    xaxis=dict(tickangle=-30),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_ch2:
                st.markdown("**Win Rate % vs. Profit Factor Matrix**")
                fig_matrix = go.Figure(
                    go.Scatter(
                        x=tourney_df["Win Rate %"],
                        y=tourney_df["Profit Factor"],
                        mode="markers+text",
                        marker=dict(
                            size=np.clip(tourney_df["Trades"] * 0.8, 12, 40),
                            color=np.where(tourney_df["Net Profit (₹)"] >= 0, "#00d084", "#ff4b4b"),
                            opacity=0.85,
                            line=dict(width=1, color="#ffffff"),
                        ),
                        text=tourney_df["Strategy"],
                        textposition="top center",
                    )
                )
                fig_matrix.add_hline(y=1.0, line_dash="dash", line_color="#8b949e", annotation_text="Breakeven PF = 1.0")
                fig_matrix.update_layout(
                    height=380,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#161b22",
                    xaxis_title="Win Rate (%)",
                    yaxis_title="Profit Factor",
                    margin=dict(l=20, r=20, t=30, b=30),
                )
                st.plotly_chart(fig_matrix, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: Candlestick Chart & Overlays
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Interactive Price Action & Fills")

        # Fetch candles for plotting
        ds = TimeframeDataService()
        candles = ds.get_data(c.symbol, c.timeframe, c.from_date, c.to_date)
        vix_candles = ds.get_data("NSE:INDIAVIX-INDEX", c.timeframe, c.from_date, c.to_date)

        if candles.empty:
            st.warning("No candle data available for chart.")
        else:
            from trade_system.domains.analysis.application.backtesting.indicators_library import (
                apply_indicator_suite,
            )

            df_plot = apply_indicator_suite(candles, vix_df=vix_candles)

            # Limit display to recent 200 bars for performance
            bars_slider = st.slider("Candles to Display (from end)", 50, min(1000, len(df_plot)), min(200, len(df_plot)))
            plot_slice = df_plot.iloc[-bars_slider:].copy().reset_index(drop=True)

            # Format categorical x-axis labels
            is_intraday = c.timeframe not in ["D", "DAY", "DAILY"]
            fmt = "%d %b %H:%M" if is_intraday else "%d %b %Y"
            plot_slice["x_label"] = plot_slice["timestamp"].dt.strftime(fmt)

            # Telemetry Banner: Current Market Regime, India VIX, Dynamic S/R Levels
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            latest_bar = plot_slice.iloc[-1]
            cur_regime = latest_bar.get("market_regime", "UNKNOWN")
            cur_vix = latest_bar.get("vix_level", 13.5)
            cur_vix_reg = latest_bar.get("vix_regime", "NORMAL")
            cur_res = latest_bar.get("nearest_res", 0.0)
            cur_sup = latest_bar.get("nearest_sup", 0.0)
            dist_res = latest_bar.get("dist_to_res_pct", 0.0)
            dist_sup = latest_bar.get("dist_to_sup_pct", 0.0)

            col_m1.metric("Market Regime", cur_regime)
            col_m2.metric("India VIX", f"{cur_vix:.2f}", cur_vix_reg)
            col_m3.metric("Nearest Resistance", f"₹{cur_res:,.1f}", f"+{dist_res:.2f}% runway")
            col_m4.metric("Nearest Support", f"₹{cur_sup:,.1f}", f"-{dist_sup:.2f}% runway")

            fig = go.Figure()

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=plot_slice["x_label"],
                    open=plot_slice["open"],
                    high=plot_slice["high"],
                    low=plot_slice["low"],
                    close=plot_slice["close"],
                    name="Price",
                    increasing_line_color="#00d084",
                    decreasing_line_color="#ff4b4b",
                )
            )

            # Dynamic Support & Resistance Zones
            if "nearest_res" in plot_slice.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_slice["x_label"],
                        y=plot_slice["nearest_res"],
                        mode="lines",
                        line=dict(color="#ff7b72", width=1.5, dash="dash"),
                        name="Resistance Ceiling",
                    )
                )
            if "nearest_sup" in plot_slice.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_slice["x_label"],
                        y=plot_slice["nearest_sup"],
                        mode="lines",
                        line=dict(color="#39d353", width=1.5, dash="dash"),
                        name="Support Floor",
                    )
                )

            # Indicator Overlays
            if "supertrend" in plot_slice.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_slice["x_label"],
                        y=plot_slice["supertrend"],
                        mode="lines",
                        line=dict(color="#00d084", width=1.5, dash="dot"),
                        name="SuperTrend",
                    )
                )

            if "ema_200" in plot_slice.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_slice["x_label"],
                        y=plot_slice["ema_200"],
                        mode="lines",
                        line=dict(color="#ffa500", width=1.5),
                        name="200 EMA",
                    )
                )


            # Map trades to slice
            time_to_label = dict(zip(plot_slice["timestamp"], plot_slice["x_label"]))
            slice_start = plot_slice["timestamp"].iloc[0]
            slice_end = plot_slice["timestamp"].iloc[-1]

            active_trades = [
                t for t in result.trades if (t.entry_time >= slice_start or (t.exit_time and t.exit_time >= slice_start))
            ]

            # Add Trade Markers
            for t in active_trades:
                entry_lbl = time_to_label.get(t.entry_time)
                if entry_lbl:
                    marker_symbol = "triangle-up" if t.side == TradeSide.LONG else "triangle-down"
                    marker_color = "#00d084" if t.side == TradeSide.LONG else "#ff4b4b"
                    fig.add_trace(
                        go.Scatter(
                            x=[entry_lbl],
                            y=[t.entry_price],
                            mode="markers+text",
                            marker=dict(symbol=marker_symbol, size=14, color=marker_color),
                            text=[f"#{t.trade_id} {t.side.value} @ ₹{t.entry_price:,.1f}"],
                            textposition="top center" if t.side == TradeSide.LONG else "bottom center",
                            name=f"Trade #{t.trade_id} Entry",
                            showlegend=False,
                        )
                    )

                if t.exit_time:
                    exit_lbl = time_to_label.get(t.exit_time)
                    if exit_lbl:
                        exit_color = "#00d084" if t.net_pnl > 0 else "#ff4b4b"
                        fig.add_trace(
                            go.Scatter(
                                x=[exit_lbl],
                                y=[t.exit_price],
                                mode="markers+text",
                                marker=dict(symbol="x", size=12, color=exit_color),
                                text=[f"Exit #{t.trade_id}: ₹{t.net_pnl:+,.1f}"],
                                textposition="bottom center" if t.side == TradeSide.LONG else "top center",
                                name=f"Trade #{t.trade_id} Exit",
                                showlegend=False,
                            )
                        )

            fig.update_layout(
                height=550,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                xaxis=dict(type="category", nticks=15, tickangle=-30),
                yaxis=dict(title="Price (₹)", side="right"),
                margin=dict(l=20, r=40, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: Equity Curve & Drawdown
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Cumulative Equity & Underwater Drawdown")

        if result.equity_curve.empty:
            st.info("No equity curve available.")
        else:
            eq_df = result.equity_curve.copy()
            peak = np.maximum.accumulate(eq_df["equity"].values)
            eq_df["peak"] = peak
            eq_df["drawdown_pct"] = -((peak - eq_df["equity"].values) / peak) * 100.0

            fig_eq = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Portfolio Equity Curve (₹)", "Drawdown Depth (%)"),
                row_heights=[0.7, 0.3],
            )

            # Equity line
            fig_eq.add_trace(
                go.Scatter(
                    x=eq_df["timestamp"],
                    y=eq_df["equity"],
                    mode="lines",
                    line=dict(color="#00d084", width=2),
                    name="Equity (₹)",
                    fill="tozeroy",
                    fillcolor="rgba(0, 208, 132, 0.08)",
                ),
                row=1,
                col=1,
            )

            # Cash baseline
            fig_eq.add_trace(
                go.Scatter(
                    x=eq_df["timestamp"],
                    y=eq_df["cash"],
                    mode="lines",
                    line=dict(color="#8b949e", width=1, dash="dash"),
                    name="Realized Cash",
                ),
                row=1,
                col=1,
            )

            # Drawdown area
            fig_eq.add_trace(
                go.Scatter(
                    x=eq_df["timestamp"],
                    y=eq_df["drawdown_pct"],
                    mode="lines",
                    line=dict(color="#ff4b4b", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(255, 75, 75, 0.2)",
                    name="Drawdown %",
                ),
                row=2,
                col=1,
            )

            fig_eq.update_layout(
                height=520,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                margin=dict(l=30, r=30, t=40, b=30),
                showlegend=True,
            )
            fig_eq.update_yaxes(title_text="Equity (₹)", row=1, col=1)
            fig_eq.update_yaxes(title_text="Drawdown %", row=2, col=1)
            st.plotly_chart(fig_eq, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: Duration & MFE / MAE Excursions
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Trade Duration, Excursion & Exit Analysis")

        if result.trades_df.empty:
            st.info("No trades executed.")
        else:
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown("**MFE vs. MAE Scatter Distribution**")
                fig_scatter = go.Figure()
                fig_scatter.add_trace(
                    go.Scatter(
                        x=result.trades_df["mae_pct"],
                        y=result.trades_df["mfe_pct"],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=np.where(result.trades_df["net_pnl"] > 0, "#00d084", "#ff4b4b"),
                            opacity=0.8,
                            line=dict(width=1, color="#ffffff"),
                        ),
                        text=[
                            f"#{r['trade_id']} PnL: ₹{r['net_pnl']:.1f}<br>Bars: {r['bars_held']}"
                            for _, r in result.trades_df.iterrows()
                        ],
                        hoverinfo="text",
                    )
                )
                fig_scatter.update_layout(
                    height=360,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#161b22",
                    xaxis_title="Max Adverse Excursion (MAE Drawdown %)",
                    yaxis_title="Max Favorable Excursion (MFE Run-Up %)",
                    margin=dict(l=30, r=30, t=30, b=30),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_d2:
                st.markdown("**Exit Trigger Breakdown**")
                exit_counts = result.trades_df["exit_reason"].value_counts()
                fig_pie = go.Figure(
                    go.Pie(
                        labels=exit_counts.index,
                        values=exit_counts.values,
                        hole=0.45,
                        marker=dict(
                            colors=["#00d084", "#ff4b4b", "#ffa500", "#58a6ff", "#a371f7"],
                        ),
                    )
                )
                fig_pie.update_layout(
                    height=360,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#161b22",
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Holding time distribution
            st.markdown("**Holding Duration (Bars Held) Distribution**")
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=result.trades_df[result.trades_df["net_pnl"] > 0]["bars_held"],
                    name="Winning Trades",
                    marker_color="#00d084",
                    opacity=0.75,
                )
            )
            fig_hist.add_trace(
                go.Histogram(
                    x=result.trades_df[result.trades_df["net_pnl"] < 0]["bars_held"],
                    name="Losing Trades",
                    marker_color="#ff4b4b",
                    opacity=0.75,
                )
            )
            fig_hist.update_layout(
                barmode="overlay",
                height=280,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                xaxis_title="Bars Held in Trade",
                yaxis_title="Trade Count",
                margin=dict(l=30, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: Detailed Trade Ledger & Export
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Executed Trades Ledger")

        if result.trades_df.empty:
            st.info("No trades executed.")
        else:
            col_f1, col_f2 = st.columns([3, 1])
            with col_f1:
                side_filter = st.multiselect("Filter by Side", ["LONG", "SHORT"], default=["LONG", "SHORT"])
            with col_f2:
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = result.trades_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Trades CSV",
                    csv_data,
                    f"backtest_trades_{c.symbol.replace(':', '_')}_{c.timeframe}.csv",
                    "text/csv",
                    use_container_width=True,
                )

            filtered_df = result.trades_df[result.trades_df["side"].isin(side_filter)].copy()

            # Format for display
            display_df = filtered_df.copy()
            display_df["entry_time"] = display_df["entry_time"].astype(str).str.slice(0, 16)
            display_df["exit_time"] = display_df["exit_time"].astype(str).str.slice(0, 16)

            cols_to_show = [
                "trade_id",
                "side",
                "entry_time",
                "entry_price",
                "shares",
                "capital",
                "initial_sl",
                "target",
                "exit_time",
                "exit_price",
                "exit_reason",
                "bars_held",
                "duration_mins",
            ]
            for extra_col in ["regime", "vix", "notes"]:
                if extra_col in display_df.columns:
                    cols_to_show.append(extra_col)
            cols_to_show.extend(["net_pnl", "return_pct", "r_multiple", "mfe_pct", "mae_pct"])

            st.dataframe(
                display_df[cols_to_show],
                use_container_width=True,
                height=450,
            )


            # Export Markdown Report
            st.markdown("---")
            st.subheader("📄 Markdown Report Preview")
            md_content = result.summary_markdown(title=f"{c.symbol} ({c.timeframe}) Quantitative Backtest")
            st.download_button(
                "📥 Download Markdown Report",
                md_content,
                f"backtest_report_{c.symbol.replace(':', '_')}_{c.timeframe}.md",
                "text/markdown",
            )
            with st.expander("View Full Markdown Text"):
                st.code(md_content, language="markdown")


if __name__ == "__main__":
    main()
