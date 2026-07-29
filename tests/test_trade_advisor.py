from trade_system.domains.advisory.application.advisory import TradeAdvisor, TradeContext, TradeProposal


def test_trade_advisor_accepts_high_quality_trade() -> None:
    advisor = TradeAdvisor()
    context = TradeContext(
        symbol="NSE:NIFTY50-INDEX",
        spot_price=22485.0,
        trend="up",
        setup="breakout above opening range and VWAP",
        thesis="The index is reclaiming VWAP after a shallow pullback and is breaking the opening range with sector breadth support.",
        timeframe="intraday",
        expected_move_points=100.0,
        event_risk="low",
        iv_percentile=44.0,
    )
    proposal = TradeProposal(
        option_type="CE",
        strike=22500,
        expiry="2026-03-26",
        premium=180.0,
        quantity=50,
        stop_loss_premium=150.0,
        target_premium=245.0,
        capital=250000.0,
        bid_ask_spread_pct=0.3,
        open_interest=100000,
        volume=50000,
        delta=0.48,
        days_to_expiry=3,
    )

    advice = advisor.evaluate(context, proposal)

    assert advice.verdict == "TRADE_OK"
    assert advice.score >= 80


def test_trade_advisor_rejects_far_otm_high_iv_trade() -> None:
    advisor = TradeAdvisor()
    context = TradeContext(
        symbol="NSE:NIFTY50-INDEX",
        spot_price=22485.0,
        trend="up",
        setup="momentum",
        thesis="Quick upside scalp.",
        timeframe="intraday",
        expected_move_points=35.0,
        event_risk="major",
        iv_percentile=88.0,
    )
    proposal = TradeProposal(
        option_type="CE",
        strike=22900,
        expiry="2026-03-26",
        premium=42.0,
        quantity=300,
        stop_loss_premium=30.0,
        target_premium=50.0,
        capital=200000.0,
        bid_ask_spread_pct=1.4,
        open_interest=250,
        volume=80,
        delta=0.12,
        days_to_expiry=2,
        average_down_planned=True,
    )

    advice = advisor.evaluate(context, proposal)

    assert advice.verdict == "NO_TRADE"
    failed = {check.name for check in advice.checks if check.status == "fail"}
    assert {"thesis", "liquidity", "strike_selection", "risk_reward", "iv", "average_down"} <= failed
