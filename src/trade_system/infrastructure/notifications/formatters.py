from trade_system.core import SessionPlan, TradeSuggestion


def format_session_plan_telegram(plan: SessionPlan) -> str:
    """Format SessionPlan as a rich Telegram HTML message."""
    lines = [
        f"🤖 <b>AI Swarm Trade Plan — {plan.date}</b>",
        f"Generated: {plan.generated_at.strftime('%H:%M IST')}",
        "",
    ]

    if plan.market_context:
        ctx = plan.market_context
        bias_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(ctx.bias, "⚪")
        risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨", "EXTREME": "🛑"}.get(ctx.risk_level, "⚪")
        lines += [
            f"📊 <b>Market Context</b>",
            f"  {bias_emoji} Bias: {ctx.bias} | {risk_emoji} Risk: {ctx.risk_level}",
            f"  Regime: {ctx.regime.value}",
        ]
        if ctx.vix:
            lines.append(f"  VIX: {ctx.vix:.1f} | PCR: {ctx.pcr:.2f}" if ctx.pcr else f"  VIX: {ctx.vix:.1f}")
        if ctx.narrative:
            lines.append(f"  📝 {ctx.narrative}")
        lines.append("")

    if plan.nifty_suggestions:
        lines.append(f"🎯 <b>NIFTY Trades ({len(plan.nifty_suggestions)}/{plan.MAX_NIFTY})</b>")
        for i, s in enumerate(plan.nifty_suggestions, 1):
            lines += _format_suggestion(i, s)
        lines.append("")

    if plan.fo_suggestions:
        lines.append(f"🎯 <b>F&O Stock Trades ({len(plan.fo_suggestions)}/{plan.MAX_FO})</b>")
        for i, s in enumerate(plan.fo_suggestions, 1):
            lines += _format_suggestion(i, s)
        lines.append("")

    if not plan.nifty_suggestions and not plan.fo_suggestions:
        lines.append("⚠️ No valid trade setups found for this session.")

    if plan.agent_notes:
        lines.append(f"💬 {plan.agent_notes}")

    return "\n".join(lines)


def _format_suggestion(num: int, s: TradeSuggestion) -> list[str]:
    dir_emoji = "📈" if s.direction.value == "CALL" else "📉"
    hor_badge = "🔄 Intraday" if s.horizon.value == "INTRADAY" else "🌊 Swing"
    conf_stars = "⭐" * max(1, round(s.confidence * 5))
    sym_short = s.symbol.split(":")[-1].replace("-EQ", "").replace("-INDEX", "")

    lines = [
        f"  {num}. {dir_emoji} <b>BUY {s.direction.value} — {sym_short}</b> [{hor_badge}]",
        f"     Entry: ₹{s.entry_zone_low:.0f} – ₹{s.entry_zone_high:.0f}",
        f"     Target: ₹{s.target:.0f} | SL: ₹{s.stop_loss:.0f}",
        f"     Strike: ₹{s.option_params.suggested_strike:.0f} {s.option_params.expiry_type.upper()} {s.direction.value}",
        f"     R:R: {s.risk_reward():.1f} | Conf: {conf_stars} ({s.confidence:.0%})",
        f"     📝 {s.narrative[:200]}..." if len(s.narrative) > 200 else f"     📝 {s.narrative}",
    ]

    if s.tags:
        lines.append(f"     🏷️ {' | '.join(s.tags)}")

    return lines
