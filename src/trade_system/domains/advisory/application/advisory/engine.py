from __future__ import annotations

from dataclasses import dataclass, field
from math import inf


@dataclass(slots=True)
class TradeContext:
    symbol: str
    spot_price: float
    trend: str = "neutral"
    setup: str = ""
    thesis: str = ""
    timeframe: str = "intraday"
    expected_move_points: float = 0.0
    event_risk: str = "none"
    iv_percentile: float | None = None
    vix: float | None = None


@dataclass(slots=True)
class TradeProposal:
    option_type: str
    strike: float
    expiry: str
    premium: float
    quantity: int
    stop_loss_premium: float
    target_premium: float
    capital: float
    bid_ask_spread_pct: float
    open_interest: int
    volume: int
    delta: float | None = None
    days_to_expiry: int | None = None
    expected_hold_minutes: int | None = None
    average_down_planned: bool = False


@dataclass(slots=True)
class RuleCheck:
    name: str
    status: str
    detail: str


@dataclass(slots=True)
class TradeAdvice:
    verdict: str
    score: int
    checks: list[RuleCheck] = field(default_factory=list)
    summary: str = ""
    suggested_adjustments: list[str] = field(default_factory=list)


class TradeAdvisor:
    def evaluate(self, context: TradeContext, proposal: TradeProposal) -> TradeAdvice:
        checks = [
            self._check_thesis(context),
            self._check_setup(context),
            self._check_liquidity(proposal),
            self._check_strike_selection(proposal),
            self._check_expiry(context, proposal),
            self._check_risk_reward(proposal),
            self._check_position_risk(proposal),
            self._check_iv(context),
            self._check_event_risk(context),
            self._check_average_down(proposal),
        ]

        failed = [check for check in checks if check.status == "fail"]
        warned = [check for check in checks if check.status == "warn"]
        score = sum(10 for check in checks if check.status == "pass") + sum(
            4 for check in checks if check.status == "warn"
        )

        if failed:
            verdict = "NO_TRADE"
        elif len(warned) >= 3:
            verdict = "WAIT"
        else:
            verdict = "TRADE_OK"

        adjustments = self._build_adjustments(context, proposal, checks)
        summary = self._build_summary(verdict=verdict, failed=failed, warned=warned)
        return TradeAdvice(
            verdict=verdict,
            score=score,
            checks=checks,
            summary=summary,
            suggested_adjustments=adjustments,
        )

    def _check_thesis(self, context: TradeContext) -> RuleCheck:
        if len(context.thesis.strip()) < 25:
            return RuleCheck("thesis", "fail", "Trade thesis is missing or too vague.")
        return RuleCheck("thesis", "pass", "Trade has a written thesis.")

    def _check_setup(self, context: TradeContext) -> RuleCheck:
        valid_keywords = ("breakout", "pullback", "reversal", "vwap", "support", "resistance", "range")
        if any(word in context.setup.lower() for word in valid_keywords):
            return RuleCheck("setup", "pass", f"Setup is anchored to structure: {context.setup}.")
        return RuleCheck("setup", "warn", f"Setup `{context.setup}` is not clearly anchored to market structure.")

    def _check_liquidity(self, proposal: TradeProposal) -> RuleCheck:
        if proposal.bid_ask_spread_pct > 1.0:
            return RuleCheck("liquidity", "fail", "Bid/ask spread is too wide for disciplined option buying.")
        if proposal.open_interest < 1000 or proposal.volume < 500:
            return RuleCheck("liquidity", "warn", "Contract liquidity is weaker than preferred.")
        return RuleCheck("liquidity", "pass", "Contract liquidity is acceptable.")

    def _check_strike_selection(self, proposal: TradeProposal) -> RuleCheck:
        if proposal.delta is None:
            return RuleCheck("strike_selection", "warn", "Delta is missing, so moneyness quality cannot be verified.")
        if 0.35 <= abs(proposal.delta) <= 0.65:
            return RuleCheck("strike_selection", "pass", "Strike looks close to ATM or slightly ITM.")
        if abs(proposal.delta) < 0.2:
            return RuleCheck("strike_selection", "fail", "Strike appears too far OTM for regular directional buying.")
        return RuleCheck("strike_selection", "warn", "Strike is tradable but not ideal for a standard option buy.")

    def _check_expiry(self, context: TradeContext, proposal: TradeProposal) -> RuleCheck:
        if proposal.days_to_expiry is None:
            return RuleCheck("expiry", "warn", "Days to expiry not provided.")
        if context.timeframe == "intraday" and proposal.days_to_expiry >= 0:
            return RuleCheck("expiry", "pass", "Expiry matches an intraday-style trade.")
        if context.timeframe != "intraday" and proposal.days_to_expiry < 3:
            return RuleCheck("expiry", "fail", "Swing-style thesis is using too little time to expiry.")
        return RuleCheck("expiry", "pass", "Expiry buffer looks acceptable.")

    def _check_risk_reward(self, proposal: TradeProposal) -> RuleCheck:
        risk = max(proposal.premium - proposal.stop_loss_premium, 0)
        reward = max(proposal.target_premium - proposal.premium, 0)
        ratio = reward / risk if risk else inf
        if risk <= 0:
            return RuleCheck("risk_reward", "fail", "Stop-loss premium must be below entry premium.")
        if ratio >= 2.0:
            return RuleCheck("risk_reward", "pass", f"Reward/risk is healthy at {ratio:.2f}.")
        if ratio >= 1.2:
            return RuleCheck("risk_reward", "warn", f"Reward/risk is only {ratio:.2f}.")
        return RuleCheck("risk_reward", "fail", f"Reward/risk is too weak at {ratio:.2f}.")

    def _check_position_risk(self, proposal: TradeProposal) -> RuleCheck:
        per_unit_risk = max(proposal.premium - proposal.stop_loss_premium, 0)
        total_risk = per_unit_risk * proposal.quantity
        capital_risk_pct = (total_risk / proposal.capital) * 100 if proposal.capital else inf
        if capital_risk_pct > 1.5:
            return RuleCheck("position_risk", "fail", f"Risk is {capital_risk_pct:.2f}% of capital.")
        if capital_risk_pct > 1.0:
            return RuleCheck("position_risk", "warn", f"Risk is elevated at {capital_risk_pct:.2f}% of capital.")
        return RuleCheck("position_risk", "pass", f"Risk is controlled at {capital_risk_pct:.2f}% of capital.")

    def _check_iv(self, context: TradeContext) -> RuleCheck:
        if context.iv_percentile is None:
            return RuleCheck("iv", "warn", "IV percentile not provided.")
        if context.iv_percentile >= 80:
            return RuleCheck("iv", "fail", "IV percentile is very high; IV crush risk is elevated.")
        if context.iv_percentile >= 65:
            return RuleCheck("iv", "warn", "IV percentile is elevated; entry needs strong momentum.")
        return RuleCheck("iv", "pass", "IV conditions are acceptable for option buying.")

    def _check_event_risk(self, context: TradeContext) -> RuleCheck:
        level = context.event_risk.lower().strip()
        if level in {"major", "high"}:
            return RuleCheck("event_risk", "warn", "Major event risk is present and must be intentional.")
        return RuleCheck("event_risk", "pass", "No major event risk flagged.")

    def _check_average_down(self, proposal: TradeProposal) -> RuleCheck:
        if proposal.average_down_planned:
            return RuleCheck("average_down", "fail", "Averaging down is not allowed in this rulebook.")
        return RuleCheck("average_down", "pass", "No average-down plan detected.")

    def _build_adjustments(
        self,
        context: TradeContext,
        proposal: TradeProposal,
        checks: list[RuleCheck],
    ) -> list[str]:
        adjustments: list[str] = []
        statuses = {check.name: check.status for check in checks}
        if statuses["liquidity"] != "pass":
            adjustments.append("Choose a more liquid ATM contract with tighter spread and higher open interest.")
        if statuses["strike_selection"] != "pass":
            adjustments.append("Move closer to ATM or slightly ITM to improve responsiveness.")
        if statuses["risk_reward"] != "pass":
            adjustments.append("Rework target or entry so the trade offers at least 2:1 reward-to-risk.")
        if statuses["position_risk"] != "pass":
            adjustments.append("Cut size so max loss stays within 1% to 1.5% of capital.")
        if statuses["iv"] != "pass":
            adjustments.append("Avoid the entry until IV cools down or momentum confirms strongly.")
        if statuses["expiry"] != "pass" and context.timeframe != "intraday":
            adjustments.append("Use more time to expiry for a non-intraday thesis.")
        if not adjustments:
            adjustments.append("Trade qualifies under the current rulebook. Execute only if the trigger prints on the chart.")
        return adjustments

    def _build_summary(
        self,
        *,
        verdict: str,
        failed: list[RuleCheck],
        warned: list[RuleCheck],
    ) -> str:
        if verdict == "NO_TRADE":
            names = ", ".join(check.name for check in failed)
            return f"Rejected by the rulebook due to failed checks: {names}."
        if verdict == "WAIT":
            names = ", ".join(check.name for check in warned)
            return f"Trade is not rejected, but too many warnings are active: {names}."
        return "Trade passes the current rulebook with manageable risk."
