"""Risk management for agent decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trade_system.core import Signal, SignalType


@dataclass(slots=True)
class RiskLimits:
    """Risk limits configuration."""

    max_position_size: int = 100
    max_daily_loss: float = 5000.0        # Adjusted for option buying
    max_drawdown_pct: float = 2.0
    max_concurrent_trades: int = 1        # Only 1 trade at a time
    max_trades_per_day: int = 2          # Max 2 trades per day as requested
    max_risk_per_trade_pct: float = 2.0  # Slightly higher for options
    min_risk_reward: float = 2.0         # Minimum 1:2 RR ratio


@dataclass(slots=True)
class RiskAssessment:
    """Result of risk assessment."""

    approved: bool
    risk_score: float  # 0-1, higher is riskier
    position_size: int
    stop_loss: float | None
    take_profit: float | None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """Evaluates risk for trading decisions."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self._daily_pnl: float = 0.0
        self._open_trades: int = 0
        self._total_trades_today: int = 0
        self._max_loss_hit: bool = False

    def assess(
        self,
        signal: Signal,
        context: dict[str, Any],
    ) -> RiskAssessment:
        """Assess risk for a signal."""
        reasons: list[str] = []

        # 1. Check daily loss limit
        if self._daily_pnl <= -self.limits.max_daily_loss:
            return RiskAssessment(
                approved=False,
                risk_score=1.0,
                position_size=0,
                stop_loss=None,
                take_profit=None,
                reason="Daily loss limit reached",
            )

        # 2. Check total trades per day (Max 2)
        if self._total_trades_today >= self.limits.max_trades_per_day:
            return RiskAssessment(
                approved=False,
                risk_score=1.0,
                position_size=0,
                stop_loss=None,
                take_profit=None,
                reason=f"Max daily trades ({self.limits.max_trades_per_day}) reached",
            )

        # 3. Check concurrent trades
        if self._open_trades >= self.limits.max_concurrent_trades:
            return RiskAssessment(
                approved=False,
                risk_score=1.0,
                position_size=0,
                stop_loss=None,
                take_profit=None,
                reason=f"Active trade already exists ({self.limits.max_concurrent_trades} max)",
            )

        # 4. Check Risk-Reward Ratio
        stop_loss = signal.metadata.get("stop_loss")
        take_profit = signal.metadata.get("take_profit")
        entry_price = signal.price

        if stop_loss and take_profit and entry_price:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr = reward / risk if risk > 0 else 0
            
            if rr < self.limits.min_risk_reward:
                reasons.append(f"Insufficient RR ratio: {rr:.2f} (min {self.limits.min_risk_reward})")

        # 5. Calculate position size based on risk per trade
        account_value = context.get("account_value", 100000.0)
        risk_per_trade = account_value * (self.limits.max_risk_per_trade_pct / 100)

        if stop_loss and entry_price:
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit > 0:
                position_size = int(risk_per_trade / risk_per_unit)
                position_size = min(position_size, self.limits.max_position_size)
            else:
                position_size = 1
        else:
            position_size = 1

        # Calculate risk score
        risk_score = self._calculate_risk_score(signal, context)

        if risk_score > 0.8:
            reasons.append("High risk score")

        approved = len(reasons) == 0

        return RiskAssessment(
            approved=approved,
            risk_score=risk_score,
            position_size=position_size if approved else 0,
            stop_loss=stop_loss,
            take_profit=signal.metadata.get("take_profit"),
            reason="; ".join(reasons) if reasons else "Risk check passed",
            metadata={
                "max_position_size": self.limits.max_position_size,
                "risk_per_trade_pct": self.limits.max_risk_per_trade_pct,
            },
        )

    def _calculate_risk_score(self, signal: Signal, context: dict[str, Any]) -> float:
        """Calculate a risk score (0-1)."""
        score = 0.0

        # Volatility component
        volatility = context.get("volatility", 0.0)
        score += min(volatility * 10, 0.3)

        # Signal confidence inverse
        score += (1 - signal.confidence) * 0.3

        # Market condition
        if context.get("high_spread", False):
            score += 0.2
        if context.get("low_liquidity", False):
            score += 0.2

        return min(score, 1.0)

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L tracking."""
        self._daily_pnl += pnl

    def on_trade_open(self) -> None:
        """Track open trades."""
        self._open_trades += 1
        self._total_trades_today += 1

    def on_trade_close(self) -> None:
        """Track trade closure."""
        self._open_trades = max(0, self._open_trades - 1)

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self._daily_pnl = 0.0
        self._total_trades_today = 0
        self._max_loss_hit = False
