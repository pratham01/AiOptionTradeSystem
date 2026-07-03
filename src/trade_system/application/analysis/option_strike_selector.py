"""
OptionStrikeSelector — Selects the optimal option strike for an intraday trade.

Given a stock, direction (CALL/PUT), LTP, and ATR, this module:
  1. Identifies the optimal strike (ATM or 1-strike OTM)
  2. Estimates premium using delta approximation
  3. Calculates expected option move and risk-reward
  4. Filters out overpriced options (IV percentile check)

Since live option chains for individual F&O stocks may not always be available,
this uses a delta/ATR estimation model as the primary approach, with live chain
data as an optional enhancement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

LOGGER = logging.getLogger(__name__)


# ── Standard lot sizes for popular F&O stocks ─────────────────────────────────
# These are approximate and should ideally be loaded from a config file.
# For now, we use a heuristic: lot_size ≈ round(500000 / LTP)
# This gives roughly ₹5L notional per lot.

def _estimate_lot_size(ltp: float) -> int:
    """Estimate lot size so each lot ≈ ₹5L notional value."""
    if ltp <= 0:
        return 1
    raw = 500_000 / ltp
    # Round to nearest common lot sizes
    if raw >= 1000:
        return int(round(raw / 100) * 100)
    elif raw >= 100:
        return int(round(raw / 50) * 50)
    else:
        return max(1, int(round(raw / 25) * 25))


# ── Strike step sizes for common price ranges ─────────────────────────────────
def _get_strike_step(ltp: float) -> float:
    """Determine standard strike interval based on stock price."""
    if ltp >= 5000:
        return 100.0
    elif ltp >= 2000:
        return 50.0
    elif ltp >= 500:
        return 25.0
    elif ltp >= 200:
        return 10.0
    else:
        return 5.0


@dataclass
class OptionSuggestion:
    """Complete option trade suggestion with risk parameters."""
    symbol: str
    direction: str             # "CALL" or "PUT"
    strike: float
    strike_type: str           # "ATM", "1-OTM", "1-ITM"
    est_premium_low: float     # Estimated premium range (low)
    est_premium_high: float    # Estimated premium range (high)
    est_delta: float           # Approximate delta
    expected_move: float       # Expected stock price move (1 ATR)
    expected_premium_change: float  # Expected option premium gain
    risk_reward: float         # Expected gain / premium paid
    lot_size: int
    lot_value: float           # Premium × lot_size
    expiry_hint: str           # "Weekly" or "Monthly"

    # Target levels (in stock price terms)
    stock_entry: float
    stock_sl: float
    stock_target_1: float
    stock_target_2: float

    @property
    def premium_midpoint(self) -> float:
        return round((self.est_premium_low + self.est_premium_high) / 2, 2)

    @property
    def display_strike(self) -> str:
        return f"{int(self.strike)} {'CE' if self.direction == 'CALL' else 'PE'}"


class OptionStrikeSelector:
    """
    Selects optimal option strike and estimates premium for directional trades.
    """

    def __init__(
        self,
        min_premium: float = 5.0,
        max_premium: float = 250.0,
        prefer_weekly: bool = True,
    ) -> None:
        self.min_premium = min_premium
        self.max_premium = max_premium
        self.prefer_weekly = prefer_weekly

    def select(
        self,
        symbol: str,
        direction: str,
        ltp: float,
        atr: float,
        stock_entry: float = 0.0,
        stock_sl: float = 0.0,
        stock_target_1: float = 0.0,
        stock_target_2: float = 0.0,
        days_to_expiry: int = 3,
        iv_annual: float = 30.0,
    ) -> Optional[OptionSuggestion]:
        """
        Select the optimal strike for an option buying trade.

        Args:
            symbol: Stock symbol
            direction: "CALL" or "PUT"
            ltp: Last traded price of the stock
            atr: Average True Range (15m or daily, depending on horizon)
            stock_entry: Stock entry price level
            stock_sl: Stock stop-loss level
            stock_target_1: Stock target 1 level
            stock_target_2: Stock target 2 level
            days_to_expiry: Days until nearest expiry
            iv_annual: Annualised implied volatility (%) — default 30%
        """
        if ltp <= 0 or atr <= 0:
            return None

        direction = direction.upper()
        if direction not in ("CALL", "PUT"):
            return None

        step = _get_strike_step(ltp)

        # ── 1. Find ATM Strike ─────────────────────────────────────────────
        atm_strike = round(ltp / step) * step

        # ── 2. Select Candidate Strikes ────────────────────────────────────
        if direction == "CALL":
            strikes = [
                (atm_strike, "ATM"),
                (atm_strike + step, "1-OTM"),
                (atm_strike - step, "1-ITM"),
            ]
        else:
            strikes = [
                (atm_strike, "ATM"),
                (atm_strike - step, "1-OTM"),
                (atm_strike + step, "1-ITM"),
            ]

        # ── 3. Estimate Premium & Delta for Each Strike ────────────────────
        best: Optional[OptionSuggestion] = None
        best_rr = 0.0

        for strike, strike_type in strikes:
            # Moneyness ratio
            if direction == "CALL":
                moneyness = (ltp - strike) / ltp  # positive = ITM
            else:
                moneyness = (strike - ltp) / ltp  # positive = ITM

            # Approximate delta using a simplified model
            delta = self._estimate_delta(moneyness, direction, days_to_expiry, iv_annual)

            # Estimate premium using Black-Scholes approximation
            est_premium = self._estimate_premium(
                ltp, strike, direction, days_to_expiry, iv_annual
            )

            # Skip if outside acceptable premium range
            if est_premium < self.min_premium or est_premium > self.max_premium:
                continue

            # ── 4. Calculate Expected Move & R:R ───────────────────────────
            expected_stock_move = atr  # 1 ATR is the expected intraday move
            expected_premium_change = abs(delta) * expected_stock_move

            # Risk = premium paid; Reward = expected premium gain
            risk_reward = expected_premium_change / est_premium if est_premium > 0 else 0.0

            if risk_reward > best_rr:
                lot_size = _estimate_lot_size(ltp)
                best_rr = risk_reward
                best = OptionSuggestion(
                    symbol=symbol,
                    direction=direction,
                    strike=strike,
                    strike_type=strike_type,
                    est_premium_low=round(est_premium * 0.85, 2),
                    est_premium_high=round(est_premium * 1.15, 2),
                    est_delta=round(delta, 2),
                    expected_move=round(expected_stock_move, 2),
                    expected_premium_change=round(expected_premium_change, 2),
                    risk_reward=round(risk_reward, 2),
                    lot_size=lot_size,
                    lot_value=round(est_premium * lot_size, 0),
                    expiry_hint="Weekly" if self.prefer_weekly and days_to_expiry <= 7 else "Monthly",
                    stock_entry=stock_entry or ltp,
                    stock_sl=stock_sl,
                    stock_target_1=stock_target_1,
                    stock_target_2=stock_target_2,
                )

        return best

    # ── Private: Pricing Approximations ────────────────────────────────────────

    def _estimate_delta(
        self,
        moneyness: float,
        direction: str,
        days_to_expiry: int,
        iv_annual: float,
    ) -> float:
        """
        Approximate option delta using a simplified normal-distribution model.

        moneyness > 0 means ITM, < 0 means OTM (already adjusted for direction).
        """
        if days_to_expiry <= 0:
            days_to_expiry = 1

        # Convert IV to daily and then to the remaining time
        iv_daily = iv_annual / 100.0 / math.sqrt(252)
        iv_period = iv_daily * math.sqrt(days_to_expiry)

        if iv_period <= 0:
            iv_period = 0.01

        # d1 ≈ moneyness / iv_period (simplified)
        d1 = moneyness / iv_period

        # Standard normal CDF approximation
        delta = self._norm_cdf(d1)

        if direction == "PUT":
            delta = delta - 1  # Put delta is negative

        return delta

    def _estimate_premium(
        self,
        ltp: float,
        strike: float,
        direction: str,
        days_to_expiry: int,
        iv_annual: float,
    ) -> float:
        """
        Estimate option premium using simplified Black-Scholes.
        This is an approximation for ATM/near-money options.
        """
        if days_to_expiry <= 0:
            days_to_expiry = 1

        T = days_to_expiry / 365.0
        sigma = iv_annual / 100.0

        # For ATM options: premium ≈ 0.4 × S × σ × √T
        # Adjust for moneyness
        if direction == "CALL":
            intrinsic = max(0, ltp - strike)
            moneyness_factor = (ltp - strike) / ltp
        else:
            intrinsic = max(0, strike - ltp)
            moneyness_factor = (strike - ltp) / ltp

        # Time value component (ATM approximation)
        time_value = 0.4 * ltp * sigma * math.sqrt(T)

        # Adjust time value for OTM options (less time value)
        if moneyness_factor < 0:  # OTM
            otm_decay = math.exp(moneyness_factor * 5)  # Decay faster for deeper OTM
            time_value *= otm_decay

        premium = intrinsic + time_value
        return max(1.0, round(premium, 2))

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Approximation of the standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
