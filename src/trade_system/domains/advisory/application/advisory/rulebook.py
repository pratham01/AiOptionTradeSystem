from __future__ import annotations

OPTION_BUYER_RULEBOOK = """
# Option Buyer Rulebook

## 1. Philosophy
- Buy options only when you have both directional clarity and timing clarity.
- Trade the underlying setup first and treat the option as the execution vehicle.
- Preserve capital. The first job of an option buyer is to avoid low-quality trades that bleed theta.

## 2. Market Conditions
- Prefer trending or expanding markets over low-volatility chop.
- Trade liquid underlyings and liquid option contracts only.
- Avoid random contracts with wide spreads, weak volume, or poor open interest.

## 3. Entry Rules
- Every trade must have a written thesis: direction, catalyst, and timing.
- Enter near an objective market structure such as breakout, pullback, range edge, VWAP reclaim, or major support/resistance.
- Do not chase a move after a large impulsive candle unless momentum continuation is part of the setup.
- Define invalidation before entry.

## 4. Strike And Expiry Rules
- For directional buying, prefer ATM or slightly ITM contracts.
- Avoid far OTM contracts unless the risk is intentionally very small and the expected move is explosive.
- Match expiry to the expected duration of the move.
- If the setup may take time, buy more time instead of buying cheaper premium.

## 5. Risk Management
- Risk a fixed fraction of trading capital per trade.
- Typical per-trade risk should stay around 0.5% to 1.5% of total capital.
- Never average down in long options.
- Set a maximum daily loss and stop trading after reaching it.
- Seek at least 2:1 reward-to-risk unless the setup has an unusually high win probability.

## 6. Volatility And Event Awareness
- Be careful buying options into elevated implied volatility.
- Even a correct directional view can fail if IV crush or time decay dominates.
- Major events must be explicitly acknowledged before entry.

## 7. Exit Rules
- Exit when the underlying thesis is invalidated.
- If the expected move does not start quickly, reduce or close the position.
- Do not convert an intraday trade into a hope-based swing.
- Take partial profits into fast premium expansion when appropriate.

## 8. Process Rules
- No boredom trades.
- No revenge trades.
- No "cheap premium" trades without a chart-based edge.
- Journal thesis, setup, strike, expiry, risk, exit, and mistake after every trade.
""".strip()


RULEBOOK_PRINCIPLES = [
    "Trade only when direction and timing are both clear.",
    "Use the option as the execution vehicle, not as the source of the setup.",
    "Prefer ATM or slightly ITM contracts for directional buying.",
    "Avoid wide-spread or illiquid contracts.",
    "Keep per-trade risk capped to a small fraction of capital.",
    "Demand an explicit stop, target, and invalidation.",
    "Respect theta, IV, and event risk.",
    "Do not average down in long options.",
]
