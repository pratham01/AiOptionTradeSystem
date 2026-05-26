# Option Buyer Rulebook

## Purpose

This rulebook is the operating document for long-option trades inside the trade system. It is designed for Nifty or Bank Nifty style directional buying, but the same structure can be reused for other liquid index or stock options.

## Core Principles

1. Trade only when direction and timing are both clear.
2. Build the setup from the underlying chart, not from the premium chart.
3. Prefer liquid ATM or slightly ITM contracts for directional trades.
4. Keep risk fixed and small relative to capital.
5. Respect theta decay, IV expansion or crush, and scheduled event risk.
6. Exit quickly when the thesis fails or the expected move does not begin on time.

## Pre-Trade Workflow

### Step 1: Underlying Context

- Define the market regime: trend, range, reversal attempt, or event-driven expansion.
- Mark objective levels: previous day high or low, opening range, VWAP, swing highs or lows, support, resistance.
- State the catalyst in one sentence.

### Step 2: Option Selection

- Choose only contracts with acceptable spread, volume, and open interest.
- Prefer ATM or slightly ITM strikes.
- Match expiry to the expected holding period.

### Step 3: Risk Plan

- Fix the stop from underlying invalidation first.
- Translate the stop to premium risk.
- Keep risk per trade around 0.5% to 1.5% of account capital.
- Seek at least 2:1 reward to risk.

### Step 4: Execution Discipline

- Enter only on confirmation.
- No averaging down.
- No boredom trades.
- No revenge trades.

## Rule Set

### Thesis Rule

Every trade must answer:
- What is the direction?
- Why should price move?
- Why should it move now?

### Structure Rule

Only take setups anchored to structure:
- breakout
- pullback
- support bounce
- resistance rejection
- VWAP reclaim or rejection
- range expansion

### Liquidity Rule

Reject trades when:
- bid or ask spread is too wide
- open interest is weak
- volume is too low for clean execution

### Strike Rule

- ATM or slightly ITM is the default.
- Far OTM is allowed only for exceptional momentum setups with intentionally small risk.

### Expiry Rule

- Intraday trades can use short-dated expiry.
- Multi-session ideas must buy enough time.

### Volatility Rule

- Elevated IV requires stronger conviction and faster price movement.
- Avoid blindly buying options ahead of known event volatility.

### Exit Rule

- Exit when the chart-based invalidation is hit.
- Exit when time invalidates the thesis.
- Scale out into rapid expansion if the system supports it.

## LLM Integration Rules

The LLM is a secondary advisor, not the source of authority.

1. The deterministic rule engine runs first.
2. The LLM receives the rulebook and the audit output.
3. The LLM can explain, summarize, or suggest safer revisions.
4. The LLM cannot override failed risk checks.

## Suggested System Architecture

1. Market data ingestion gathers spot, option chain, IV, and event context.
2. A setup detector proposes a candidate trade.
3. The rulebook evaluator grades the proposal.
4. The LLM explains the setup and suggests refinements.
5. The execution layer acts only if the deterministic layer allows it.

## Minimal Input Schema

The suggestion engine should receive:
- symbol
- spot_price
- trend
- setup
- thesis
- timeframe
- expected_move_points
- event_risk
- iv_percentile
- option_type
- strike
- expiry
- premium
- quantity
- stop_loss_premium
- target_premium
- capital
- bid_ask_spread_pct
- open_interest
- volume
- delta
- days_to_expiry
- expected_hold_minutes

## Output Contract

The engine should return:
- verdict: `TRADE_OK`, `WAIT`, or `NO_TRADE`
- score
- rule checks
- summary
- suggested adjustments
- optional llm_note
