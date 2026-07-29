# Autonomous Bollinger Option Intraday Upgrade Cycle 2026

This experimental agent cycle tests BB squeeze to expansion dynamics for option buyers.

## Upgraded Config

```json
{
  "mode": "intraday",
  "period": 20,
  "std_dev": 2.0,
  "squeeze_lookback": 100,
  "squeeze_percentile": 7.5,
  "min_volatility_score": 1.4
}
```

## Comparison vs Baseline

| metric       |      baseline |      upgraded |   delta |
|:-------------|--------------:|--------------:|--------:|
| final_equity |  99935        |  99935        |       0 |
| initial_cash | 100000        | 100000        |       0 |
| total_return |    -65        |    -65        |       0 |
| trade_count  |      3        |      3        |       0 |
| win_rate     |      0.333333 |      0.333333 |       0 |

## LLM Review (error)

LLM Call failed: 401 Client Error: Unauthorized for url: https://api.openai.com/v1/responses