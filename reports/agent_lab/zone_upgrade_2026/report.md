# Autonomous Zone Upgrade Cycle 2026

This experimental agent cycle does not auto-deploy to live trading.

## Candidate Config

```json
{
  "enabled_zone_types": [],
  "disabled_zone_types": [
    "prev_day_vah",
    "prev_day_close",
    "prev_day_open",
    "bull_fvg",
    "range_support_20",
    "bear_fvg"
  ],
  "min_volume_ratio": 1.25,
  "min_body_ratio": 0.45
}
```

## Comparison

| metric      |   baseline |   upgraded |   delta |
|:------------|-----------:|-----------:|--------:|
| avg_points  |     -11.97 |          0 |   11.97 |
| avg_r       |       0.08 |          0 |   -0.08 |
| eod_exits   |       0    |          0 |    0    |
| losses      |       4    |          0 |   -4    |
| max_loss    |    -112.5  |          0 |  112.5  |
| max_win     |      68.7  |          0 |  -68.7  |
| net_points  |     -83.8  |          0 |   83.8  |
| sl_hits     |       3    |          0 |   -3    |
| target_hits |       2    |          0 |   -2    |
| trades      |       7    |          0 |   -7    |
| trend_exits |       2    |          0 |   -2    |
| win_rate    |      42.86 |          0 |  -42.86 |
| wins        |       3    |          0 |   -3    |

## LLM Review (deterministic)

No LLM configured. Deterministic research review only.