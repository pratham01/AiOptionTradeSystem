# Option Edge Backtest Report

## Overall Performance

| Metric | Value |
|--------|-------|
| Total Trades | 96 |
| Win Rate | 43.8% |
| Avg P&L / Trade | +0.087% |
| Cumulative P&L | +8.32% |
| Avg Holding Time | 32 min |

## By Exit Reason

| exit_reason    |   count |   avg_pnl |   total_pnl |
|:---------------|--------:|----------:|------------:|
| EOD_SQUARE_OFF |       5 |     0     |       0     |
| SL_HIT         |      49 |    -0.369 |     -18.082 |
| TARGET_1       |      42 |     0.629 |      26.402 |

## By Direction

| direction   |   count |   win_rate |   avg_pnl |
|:------------|--------:|-----------:|----------:|
| CALL        |      58 |     39.655 |     0.055 |
| PUT         |      38 |     50     |     0.135 |

## By Regime

| regime   |   count |   win_rate |   avg_pnl |
|:---------|--------:|-----------:|----------:|
| NEUTRAL  |      78 |     46.154 |     0.108 |
| TRENDING |      18 |     33.333 |    -0.004 |

## Monthly Summary

| month   |   total_trades |   wins |   losses |   win_rate |   avg_pnl_pct |   total_pnl_pct |   avg_win_pct |   avg_loss_pct |   max_win_pct |   max_loss_pct |   avg_holding_min |   regime_trending |   regime_neutral |   regime_choppy |
|:--------|---------------:|-------:|---------:|-----------:|--------------:|----------------:|--------------:|---------------:|--------------:|---------------:|------------------:|------------------:|-----------------:|----------------:|
| 2026-05 |             30 |     17 |       13 |    56.6667 |     0.19384   |          5.8152 |      0.613488 |      -0.354931 |        1.3924 |        -0.9126 |             30.5  |                 6 |               24 |               0 |
| 2026-06 |             60 |     24 |       36 |    40      |     0.0506733 |          3.0404 |      0.650012 |      -0.348886 |        1.0084 |        -0.7881 |             34.25 |                 6 |               54 |               0 |
| 2026-07 |              6 |      1 |        5 |    16.6667 |    -0.0893    |         -0.5358 |      0.372    |      -0.18156  |        0.372  |        -0.3561 |             20    |                 6 |                0 |               0 |