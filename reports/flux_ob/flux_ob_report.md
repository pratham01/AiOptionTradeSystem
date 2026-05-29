# Flux Charts Volumized Order Blocks Backtest Report

This report summarizes the performance of the Python-translated Flux Volumized Order Blocks Strategy.

## Strategy Parameters
- Swing Length: 10
- Max ATR Multiplier: 3.5
- Stop Buffer: 3.0 points
- Target R-to-R: 2.0 (fallback)
- Square-off at: 15:15 IST

## Yearly Results Summary

| year       |   trades |   wins |   losses |   win_rate |   net_points |   avg_points |   avg_win |   avg_loss |   profit_factor |   expectancy |   total_r |   avg_r |   target_hits |   sl_hits |   eod_exits |   max_drawdown_points |   max_win |   max_loss |
|:-----------|---------:|-------:|---------:|-----------:|-------------:|-------------:|----------:|-----------:|----------------:|-------------:|----------:|--------:|--------------:|----------:|------------:|----------------------:|----------:|-----------:|
| 2026       |        7 |      2 |        5 |      28.57 |       -25.45 |        -3.64 |     36.65 |     -19.75 |            0.74 |        -3.64 |     -1    |   -0.14 |             2 |         5 |           0 |                -73.45 |     40.5  |     -31.35 |
| HISTORICAL |     2666 |    732 |     1934 |      27.46 |     -5191.85 |        -1.95 |     56.46 |     -24.05 |            0.89 |        -1.95 |   -170.21 |   -0.06 |           404 |      1831 |         431 |              -6128.25 |    301.75 |    -136    |
| ALL        |     2673 |    734 |     1939 |      27.46 |     -5217.3  |        -1.95 |     56.41 |     -24.04 |            0.89 |        -1.95 |   -171.21 |   -0.06 |           406 |      1836 |         431 |              -6128.25 |    301.75 |    -136    |

## Trading Methodology
1. **Long entry:** Price touches an unmitigated Bullish Order Block.
2. **Short entry:** Price touches an unmitigated Bearish Order Block.
3. **Long stop-loss:** Placed below the bottom of the active order block.
4. **Short stop-loss:** Placed above the top of the active order block.
5. **Exits:** Target opposing order block or 2.0R multiplier, or intraday square-off at 15:15.