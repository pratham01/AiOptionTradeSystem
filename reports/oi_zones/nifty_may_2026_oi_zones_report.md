# Nifty 50 — OI Zone Backtest Report (May 2026)

_Generated: 2026-05-26 19:22_

## Strategy Comparison

| Metric | OHLCV Zones (30d) | OI Zones (5d) | Flux OB Fixed (30d) |
|--------|-------------------|---------------|---------------------|
| Trades | 24 | 0 | 18 |
| Win Rate | 33.3 | 0.0 | 16.7 |
| Total Pnl Pts | -474.5 | 0.0 | -403.4 |
| Avg Pnl Pts | -19.8 | 0.0 | -22.4 |
| Profit Factor | 0.55 | 0.0 | 0.35 |
| Best Trade | 222.2 | 0.0 | 115.1 |
| Worst Trade | -141.8 | 0.0 | -70.4 |

**Zone Hit Rate** (OI): 0.0% of bars touched a zone

## Today's Key OI Zones (May 26, 2026)

| Dir | Type | Strike | Source | Strength |
|-----|------|--------|--------|----------|
| S | SUPPORT | 23900 | PE_OI_MAX | 1.000 |
| R | RESISTANCE | 24000 | CE_OI_TOP2 | 0.900 |
| S | GAMMA_WALL | 22950 | GEX_WALL | 0.850 |
| S | SUPPORT | 23397 | PREV_LOW | 0.700 |
| S | RESISTANCE | 23691 | PREV_HIGH | 0.700 |
| S | SUPPORT | 23500 | PE_OI_TOP3 | 0.567 |
| S | SUPPORT | 23850 | PE_OI_TOP4 | 0.455 |
| R | RESISTANCE | 24500 | CE_OI_TOP5 | 0.368 |

## Interpretation


- **OHLCV Zones**: Uses only previous-day High/Low/Close as S/R. Simple and available every day.
- **OI Zones**: Enriches with max CE/PE OI strikes, volume clusters, Max Pain, and GEX walls.
  These are *institutional* levels that options market-makers must defend.
- **Flux OB (Fixed)**: Uses the corrected swing-window logic (strictly historical bars) and
  correct OB origin (last bullish/bearish candle before impulse, not highest/lowest).

### Key Insights
- OI walls (max CE/PE OI) have historically acted as strong S/R on expiry weeks
- Max Pain is the most reliable level on expiry day (Thursday)
- Gamma Walls act as mean-reversion magnets in low-IV environments
