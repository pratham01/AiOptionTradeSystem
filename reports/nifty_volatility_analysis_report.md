# Nifty 50 15m Volatility and Big Movements Analysis Report

**Analysis Period**: 2020-01-01 to 2026-06-24 (~6.5 years)
**Generated At**: 2026-06-24 11:15:11 IST

## 1. Methodology & Parameters
This off-market analysis loads 15-minute Nifty 50 index candles and filters out the largest intraday movements.
- **Detection Method**: Fixed rolling thresholds.
  - 1-Candle (15m): Body $\ge 0.5\%$ OR Range $\ge 0.8\%$
  - 2-Candle (30m): Body $\ge 0.6\%$ OR Range $\ge 0.9\%$
  - 3-Candle (45m): Body $\ge 0.7\%$ OR Range $\ge 1.0\%$
  - 4-Candle (60m): Body $\ge 0.8\%$ OR Range $\ge 1.2\%$
- **Total 15m Candles Analyzed**: 39,680
- **Big Movements Detected**: 1,953 (4.92% of all candles)

### Trigger Timeline Analysis (Reason Breakdown)
| Trigger Type | Occurrences | % of Flagged Moves |
| :--- | :---: | :---: |
| 3C-Range | 903 | 46.24% |
| 3C-Body | 807 | 41.32% |
| 2C-Body | 798 | 40.86% |
| 4C-Body | 786 | 40.25% |
| 2C-Range | 756 | 38.71% |
| 4C-Range | 744 | 38.10% |
| 1C-Body | 566 | 28.98% |
| 1C-Range | 454 | 23.25% |

## 2. Overall Nifty 50 15m Statistics
| Metric | Absolute Body Return (%) | High-Low Range (%) | India VIX |
| :--- | :---: | :---: | :---: |
| **Dataset Mean** | 0.103% | 0.204% | 17.39 |
| **Std Dev** | 0.135% | 0.178% | - |
| **Dataset Max** | 6.387% | 7.151% | - |
| **Mean on Big Move Days** | - | - | 31.95 |

### India VIX Correlation Analysis
- Correlation between daily VIX Close and 15m Absolute Body Returns: **0.4018**
- Correlation between daily VIX Close and 15m Candle Ranges: **0.5766**
- *Note: A higher VIX level is correlated with larger individual intraday candle moves, which is clearly shown in the positive correlation coefficients above.*

## 3. Distribution of Big Movements by Year
| Year | Total 15m Candles | Big Move Candles | % of Total | Avg Daily VIX (Overall) | Avg Daily VIX (Big Move Days) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 2020 | 6,273 | 938 | 14.95% | 26.76 | 45.05 |
| 2021 | 6,168 | 255 | 4.13% | 18.01 | 21.13 |
| 2022 | 6,179 | 285 | 4.61% | 19.33 | 22.63 |
| 2023 | 6,129 | 60 | 0.98% | 12.45 | 15.71 |
| 2024 | 6,168 | 181 | 2.93% | 14.65 | 17.03 |
| 2025 | 6,205 | 124 | 2.00% | 13.39 | 16.74 |
| 2026 | 2,558 | 110 | 4.30% | 16.57 | 19.92 |

## 4. Distribution of Big Movements by Day of the Week
| Day of the Week | Big Move Candles | Avg Daily VIX on Big Move Days |
| :--- | :---: | :---: |
| Monday | 450 | 30.35 |
| Tuesday | 399 | 32.77 |
| Wednesday | 380 | 34.23 |
| Thursday | 349 | 31.48 |
| Friday | 354 | 32.01 |

## 5. Top 10 Most Volatile Times of Day (IST)
These are the 15-minute candle intervals that most frequently trigger big movements:
| Rank | Candle Start Time | Occurrences | % of All Big Moves |
| :---: | :---: | :---: | :---: |
| 1 | 09:30 | 193 | 9.88% |
| 2 | 09:45 | 183 | 9.37% |
| 3 | 09:15 | 161 | 8.24% |
| 4 | 10:00 | 135 | 6.91% |
| 5 | 15:00 | 112 | 5.73% |
| 6 | 14:30 | 84 | 4.30% |
| 7 | 14:15 | 84 | 4.30% |
| 8 | 14:45 | 81 | 4.15% |
| 9 | 14:00 | 80 | 4.10% |
| 10 | 15:15 | 77 | 3.94% |

*Note: Typically, the market open (09:15) and market close (15:15) segments exhibit the highest concentration of institutional order matching and volatility.*

## 6. Top 10 Largest 15m Body Movements (Absolute Close-Open)
| Rank | Timestamp | Day | Open | Close | Body Change (%) | Range (%) | Trigger Reason | India VIX |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2020-03-13 10:15 | Friday | 8555.15 | 9101.60 | **6.39%** | 7.15% | 1C-Body, 1C-Range, 2C-Range | 51.47 |
| 2 | 2020-03-13 09:15 | Friday | 9107.60 | 8651.00 | **-5.01%** | 5.64% | 1C-Body, 1C-Range | 51.47 |
| 3 | 2020-03-13 10:30 | Friday | 9108.85 | 9489.75 | **4.18%** | 4.54% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Body, 3C-Range | 51.47 |
| 4 | 2020-03-19 11:00 | Thursday | 8040.60 | 8360.55 | **3.98%** | 3.99% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Body, 3C-Range, 4C-Body, 4C-Range | 72.20 |
| 5 | 2024-06-04 09:15 | Tuesday | 23179.50 | 22481.10 | **-3.01%** | 3.11% | 1C-Body, 1C-Range | 26.75 |
| 6 | 2020-03-19 13:45 | Thursday | 8256.85 | 8478.20 | **2.68%** | 3.00% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Body, 3C-Range, 4C-Body, 4C-Range | 72.20 |
| 7 | 2020-03-20 13:45 | Friday | 8873.80 | 8657.45 | **-2.44%** | 2.87% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Range, 4C-Range | 67.10 |
| 8 | 2024-06-04 12:30 | Tuesday | 21296.30 | 21794.35 | **2.34%** | 2.55% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Range, 4C-Body, 4C-Range | 26.75 |
| 9 | 2020-04-03 09:15 | Friday | 8356.55 | 8161.70 | **-2.33%** | 2.84% | 1C-Body, 1C-Range | 55.30 |
| 10 | 2026-02-03 09:15 | Tuesday | 26308.05 | 25786.25 | **-1.98%** | 2.24% | 1C-Body, 1C-Range | 12.90 |

## 7. Top 10 Largest 15m Candle Ranges (High-Low)
| Rank | Timestamp | Day | High | Low | Range (%) | Body Change (%) | Trigger Reason | India VIX |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2020-03-13 10:15 | Friday | 9166.90 | 8555.15 | **7.15%** | 6.39% | 1C-Body, 1C-Range, 2C-Range | 51.47 |
| 2 | 2020-03-13 09:15 | Friday | 9133.20 | 8645.60 | **5.64%** | -5.01% | 1C-Body, 1C-Range | 51.47 |
| 3 | 2020-03-13 10:30 | Friday | 9519.90 | 9106.35 | **4.54%** | 4.18% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Body, 3C-Range | 51.47 |
| 4 | 2020-03-13 10:45 | Friday | 9749.95 | 9367.65 | **4.08%** | -0.31% | 1C-Range, 2C-Body, 2C-Range, 3C-Body, 3C-Range, 4C-Body, 4C-Range | 51.47 |
| 5 | 2020-03-19 11:00 | Thursday | 8361.65 | 8040.60 | **3.99%** | 3.98% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Body, 3C-Range, 4C-Body, 4C-Range | 72.20 |
| 6 | 2020-03-19 09:15 | Thursday | 8099.40 | 7832.55 | **3.41%** | -1.51% | 1C-Body, 1C-Range | 72.20 |
| 7 | 2020-03-17 09:15 | Tuesday | 9314.60 | 9016.85 | **3.30%** | -1.98% | 1C-Body, 1C-Range | 62.93 |
| 8 | 2020-03-25 09:15 | Wednesday | 7980.35 | 7732.10 | **3.21%** | 1.89% | 1C-Body, 1C-Range | 77.63 |
| 9 | 2024-06-04 09:15 | Tuesday | 23179.50 | 22479.40 | **3.11%** | -3.01% | 1C-Body, 1C-Range | 26.75 |
| 10 | 2020-03-19 13:45 | Thursday | 8504.35 | 8256.85 | **3.00%** | 2.68% | 1C-Body, 1C-Range, 2C-Body, 2C-Range, 3C-Body, 3C-Range, 4C-Body, 4C-Range | 72.20 |

## 8. Breakdown Pre-Condition Signals
These candles had 3 or more of the 5 pre-conditions active simultaneously, indicating a high probability of an imminent large directional move.

**Total candles with 3+ pre-conditions**: 2,306 (5.811% of all candles)

### Signal Type Distribution (among flagged candles)
| Pre-Condition | Active Count | % of Flagged |
| :--- | :---: | :---: |
| Multi-Day Resistance Rejection | 2,213 | 96.0% |
| 4+ Consecutive Bearish Candles | 1,260 | 54.6% |
| Volatility Squeeze (BW < 0.30) | 1,622 | 70.3% |
| Volume Divergence (Decline → Surge) | 66 | 2.9% |
| 3+ Consecutive Lower Highs | 2,139 | 92.8% |

### Top 20 Highest-Conviction Breakdown Setups
| Timestamp | Day | Close | Body (%) | Range (%) | Signals | Reasons | VIX |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| 2025-10-08 11:00 | Wednesday | 25042.65 | -0.183% | 0.207% | **5/5** | ResistReject, Bearish4+, VolSqueeze, VolDivergence, LowerHighs3+ | 10.31 |
| 2025-09-04 14:00 | Thursday | 24765.50 | -0.045% | 0.129% | **5/5** | ResistReject, Bearish4+, VolSqueeze, VolDivergence, LowerHighs3+ | 10.85 |
| 2025-12-30 11:15 | Tuesday | 25914.95 | -0.026% | 0.060% | **5/5** | ResistReject, Bearish4+, VolSqueeze, VolDivergence, LowerHighs3+ | 9.68 |
| 2025-09-29 11:15 | Monday | 24734.15 | -0.028% | 0.042% | **5/5** | ResistReject, Bearish4+, VolSqueeze, VolDivergence, LowerHighs3+ | 11.37 |
| 2020-03-12 14:30 | Thursday | 9572.25 | -1.139% | 1.884% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 41.16 |
| 2021-02-19 14:00 | Friday | 14912.50 | -0.761% | 0.810% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 22.25 |
| 2020-05-05 14:45 | Tuesday | 9232.80 | -0.293% | 0.722% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 43.61 |
| 2022-02-24 15:00 | Thursday | 16273.10 | -0.292% | 0.688% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 31.98 |
| 2020-05-19 13:30 | Tuesday | 8908.00 | -0.377% | 0.687% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 39.45 |
| 2022-07-01 10:00 | Friday | 15555.00 | -0.383% | 0.682% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 21.25 |
| 2020-04-24 14:00 | Friday | 9185.30 | -0.391% | 0.676% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 39.12 |
| 2020-03-17 11:15 | Tuesday | 9236.15 | -0.391% | 0.669% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 62.93 |
| 2020-06-09 14:30 | Tuesday | 10039.90 | -0.480% | 0.640% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 30.21 |
| 2020-06-09 14:00 | Tuesday | 10096.00 | -0.483% | 0.613% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 30.21 |
| 2020-04-17 10:45 | Friday | 9176.40 | -0.068% | 0.596% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 42.59 |
| 2022-05-11 10:15 | Wednesday | 16103.80 | -0.305% | 0.565% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 22.80 |
| 2021-12-20 12:45 | Monday | 16422.20 | -0.471% | 0.548% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 18.97 |
| 2022-06-16 13:15 | Thursday | 15406.80 | -0.397% | 0.532% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 22.87 |
| 2020-04-01 11:00 | Wednesday | 8274.95 | -0.258% | 0.525% | **4/5** | ResistReject, Bearish4+, VolSqueeze, LowerHighs3+ | 60.05 |
| 2026-02-01 15:15 | Sunday | 24768.00 | -0.331% | 0.524% | **4/5** | ResistReject, Bearish4+, VolDivergence, LowerHighs3+ | 15.10 |
