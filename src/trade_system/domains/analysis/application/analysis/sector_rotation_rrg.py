"""
Sector Rotation Relative Rotation Graph (RRG) Engine.

Modeled after Julius de Kempenaer's Relative Rotation Graph (RRG) framework,
as popularized on institutional trading platforms and modern market analytics
(e.g., StockMojo Sector Rotation RRG).

Key Metrics:
- RS-Ratio (X-Axis): Measures the relative strength of a sector vs a benchmark (e.g. Nifty 50).
  Values > 100 indicate outperformance; < 100 indicate underperformance.
- RS-Momentum (Y-Axis): Measures the rate of change / momentum of the RS-Ratio.
  Values > 100 indicate accelerating relative strength; < 100 indicate decelerating.

Quadrants:
- 🟢 LEADING (RS-Ratio >= 100, RS-Momentum >= 100): High relative strength & positive momentum.
- 🟡 WEAKENING (RS-Ratio >= 100, RS-Momentum < 100): Outperforming, but relative momentum is slowing.
- 🔴 LAGGING (RS-Ratio < 100, RS-Momentum < 100): Low relative strength & negative momentum.
- 🔵 IMPROVING (RS-Ratio < 100, RS-Momentum >= 100): Underperforming, but relative momentum is turning up.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class SectorRRGPoint:
    sector: str
    rs_ratio: float
    rs_momentum: float
    quadrant: str  # "LEADING", "WEAKENING", "LAGGING", "IMPROVING"
    sector_return: float
    benchmark_return: float
    advances: int
    declines: int
    unchanged: int
    total_stocks: int
    advance_pct: float
    avg_vol_surge: float
    history_tail: List[Tuple[float, float]] = field(default_factory=list)  # [(rs_ratio, rs_momentum), ...]


class SectorRRGEngine:
    """Computes intraday Sector Rotation RRG and Market Advance/Decline Breadth."""

    QUADRANT_CONFIG = {
        "LEADING": {
            "label": "Leading",
            "emoji": "🟢",
            "badge_color": "#22c55e",
            "bg_color": "rgba(34, 197, 94, 0.12)",
            "description": "Outperforming benchmark with accelerating momentum"
        },
        "WEAKENING": {
            "label": "Weakening",
            "emoji": "🟡",
            "badge_color": "#eab308",
            "bg_color": "rgba(234, 179, 8, 0.12)",
            "description": "Outperforming benchmark, but relative momentum is slowing"
        },
        "LAGGING": {
            "label": "Lagging",
            "emoji": "🔴",
            "badge_color": "#ef4444",
            "bg_color": "rgba(239, 68, 68, 0.12)",
            "description": "Underperforming benchmark with negative momentum"
        },
        "IMPROVING": {
            "label": "Improving",
            "emoji": "🔵",
            "badge_color": "#3b82f6",
            "bg_color": "rgba(59, 130, 246, 0.12)",
            "description": "Underperforming benchmark, but relative momentum is turning up"
        }
    }

    @classmethod
    def compute_sector_breadth(cls, merged_closes: pd.DataFrame) -> Dict[str, dict]:
        """Calculates advances, declines, neutral stocks, and average volume surge per sector."""
        breadth_by_sector = {}
        if merged_closes.empty:
            return breadth_by_sector

        for sec, grp in merged_closes.groupby('sector'):
            if sec == "UNKNOWN":
                continue
            adv = int((grp['pChange'] > 0).sum())
            dec = int((grp['pChange'] < 0).sum())
            unc = int((grp['pChange'] == 0).sum())
            tot = len(grp)
            adv_pct = round((adv / tot * 100), 1) if tot > 0 else 0.0
            avg_vs = round(float(grp['vol_surge'].mean()), 2) if 'vol_surge' in grp.columns else 1.0

            breadth_by_sector[sec] = {
                "advances": adv,
                "declines": dec,
                "unchanged": unc,
                "total_stocks": tot,
                "advance_pct": adv_pct,
                "avg_vol_surge": avg_vs
            }
        return breadth_by_sector

    @classmethod
    def compute_rrg(
        cls,
        today_15m_df: pd.DataFrame,
        merged_closes: pd.DataFrame,
        benchmark_symbol: str = "NSE:NIFTY50-INDEX",
        tail_bars: int = 5
    ) -> Dict[str, SectorRRGPoint]:
        """
        Computes Relative Rotation Graph (RRG) points and trajectory tails for all sectors.
        
        Args:
            today_15m_df: Intraday 15-minute candles for the target date.
            merged_closes: Pre-aggregated stock dataframe containing 'symbol', 'sector', 'pChange', 'vol_surge'.
            benchmark_symbol: Benchmark symbol (defaults to Nifty 50 Index).
            tail_bars: Number of recent historical bars to calculate trajectory trails for.
        """
        breadth_map = cls.compute_sector_breadth(merged_closes)
        results: Dict[str, SectorRRGPoint] = {}

        # 0. Benchmark Return from merged_closes as baseline
        bench_ret_baseline = 0.0
        if not merged_closes.empty:
            bench_match = merged_closes[merged_closes['symbol'] == benchmark_symbol]
            if not bench_match.empty and 'pChange' in bench_match.columns:
                bench_ret_baseline = round(float(bench_match.iloc[0]['pChange']), 2)
            elif 'pChange' in merged_closes.columns:
                bench_ret_baseline = round(float(merged_closes['pChange'].mean()), 2)

        # Helper for snapshot computation
        def _build_snapshot_point(s: str, bdata: dict) -> SectorRRGPoint:
            sec_stocks = merged_closes[merged_closes['sector'] == s] if not merged_closes.empty else pd.DataFrame()
            sec_ret = round(float(sec_stocks['pChange'].mean()), 2) if not sec_stocks.empty and 'pChange' in sec_stocks.columns else 0.0
            rs_diff = sec_ret - bench_ret_baseline
            rs_ratio = round(100.0 + rs_diff * 2.0, 2)
            rs_mom = round(100.0 + rs_diff * 1.5, 2)
            quad = cls._classify_quadrant(rs_ratio, rs_mom)
            # 3-bar rotational trajectory tail
            tail = [
                (round(rs_ratio - rs_diff * 0.4, 2), round(rs_mom - 0.5, 2)),
                (round(rs_ratio - rs_diff * 0.2, 2), round(rs_mom - 0.2, 2)),
                (rs_ratio, rs_mom)
            ]
            return SectorRRGPoint(
                sector=s,
                rs_ratio=rs_ratio,
                rs_momentum=rs_mom,
                quadrant=quad,
                sector_return=sec_ret,
                benchmark_return=bench_ret_baseline,
                advances=bdata.get("advances", 0),
                declines=bdata.get("declines", 0),
                unchanged=bdata.get("unchanged", 0),
                total_stocks=bdata.get("total_stocks", 0),
                advance_pct=bdata.get("advance_pct", 0.0),
                avg_vol_surge=bdata.get("avg_vol_surge", 1.0),
                history_tail=tail
            )

        # Fallback to static snapshot if intraday candles are missing
        if today_15m_df.empty:
            for sec, bdata in breadth_map.items():
                results[sec] = _build_snapshot_point(sec, bdata)
            return results

        # Ensure today_15m_df has 'sector' column
        today_df = today_15m_df.copy()
        if 'sector' not in today_df.columns and not merged_closes.empty and 'sector' in merged_closes.columns:
            sym_sec = dict(zip(merged_closes['symbol'], merged_closes['sector']))
            today_df['sector'] = today_df['symbol'].map(sym_sec).fillna("UNKNOWN")

        # 1. Resolve Benchmark Intraday Return Series
        today_sorted = today_df.sort_values('timestamp')
        bench_df = today_sorted[today_sorted['symbol'].str.contains("NIFTY50|NIFTY 50", case=False)]

        if not bench_df.empty:
            bench_df = bench_df.sort_values('timestamp')
            bench_base = float(bench_df.iloc[0]['open'])
            bench_series = ((bench_df.set_index('timestamp')['close'] - bench_base) / bench_base * 100) if bench_base > 0 else pd.Series(dtype=float)
        else:
            # Composite market return as fallback benchmark
            bench_piv = today_sorted.groupby('timestamp')['close'].mean()
            bench_base = float(today_sorted.groupby('timestamp')['open'].mean().iloc[0]) if not today_sorted.empty else 0.0
            bench_series = ((bench_piv - bench_base) / bench_base * 100) if bench_base > 0 else pd.Series(dtype=float)

        timestamps = sorted(bench_series.index.unique())
        if len(timestamps) < 2:
            # Insufficient intraday candles for full time-series: use robust snapshot
            for sec, bdata in breadth_map.items():
                results[sec] = _build_snapshot_point(sec, bdata)
            return results

        sectors = [s for s in today_sorted['sector'].unique() if s and s != "UNKNOWN"]

        # 2. Compute Return Series & RRG for each sector
        for sec in sectors:
            bdata = breadth_map.get(sec, {
                "advances": 0, "declines": 0, "unchanged": 0,
                "total_stocks": 0, "advance_pct": 0.0, "avg_vol_surge": 1.0
            })
            sec_df = today_sorted[today_sorted['sector'] == sec]
            if sec_df.empty:
                results[sec] = _build_snapshot_point(sec, bdata)
                continue

            sec_close_piv = sec_df.groupby('timestamp')['close'].mean()
            sec_open_piv = sec_df.groupby('timestamp')['open'].mean()
            if sec_open_piv.empty or sec_close_piv.empty:
                results[sec] = _build_snapshot_point(sec, bdata)
                continue

            sec_base = float(sec_open_piv.iloc[0])
            if sec_base <= 0:
                results[sec] = _build_snapshot_point(sec, bdata)
                continue

            sec_ret_series = (sec_close_piv - sec_base) / sec_base * 100

            # Align timestamps with benchmark
            aligned = pd.DataFrame({'sector': sec_ret_series, 'bench': bench_series}).dropna()
            if aligned.empty:
                results[sec] = _build_snapshot_point(sec, bdata)
                continue

            # RS Difference = Sector Return - Benchmark Return
            rs_diff = aligned['sector'] - aligned['bench']

            # JdK normalized RS-Ratio around 100
            rs_ratio_series = 100.0 + rs_diff * 2.0

            # RS-Momentum: Rate of change of RS-Ratio relative to 2 bars ago (30-minute change)
            shift_bars = 2 if len(rs_ratio_series) >= 3 else 1
            rs_mom_series = 100.0 + (rs_ratio_series - rs_ratio_series.shift(shift_bars)).fillna(0.0) * 1.5

            latest_ratio = round(float(rs_ratio_series.iloc[-1]), 2)
            latest_mom = round(float(rs_mom_series.iloc[-1]), 2)
            quadrant = cls._classify_quadrant(latest_ratio, latest_mom)

            latest_sec_ret = round(float(aligned['sector'].iloc[-1]), 2)
            latest_bench_ret = round(float(aligned['bench'].iloc[-1]), 2)

            # Extract trajectory tail (last N bars)
            tail_points = []
            tail_indices = rs_ratio_series.tail(tail_bars).index
            for ts in tail_indices:
                r_val = round(float(rs_ratio_series.loc[ts]), 2)
                m_val = round(float(rs_mom_series.loc[ts]), 2)
                tail_points.append((r_val, m_val))

            if len(tail_points) == 1:
                # Add synthetic trailing point for visual directional line
                tail_points.insert(0, (round(latest_ratio - (latest_ratio - 100) * 0.2, 2), round(latest_mom - 0.3, 2)))

            results[sec] = SectorRRGPoint(
                sector=sec,
                rs_ratio=latest_ratio,
                rs_momentum=latest_mom,
                quadrant=quadrant,
                sector_return=latest_sec_ret,
                benchmark_return=latest_bench_ret,
                advances=bdata["advances"],
                declines=bdata["declines"],
                unchanged=bdata["unchanged"],
                total_stocks=bdata["total_stocks"],
                advance_pct=bdata["advance_pct"],
                avg_vol_surge=bdata["avg_vol_surge"],
                history_tail=tail_points
            )

        # Make sure any sector present in breadth_map is in results
        for sec, bdata in breadth_map.items():
            if sec not in results:
                results[sec] = _build_snapshot_point(sec, bdata)

        return results

    @staticmethod
    def _classify_quadrant(rs_ratio: float, rs_momentum: float) -> str:
        """Assigns the RRG Quadrant based on RS-Ratio and RS-Momentum values."""
        if rs_ratio >= 100.0:
            return "LEADING" if rs_momentum >= 100.0 else "WEAKENING"
        else:
            return "IMPROVING" if rs_momentum >= 100.0 else "LAGGING"
