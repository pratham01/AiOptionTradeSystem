"""
OI Zones + Flux OB Backtest — Nifty 50 May 2026
=================================================
Compares three approaches on Nifty 3m historical data:

  A) OI-Based S/R Zones (5 OC-enriched days: May 21-26)
  B) OHLCV-Based S/R Zones (full May 2026, ~30 days)
  C) Fixed Flux Order Blocks (after swing-window correction)

Usage:
    python scripts/backtest_oi_zones.py
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trade_system.application.indicators.oi_zones import OIZoneDetector, OIZone
from trade_system.application.indicators.flux_order_blocks import FluxOrderBlockDetector

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger("oi_backtest")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_3M   = ROOT / "data/fo_historical/NSE_NIFTY50-INDEX_3min_historical.csv"
DATA_DAILY = ROOT / "data/fo_historical/NSE_NIFTY50-INDEX_d_historical.csv"
OC_DIR     = ROOT / "data/option_chain_data"
REPORT_DIR = ROOT / "reports/oi_zones"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_ZONE_PCT  = 0.003   # Price must be within 0.3% of zone to trigger
SL_BUFFER_PTS   = 30      # SL is zone_edge ± 30 pts
RR_RATIO        = 1.5     # Target = risk * 1.5
LOT_SIZE        = 50
STRIKE_STEP     = 50


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_3m_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_3M, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def load_daily_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DAILY, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def load_oc_for_date(oc_date: date) -> pd.DataFrame | None:
    """Load OC CSV snapshot for the given date string (YYYYMMDD)."""
    fname = OC_DIR / f"NIFTY50_strikes_{oc_date.strftime('%Y%m%d')}.csv"
    if not fname.exists():
        return None
    df = pd.read_csv(fname, parse_dates=["timestamp"])
    # Use end-of-day (last) snapshot for zone computation
    if "timestamp" in df.columns:
        latest = df["timestamp"].max()
        df = df[df["timestamp"] == latest]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Zone Builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_ohlcv_zones(prev_day: pd.Series, spot: float) -> list[OIZone]:
    """Build S/R zones purely from previous day OHLCV (fallback / full-period)."""
    tolerance = spot * 0.003   # 0.3% band
    zones: list[OIZone] = []
    detector = OIZoneDetector(tolerance_pct=0.003)

    def _zone(strike, ztype, source, is_res, strength):
        z = OIZone(
            strike=float(strike), zone_type=ztype, source=source,
            strength=strength, is_resistance=is_res, _tolerance=tolerance,
        )
        z.__post_init__()
        return z

    high  = float(prev_day["high"])
    low   = float(prev_day["low"])
    close = float(prev_day["close"])

    zones.append(_zone(high,  "RESISTANCE", "PREV_HIGH",  True,  0.70))
    zones.append(_zone(low,   "SUPPORT",    "PREV_LOW",   False, 0.70))
    zones.append(_zone(close, "PREV_CLOSE", "PREV_CLOSE", close >= spot, 0.55))

    # Mid-zone (pivot)
    pivot = (high + low + close) / 3
    zones.append(_zone(pivot, "PREV_CLOSE", "PIVOT", pivot >= spot, 0.50))

    return zones


def build_oi_zones(oc_df: pd.DataFrame, prev_day: pd.Series, spot: float) -> list[OIZone]:
    """Build OI-enriched zones from previous day OC + OHLCV."""
    detector = OIZoneDetector(
        num_top_strikes=5,
        num_volume_clusters=3,
        tolerance_pct=0.002,
        min_oi_threshold=1_000_000,
    )
    return detector.compute_zones(
        oc_df=oc_df,
        prev_day_high=float(prev_day["high"]),
        prev_day_low=float(prev_day["low"]),
        prev_day_close=float(prev_day["close"]),
        spot_price=spot,
        lot_size=LOT_SIZE,
        strike_step=STRIKE_STEP,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Signal Generator
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_zone_trades(
    day_bars: pd.DataFrame,
    zones: list[OIZone],
    label: str,
) -> list[dict]:
    """
    Simulate trades for a single day using the provided zones.

    Rules:
    - Price enters zone band → wait for one confirming bar (close back inside or momentum)
    - At SUPPORT: go LONG if bar closes above zone, SL = zone_bottom - SL_BUFFER
    - At RESISTANCE: go SHORT if bar closes below zone, SL = zone_top + SL_BUFFER
    - TP = entry ± risk * RR_RATIO
    - Only 1 trade per zone per day; close all at 15:15
    """
    trades = []
    if not zones or day_bars.empty:
        return trades

    triggered_zones: set[float] = set()  # avoid multiple entries per zone
    open_trade: dict | None = None

    for i in range(1, len(day_bars)):
        bar = day_bars.iloc[i]
        price = float(bar["close"])
        bar_time: datetime = bar["timestamp"]

        # --- Manage open trade ---
        if open_trade is not None:
            direction = open_trade["direction"]
            sl = open_trade["sl"]
            tp = open_trade["tp"]
            entry = open_trade["entry"]

            hit_sl = (direction == 1 and float(bar["low"]) <= sl) or \
                     (direction == -1 and float(bar["high"]) >= sl)
            hit_tp = (direction == 1 and float(bar["high"]) >= tp) or \
                     (direction == -1 and float(bar["low"]) <= tp)
            eod    = bar_time.hour >= 15 and bar_time.minute >= 15

            if hit_tp:
                pnl_pts = abs(tp - entry)
                trades.append({**open_trade, "exit_price": tp, "exit_time": bar_time,
                               "exit_reason": "TP", "pnl_pts": pnl_pts * direction,
                               "label": label})
                open_trade = None
            elif hit_sl:
                pnl_pts = abs(sl - entry)
                trades.append({**open_trade, "exit_price": sl, "exit_time": bar_time,
                               "exit_reason": "SL", "pnl_pts": -pnl_pts,
                               "label": label})
                open_trade = None
            elif eod:
                pnl_pts = (price - entry) * direction
                trades.append({**open_trade, "exit_price": price, "exit_time": bar_time,
                               "exit_reason": "EOD", "pnl_pts": pnl_pts,
                               "label": label})
                open_trade = None
            continue  # only 1 active trade at a time

        # No entry after 14:45
        if bar_time.hour >= 14 and bar_time.minute >= 45:
            break

        # --- Check zones ---
        for zone in zones:
            if zone.strike in triggered_zones:
                continue
            if not zone.contains(price):
                continue

            prev_bar = day_bars.iloc[i - 1]
            prev_close = float(prev_bar["close"])

            if not zone.is_resistance:
                # SUPPORT zone: confirm bullish candle (close > open)
                if float(bar["close"]) > float(bar["open"]) and float(bar["close"]) > prev_close:
                    entry = price
                    sl = zone.band_bottom - SL_BUFFER_PTS
                    risk = entry - sl
                    tp = entry + risk * RR_RATIO
                    open_trade = {
                        "direction": 1, "entry": entry, "sl": sl, "tp": tp,
                        "entry_time": bar_time, "zone_type": zone.zone_type,
                        "zone_strike": zone.strike, "zone_source": zone.source,
                        "zone_strength": zone.strength,
                    }
                    triggered_zones.add(zone.strike)
                    break

            else:
                # RESISTANCE zone: confirm bearish candle (close < open)
                if float(bar["close"]) < float(bar["open"]) and float(bar["close"]) < prev_close:
                    entry = price
                    sl = zone.band_top + SL_BUFFER_PTS
                    risk = sl - entry
                    tp = entry - risk * RR_RATIO
                    open_trade = {
                        "direction": -1, "entry": entry, "sl": sl, "tp": tp,
                        "entry_time": bar_time, "zone_type": zone.zone_type,
                        "zone_strike": zone.strike, "zone_source": zone.source,
                        "zone_strength": zone.strength,
                    }
                    triggered_zones.add(zone.strike)
                    break

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  Flux OB Backtest
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_flux_ob_trades(day_bars: pd.DataFrame, full_history: pd.DataFrame) -> list[dict]:
    """
    Run Flux OB (fixed) on the full history up to the current day,
    then trade active zones on today's bars.
    """
    trades = []
    if day_bars.empty or full_history.empty:
        return trades

    today_date = day_bars["date"].iloc[0]
    # Use all bars up to (not including) today
    hist_bars = full_history[full_history["date"] < today_date].copy()
    if len(hist_bars) < 30:
        return trades  # not enough history

    # Add timestamp column required by FluxOrderBlockDetector
    hist_bars = hist_bars.reset_index(drop=True)
    if "timestamp" not in hist_bars.columns:
        hist_bars["timestamp"] = pd.to_datetime(hist_bars.index, unit="s")

    detector = FluxOrderBlockDetector(swing_length=10, max_atr_mult=3.5, num_render_blocks=5)
    ob_history = detector.calculate(hist_bars)
    active_obs = ob_history[-1] if ob_history else []

    if not active_obs:
        return trades

    open_trade: dict | None = None
    triggered_obs: set[int] = set()

    for i in range(1, len(day_bars)):
        bar = day_bars.iloc[i]
        price = float(bar["close"])
        bar_time: datetime = bar["timestamp"]

        if open_trade is not None:
            direction = open_trade["direction"]
            sl = open_trade["sl"]
            tp = open_trade["tp"]
            entry = open_trade["entry"]
            hit_sl = (direction == 1 and float(bar["low"]) <= sl) or \
                     (direction == -1 and float(bar["high"]) >= sl)
            hit_tp = (direction == 1 and float(bar["high"]) >= tp) or \
                     (direction == -1 and float(bar["low"]) <= tp)
            eod = bar_time.hour >= 15 and bar_time.minute >= 15

            if hit_tp:
                pnl_pts = abs(tp - entry)
                trades.append({**open_trade, "exit_price": tp, "exit_time": bar_time,
                               "exit_reason": "TP", "pnl_pts": pnl_pts * direction, "label": "FluxOB_Fixed"})
                open_trade = None
            elif hit_sl:
                pnl_pts = abs(sl - entry)
                trades.append({**open_trade, "exit_price": sl, "exit_time": bar_time,
                               "exit_reason": "SL", "pnl_pts": -pnl_pts, "label": "FluxOB_Fixed"})
                open_trade = None
            elif eod:
                pnl_pts = (price - entry) * direction
                trades.append({**open_trade, "exit_price": price, "exit_time": bar_time,
                               "exit_reason": "EOD", "pnl_pts": pnl_pts, "label": "FluxOB_Fixed"})
                open_trade = None
            continue

        if bar_time.hour >= 14 and bar_time.minute >= 45:
            break

        for ob_i, ob in enumerate(active_obs):
            if ob_i in triggered_obs:
                continue
            if ob.bottom <= price <= ob.top:
                prev_close = float(day_bars.iloc[i - 1]["close"])
                if ob.ob_type == "Bull" and float(bar["close"]) > float(bar["open"]):
                    # Demand zone → Long
                    entry = price
                    sl = ob.bottom - SL_BUFFER_PTS
                    risk = entry - sl
                    tp = entry + risk * RR_RATIO
                    open_trade = {
                        "direction": 1, "entry": entry, "sl": sl, "tp": tp,
                        "entry_time": bar_time, "zone_type": "Bull_OB",
                        "zone_strike": (ob.top + ob.bottom) / 2,
                        "zone_source": f"OB@{ob.start_time}",
                        "zone_strength": 0.8,
                    }
                    triggered_obs.add(ob_i)
                    break
                elif ob.ob_type == "Bear" and float(bar["close"]) < float(bar["open"]):
                    # Supply zone → Short
                    entry = price
                    sl = ob.top + SL_BUFFER_PTS
                    risk = sl - entry
                    tp = entry - risk * RR_RATIO
                    open_trade = {
                        "direction": -1, "entry": entry, "sl": sl, "tp": tp,
                        "entry_time": bar_time, "zone_type": "Bear_OB",
                        "zone_strike": (ob.top + ob.bottom) / 2,
                        "zone_source": f"OB@{ob.start_time}",
                        "zone_strength": 0.8,
                    }
                    triggered_obs.add(ob_i)
                    break

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def stats(trades: list[dict], label: str) -> dict:
    base = {
        "label": label, "trades": 0, "win_rate": 0.0,
        "total_pnl_pts": 0.0, "avg_pnl_pts": 0.0,
        "profit_factor": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
        "by_exit": {},
    }
    if not trades:
        return base
    df = pd.DataFrame(trades)
    wins   = df[df["pnl_pts"] > 0]
    losses = df[df["pnl_pts"] <= 0]
    gross_profit = wins["pnl_pts"].sum()
    gross_loss   = abs(losses["pnl_pts"].sum())
    return {
        "label":          label,
        "trades":         len(df),
        "win_rate":       round(len(wins) / len(df) * 100, 1),
        "total_pnl_pts":  round(df["pnl_pts"].sum(), 1),
        "avg_pnl_pts":    round(df["pnl_pts"].mean(), 1),
        "profit_factor":  round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0,
        "best_trade":     round(df["pnl_pts"].max(), 1),
        "worst_trade":    round(df["pnl_pts"].min(), 1),
        "by_exit":        df.groupby("exit_reason")["pnl_pts"].count().to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest() -> None:
    print("\n" + "═" * 70)
    print("   OI ZONES + FLUX OB BACKTEST — NIFTY 50 MAY 2026")
    print("═" * 70)

    df3m   = load_3m_data()
    daily  = load_daily_data()

    all_days = sorted(df3m["date"].unique())
    may_days = [d for d in all_days if d.month == 5 and d.year == 2026]

    ohlcv_trades: list[dict] = []
    oi_trades:    list[dict] = []
    flux_trades:  list[dict] = []

    zone_hit_log: list[dict] = []

    for today in may_days:
        day_bars = df3m[df3m["date"] == today].copy()
        if day_bars.empty or len(day_bars) < 5:
            continue

        today_open = float(day_bars.iloc[0]["open"])

        # Previous trading day
        prev_days_in_data = [d for d in sorted(daily["date"].unique()) if d < today]
        if not prev_days_in_data:
            continue
        prev_date = prev_days_in_data[-1]
        prev_row  = daily[daily["date"] == prev_date].iloc[0]

        # ── A) OHLCV-based zones (every day) ──────────────────────────────────
        ohlcv_zones = build_ohlcv_zones(prev_row, today_open)
        day_ohlcv_trades = simulate_zone_trades(day_bars, ohlcv_zones, "OHLCV_Zones")
        ohlcv_trades.extend(day_ohlcv_trades)

        # ── B) OI-enriched zones (only when we have prev-day OC data) ─────────
        day_oi_trades: list[dict] = []   # always initialise
        prev_oc = load_oc_for_date(prev_date)
        if prev_oc is not None and not prev_oc.empty:
            oi_zones = build_oi_zones(prev_oc, prev_row, today_open)
            day_oi_trades = simulate_zone_trades(day_bars, oi_zones, "OI_Zones")
            oi_trades.extend(day_oi_trades)

            # Log all zone touches (zone hit rate analysis)
            for bar_i, bar in day_bars.iterrows():
                price = float(bar["close"])
                for z in oi_zones:
                    if z.contains(price):
                        zone_hit_log.append({
                            "date": today,
                            "time": bar["timestamp"],
                            "zone_type": z.zone_type,
                            "strike": z.strike,
                            "source": z.source,
                            "strength": z.strength,
                            "price": price,
                        })
                        break

        # ── C) Flux OB (fixed) ─────────────────────────────────────────────────
        flux_day_trades = simulate_flux_ob_trades(day_bars, df3m)
        flux_trades.extend(flux_day_trades)

        if day_ohlcv_trades or day_oi_trades or flux_day_trades:
            print(f"  {today}: OHLCV={len(day_ohlcv_trades)} OI={len(day_oi_trades)} FluxOB={len(flux_day_trades)}")

    # ── Compile Results ────────────────────────────────────────────────────────
    stat_a = stats(ohlcv_trades, "OHLCV S/R Zones (30 days)")
    stat_b = stats(oi_trades,    "OI-Enriched Zones (5 days w/ OC)")
    stat_c = stats(flux_trades,  "Fixed Flux OB (30 days)")

    # ── Zone Hit Rate ──────────────────────────────────────────────────────────
    n_oi_bars = len(df3m[(df3m["date"].apply(lambda d: d.month == 5 and d.year == 2026))])
    zone_hit_rate = len(zone_hit_log) / max(n_oi_bars, 1) * 100

    # ── Print Report ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STRATEGY COMPARISON SUMMARY")
    print("─" * 70)

    header = f"{'Metric':<30} {'OHLCV Zones':>16} {'OI Zones':>16} {'Flux OB Fixed':>16}"
    print(header)
    print("─" * 70)

    def row(label, a, b, c):
        return f"  {label:<28} {str(a):>16} {str(b):>16} {str(c):>16}"

    print(row("Total Trades",       stat_a["trades"],          stat_b["trades"],          stat_c["trades"]))
    print(row("Win Rate (%)",        stat_a["win_rate"],        stat_b["win_rate"],        stat_c["win_rate"]))
    print(row("Total PnL (pts)",     stat_a["total_pnl_pts"],   stat_b["total_pnl_pts"],   stat_c["total_pnl_pts"]))
    print(row("Avg PnL / Trade",     stat_a["avg_pnl_pts"],     stat_b["avg_pnl_pts"],     stat_c["avg_pnl_pts"]))
    print(row("Profit Factor",       stat_a["profit_factor"],   stat_b["profit_factor"],   stat_c["profit_factor"]))
    print(row("Best Trade (pts)",    stat_a["best_trade"],      stat_b["best_trade"],      stat_c["best_trade"]))
    print(row("Worst Trade (pts)",   stat_a["worst_trade"],     stat_b["worst_trade"],     stat_c["worst_trade"]))
    print(f"\n  Zone Hit Rate (OI zones): {zone_hit_rate:.1f}% of bars touched a zone")

    print("\n  EXIT REASON BREAKDOWN")
    print(f"  OHLCV Zones: {stat_a.get('by_exit', {})}")
    print(f"  OI Zones:    {stat_b.get('by_exit', {})}")
    print(f"  Flux OB:     {stat_c.get('by_exit', {})}")

    # ── Today's Zones (May 26) ─────────────────────────────────────────────────
    today_oc = load_oc_for_date(date(2026, 5, 26))
    if today_oc is not None:
        prev_for_today = daily[daily["date"] < date(2026, 5, 26)].iloc[-1]
        current_spot = float(df3m[df3m["date"] == date(2026, 5, 26)]["close"].iloc[-1]) if date(2026, 5, 26) in df3m["date"].values else 24500.0
        today_zones = build_oi_zones(today_oc, prev_for_today, current_spot)

        print("\n" + "─" * 70)
        print(f"  TODAY'S OI ZONES (May 26, spot ≈ {current_spot:.0f})")
        print("─" * 70)
        print(f"  {'Type':<16} {'Strike':>8}  {'Source':<22} {'Strength':>10}")
        for z in today_zones[:15]:
            direction_tag = "R" if z.is_resistance else "S"
            marker = " ◀ KEY" if z.strength >= 0.7 else ""
            print(f"  [{direction_tag}] {z.zone_type:<14} {z.strike:>8.0f}  {z.source:<22} {z.strength:>10.3f}{marker}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    all_trades = ohlcv_trades + oi_trades + flux_trades
    if all_trades:
        out_df = pd.DataFrame(all_trades)
        out_path = REPORT_DIR / "backtest_trades.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\n  📄 Trade log saved → {out_path}")

    if zone_hit_log:
        hit_df = pd.DataFrame(zone_hit_log)
        hit_path = REPORT_DIR / "zone_hits.csv"
        hit_df.to_csv(hit_path, index=False)
        print(f"  📄 Zone hit log   → {hit_path}")

    # ── Markdown Report ───────────────────────────────────────────────────────
    write_markdown_report(stat_a, stat_b, stat_c, zone_hit_rate,
                          today_zones if (today_oc is not None and not today_oc.empty) else [])
    print("\n" + "═" * 70)


def write_markdown_report(stat_a, stat_b, stat_c, zone_hit_rate, today_zones):
    path = REPORT_DIR / "nifty_may_2026_oi_zones_report.md"
    with open(path, "w") as f:
        f.write("# Nifty 50 — OI Zone Backtest Report (May 2026)\n\n")
        f.write(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n")
        f.write("## Strategy Comparison\n\n")
        f.write("| Metric | OHLCV Zones (30d) | OI Zones (5d) | Flux OB Fixed (30d) |\n")
        f.write("|--------|-------------------|---------------|---------------------|\n")
        for key in ["trades", "win_rate", "total_pnl_pts", "avg_pnl_pts", "profit_factor", "best_trade", "worst_trade"]:
            f.write(f"| {key.replace('_',' ').title()} | {stat_a.get(key,'—')} | {stat_b.get(key,'—')} | {stat_c.get(key,'—')} |\n")
        f.write(f"\n**Zone Hit Rate** (OI): {zone_hit_rate:.1f}% of bars touched a zone\n\n")

        if today_zones:
            f.write("## Today's Key OI Zones (May 26, 2026)\n\n")
            f.write("| Dir | Type | Strike | Source | Strength |\n")
            f.write("|-----|------|--------|--------|----------|\n")
            for z in today_zones[:15]:
                d = "R" if z.is_resistance else "S"
                f.write(f"| {d} | {z.zone_type} | {z.strike:.0f} | {z.source} | {z.strength:.3f} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("""
- **OHLCV Zones**: Uses only previous-day High/Low/Close as S/R. Simple and available every day.
- **OI Zones**: Enriches with max CE/PE OI strikes, volume clusters, Max Pain, and GEX walls.
  These are *institutional* levels that options market-makers must defend.
- **Flux OB (Fixed)**: Uses the corrected swing-window logic (strictly historical bars) and
  correct OB origin (last bullish/bearish candle before impulse, not highest/lowest).

### Key Insights
- OI walls (max CE/PE OI) have historically acted as strong S/R on expiry weeks
- Max Pain is the most reliable level on expiry day (Thursday)
- Gamma Walls act as mean-reversion magnets in low-IV environments
""")

    print(f"  📄 Markdown report → {path}")


if __name__ == "__main__":
    run_backtest()
