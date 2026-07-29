from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class Recommendation:
    symbol: str
    score: float
    rationale: str
    spot: float
    sector: str = "Unknown"


def _safe_pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return (num / den) * 100.0


def load_sector_map() -> dict[str, str]:
    """Load stock-to-sector mapping from config."""
    path = Path("config/fo_universe.json")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            LOGGER.error(f"Failed to load sector map: {e}")
    return {}


def load_latest_sector_performance() -> dict[str, float]:
    """Load latest sector average performance from DB fallback."""
    try:
        from trade_system.domains.market_data.infrastructure.database.connection import get_engine
        from sqlalchemy import text
        import pandas as pd
        
        engine = get_engine()
        config_path = Path("config/fo_universe.json")
        if not config_path.exists():
            return {}
            
        with open(config_path, "r") as f:
            fo_metadata = json.load(f)
            
        query = text("""
            SELECT symbol, close 
            FROM ohlcv_15m 
            WHERE timestamp >= date('now', '-3 days')
            ORDER BY timestamp DESC
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            return {}
            
        latest_closes = df.groupby('symbol').first().reset_index()
        sectors = []
        for sym in latest_closes['symbol']:
            sectors.append(fo_metadata.get(sym, "UNKNOWN"))
        latest_closes['sector'] = sectors
        
        grouped = df.groupby('symbol')
        changes = []
        for sym, group in grouped:
            if len(group) > 5:
                last = group.iloc[0]['close']
                prev = group.iloc[-1]['close']
                pct = ((last - prev) / prev) * 100
            else:
                pct = 0
            changes.append({'symbol': sym, 'pChange': pct})
            
        changes_df = pd.DataFrame(changes)
        merged = pd.merge(latest_closes, changes_df, on='symbol', suffixes=('_dummy', ''))
        
        sector_df = merged.groupby('sector').agg({'pChange': 'mean'}).reset_index()
        
        perf = {}
        for _, row in sector_df.iterrows():
            if row['sector'] != 'UNKNOWN':
                perf[row['sector']] = float(row['pChange'])
        return perf
    except Exception as e:
        LOGGER.error(f"Failed to load sector performance from DB: {e}")
    return {}


def score_stock_candidate(
    quote: Any, 
    vix: float | None, 
    pcr: float | None,
    sector_perf: float | None = None
) -> tuple[float, str]:
    # Price action
    intraday_breakout = _safe_pct(quote.last_price - quote.open, quote.open)
    range_expansion = _safe_pct(quote.high - quote.low, quote.open)
    momentum = float(quote.change_percent)

    # Option-buyer favorability proxy
    vix_score = 12.0
    if vix is not None:
        if 12 <= vix <= 18:
            vix_score = 15.0
        elif vix > 22:
            vix_score = 6.0

    pcr_score = 10.0
    if pcr is not None:
        if 0.9 <= pcr <= 1.2:
            pcr_score = 14.0
        elif pcr < 0.7 or pcr > 1.5:
            pcr_score = 6.0

    # Sector strength score (-10 to 15)
    sector_score = 0.0
    if sector_perf is not None:
        sector_score = max(min(sector_perf * 5.0, 15.0), -10.0)

    trend_score = max(min(intraday_breakout * 4, 20), -10)
    momentum_score = max(min(momentum * 3.5, 20), -10)
    volume_score = 15.0 if quote.volume >= 500_000 else 8.0
    expansion_score = 12.0 if range_expansion >= 1.2 else 6.0

    raw = 30 + trend_score + momentum_score + volume_score + expansion_score + vix_score + pcr_score + sector_score
    final = max(0.0, min(raw, 100.0))

    rationale = (
        f"Strong momentum ({momentum:.2f}%) with intraday breakout ({intraday_breakout:.2f}%). "
        f"Range expanded by {range_expansion:.2f}% with volume of {quote.volume}. "
        f"Sector Performance: {sector_perf:.2f}%."
        if sector_perf is not None else
        f"Strong momentum ({momentum:.2f}%) with intraday breakout ({intraday_breakout:.2f}%). "
        f"Range expanded by {range_expansion:.2f}% with volume of {quote.volume}."
    )
    return round(final, 2), rationale


def build_recommendations(
    quotes: dict[str, Any],
    top_n: int = 5,
    vix: float | None = None,
    pcr: float | None = None,
) -> list[Recommendation]:
    sector_map = load_sector_map()
    sector_perf_map = load_latest_sector_performance()
    
    ranked: list[Recommendation] = []
    for symbol, quote in quotes.items():
        sector = sector_map.get(symbol, "Unknown")
        sec_perf = sector_perf_map.get(sector)
        
        score, rationale = score_stock_candidate(quote, vix=vix, pcr=pcr, sector_perf=sec_perf)
        ranked.append(
            Recommendation(
                symbol=symbol, 
                score=score, 
                rationale=rationale, 
                spot=float(quote.last_price),
                sector=sector
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_n]


def validate_recommendations(
    recommendation_date: str,
    recommendations: list[Recommendation],
    latest_quotes: dict[str, Any],
    benchmark_change_pct: float = 0.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hits = 0
    total_change = 0.0

    for rec in recommendations:
        q = latest_quotes.get(rec.symbol)
        if not q:
            continue
        change_pct = _safe_pct(float(q.last_price) - rec.spot, rec.spot)
        outperform = change_pct > benchmark_change_pct
        if outperform:
            hits += 1
        total_change += change_pct
        rows.append(
            {
                "symbol": rec.symbol,
                "sector": rec.sector,
                "entry_spot": rec.spot,
                "eod_spot": float(q.last_price),
                "change_pct": round(change_pct, 2),
                "outperform_benchmark": outperform,
                "score": rec.score,
                "rationale": rec.rationale,
            }
        )

    count = len(rows)
    hit_rate = round((hits / count) * 100.0, 2) if count else 0.0
    avg_change = round(total_change / count, 2) if count else 0.0
    
    # EOD Analysis Synthesis
    analysis = f"Analysis for {recommendation_date}: "
    if hit_rate >= 60:
        analysis += f"High conviction picks performed well with {hit_rate}% hit rate. "
    else:
        analysis += f"Mixed performance with {hit_rate}% hit rate. "
    
    analysis += f"Average return: {avg_change:+.2f}% vs Benchmark: {benchmark_change_pct:+.2f}%."

    return {
        "date": recommendation_date,
        "count": count,
        "hits": hits,
        "hit_rate": hit_rate,
        "avg_change_pct": avg_change,
        "benchmark_change_pct": benchmark_change_pct,
        "analysis": analysis,
        "rows": rows,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def serialize_recommendations(recs: list[Recommendation]) -> list[dict[str, Any]]:
    return [asdict(r) for r in recs]


def today_file(base_dir: Path, prefix: str) -> Path:
    return base_dir / f"{prefix}_{date.today().strftime('%Y%m%d')}.json"


def backtest_daily_recommendations(rows: list[dict[str, Any]], top_n: int = 5) -> dict[str, Any]:
    if not rows:
        return {"days": 0, "avg_portfolio_change_pct": 0.0, "hit_rate": 0.0}

    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        d = str(row["date"])
        by_date.setdefault(d, []).append(row)

    ordered_dates = sorted(by_date.keys())
    if len(ordered_dates) < 2:
        return {"days": 0, "avg_portfolio_change_pct": 0.0, "hit_rate": 0.0}

    daily_changes: list[float] = []
    total_positions = 0
    total_hits = 0

    for i in range(len(ordered_dates) - 1):
        d0 = ordered_dates[i]
        d1 = ordered_dates[i + 1]
        candidates = by_date[d0]

        scored: list[tuple[str, float]] = []
        for c in candidates:
            close = float(c.get("close", 0.0))
            open_p = float(c.get("open", close or 1.0))
            high = float(c.get("high", close))
            low = float(c.get("low", close))
            volume = int(c.get("volume", 0))
            chg = float(c.get("change_pct", _safe_pct(close - open_p, open_p)))
            proxy = type("Q", (), {
                "last_price": close,
                "open": open_p,
                "high": high,
                "low": low,
                "volume": volume,
                "change_percent": chg,
            })
            score, _ = score_stock_candidate(proxy, vix=15.0, pcr=1.0)
            scored.append((str(c["symbol"]), score))

        picks = [s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        next_day_map = {str(r["symbol"]): r for r in by_date[d1]}

        pick_changes: list[float] = []
        for sym in picks:
            nd = next_day_map.get(sym)
            if not nd:
                continue
            chg = float(nd.get("change_pct", 0.0))
            pick_changes.append(chg)
            total_positions += 1
            if chg > 0:
                total_hits += 1

        if pick_changes:
            daily_changes.append(sum(pick_changes) / len(pick_changes))

    avg_change = round(sum(daily_changes) / len(daily_changes), 2) if daily_changes else 0.0
    hit_rate = round((total_hits / total_positions) * 100.0, 2) if total_positions else 0.0
    return {
        "days": len(daily_changes),
        "avg_portfolio_change_pct": avg_change,
        "hit_rate": hit_rate,
    }
