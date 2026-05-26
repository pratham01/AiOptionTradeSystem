from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class StrategySource:
    strategy: str
    family: str
    summary_path: str
    year_trade_template: str
    strategy_column: str | None = None


SOURCES: tuple[StrategySource, ...] = (
    StrategySource(
        strategy="supertrend_yearly",
        family="Trend",
        summary_path="reports/supertrend_yearly/NSE_NIFTY50-INDEX_supertrend_7_3_yearly_summary.csv",
        year_trade_template="reports/supertrend_yearly/NSE_NIFTY50-INDEX_supertrend_7_3_{year}_trades.csv",
    ),
    StrategySource(
        strategy="supertrend_3m",
        family="Trend",
        summary_path="reports/supertrend_3m/nifty50_3m_summary.csv",
        year_trade_template="reports/supertrend_3m/nifty50_3m_last_month.csv",
    ),
    StrategySource(
        strategy="smc_signal",
        family="SMC",
        summary_path="reports/smc_signal/smc_signal_summary.csv",
        year_trade_template="reports/smc_signal/smc_signal_{year}_trades.csv",
    ),
    StrategySource(
        strategy="ict_fvg_liquidity",
        family="ICT / SMC",
        summary_path="reports/ict_fvg_liquidity/ict_fvg_liquidity_summary.csv",
        year_trade_template="reports/ict_fvg_liquidity/ict_fvg_liquidity_{year}_trades.csv",
    ),
    StrategySource(
        strategy="option_buyer_15m",
        family="Option Buyer",
        summary_path="reports/option_buyer_15m/option_buyer_15m_summary.csv",
        year_trade_template="reports/option_buyer_15m/{strategy}/{strategy}_{year}_trades.csv",
        strategy_column="strategy",
    ),
    StrategySource(
        strategy="price_action_concepts",
        family="Price Action",
        summary_path="reports/price_action_concepts/price_action_concepts_summary.csv",
        year_trade_template="reports/price_action_concepts/{strategy}/{strategy}_{year}_trades.csv",
        strategy_column="strategy",
    ),
    StrategySource(
        strategy="baseline_supertrend",
        family="Master Tournament",
        summary_path="reports/master_tournament/tournament_summary.csv",
        year_trade_template="reports/master_tournament/{strategy}/{strategy}_trades.csv",
        strategy_column="strategy",
    ),
)


def _normalize_summary(frame: pd.DataFrame, *, source: StrategySource, root: Path) -> pd.DataFrame:
    df = frame.copy()
    if source.strategy_column and source.strategy_column in df.columns:
        df["strategy"] = df[source.strategy_column].astype(str)
    else:
        df["strategy"] = source.strategy

    df["family"] = source.family
    df["source_summary"] = str((root / source.summary_path).resolve())
    if "net_points" not in df.columns:
        source_col = "gross_points" if "gross_points" in df.columns else None
        if source_col:
            df["net_points"] = pd.to_numeric(df[source_col], errors="coerce").fillna(0.0)
        else:
            df["net_points"] = 0.0

    if "expectancy" not in df.columns:
        avg_pts_source = df["avg_points"] if "avg_points" in df.columns else pd.Series(0.0, index=df.index)
        df["expectancy"] = pd.to_numeric(avg_pts_source, errors="coerce").fillna(0.0)
    if "profit_factor" not in df.columns:
        avg_win_source = df["avg_win"] if "avg_win" in df.columns else pd.Series(0.0, index=df.index)
        avg_loss_source = df["avg_loss"] if "avg_loss" in df.columns else pd.Series(0.0, index=df.index)
        gross_profit = pd.to_numeric(avg_win_source, errors="coerce").fillna(0.0)
        gross_loss = pd.to_numeric(avg_loss_source, errors="coerce").abs().fillna(0.0)
        df["profit_factor"] = gross_profit.where(gross_loss == 0, gross_profit / gross_loss.replace(0, pd.NA)).fillna(0.0)
    if "avg_r" not in df.columns:
        df["avg_r"] = 0.0
    if "max_drawdown_points" not in df.columns:
        df["max_drawdown_points"] = 0.0
    if "wins" not in df.columns:
        wins = (pd.to_numeric(df.get("trades", 0), errors="coerce").fillna(0.0) * pd.to_numeric(df.get("win_rate", 0), errors="coerce").fillna(0.0) / 100.0)
        df["wins"] = wins.round().astype(int)
    if "losses" not in df.columns:
        trades = pd.to_numeric(df.get("trades", 0), errors="coerce").fillna(0).astype(int)
        df["losses"] = trades - pd.to_numeric(df["wins"], errors="coerce").fillna(0).astype(int)

    wanted = [
        "strategy",
        "family",
        "year",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "net_points",
        "avg_points",
        "expectancy",
        "profit_factor",
        "avg_r",
        "sl_hits",
        "trend_exits",
        "eod_exits",
        "max_win",
        "max_loss",
        "max_drawdown_points",
        "source_summary",
    ]
    for column in wanted:
        if column not in df.columns:
            df[column] = 0
    return df[wanted].copy()


def load_strategy_summaries(root: Path | str) -> pd.DataFrame:
    base = Path(root)
    frames: list[pd.DataFrame] = []
    for source in SOURCES:
        summary_path = base / source.summary_path
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue
        if "year" in frame.columns:
            frame = frame[frame["year"].astype(str).str.upper() != "ALL"].copy()
        frames.append(_normalize_summary(frame, source=source, root=base))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    for column in [
        "trades",
        "wins",
        "losses",
        "win_rate",
        "net_points",
        "avg_points",
        "expectancy",
        "profit_factor",
        "avg_r",
        "sl_hits",
        "trend_exits",
        "eod_exits",
        "max_win",
        "max_loss",
        "max_drawdown_points",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result.sort_values(["year", "family", "strategy"]).reset_index(drop=True)


def _find_source(strategy: str) -> StrategySource | None:
    for source in SOURCES:
        if source.strategy == strategy:
            return source
    for source in SOURCES:
        if source.strategy_column:
            return source
    return None


def _candidate_trade_paths(root: Path, strategy: str, years: list[int] | None) -> list[Path]:
    paths: list[Path] = []
    years = years or []
    for source in SOURCES:
        if source.strategy == strategy:
            if years:
                for year in years:
                    paths.append(root / source.year_trade_template.format(strategy=strategy, year=year))
            else:
                for year in range(2018, 2030):
                    paths.append(root / source.year_trade_template.format(strategy=strategy, year=year))
        elif source.strategy_column:
            if years:
                for year in years:
                    paths.append(root / source.year_trade_template.format(strategy=strategy, year=year))
            else:
                for year in range(2018, 2030):
                    paths.append(root / source.year_trade_template.format(strategy=strategy, year=year))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return [path for path in unique if path.exists()]


def load_strategy_trades(root: Path | str, strategy: str, years: list[int] | None = None) -> pd.DataFrame:
    base = Path(root)
    paths = _candidate_trade_paths(base, strategy, years)
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_path"] = str(path.resolve())
        frame["strategy"] = strategy
        if "entry_time" in frame.columns:
            frame["entry_time"] = pd.to_datetime(frame["entry_time"], errors="coerce")
        if "exit_time" in frame.columns:
            frame["exit_time"] = pd.to_datetime(frame["exit_time"], errors="coerce")
        if "year" not in frame.columns:
            if "entry_time" in frame.columns and frame["entry_time"].notna().any():
                frame["year"] = frame["entry_time"].dt.year
            else:
                stem_parts = path.stem.split("_")
                maybe_year = stem_parts[-2] if stem_parts[-1] == "trades" and len(stem_parts) >= 2 else stem_parts[-1]
                frame["year"] = pd.to_numeric(maybe_year, errors="coerce")
        if "points_captured" not in frame.columns:
            if "gross_points" in frame.columns:
                frame["points_captured"] = pd.to_numeric(frame["gross_points"], errors="coerce").fillna(0.0)
            elif {"entry_price", "exit_price", "direction"}.issubset(frame.columns):
                side = frame["direction"].astype(str).str.upper().map({"LONG": 1, "SHORT": -1}).fillna(0)
                frame["points_captured"] = (pd.to_numeric(frame["exit_price"], errors="coerce").fillna(0.0) - pd.to_numeric(frame["entry_price"], errors="coerce").fillna(0.0)) * side
            else:
                frame["points_captured"] = 0.0
        if "r_multiple" not in frame.columns:
            frame["r_multiple"] = 0.0
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "entry_time" in result.columns and "exit_time" in result.columns:
        result["holding_minutes"] = (
            pd.to_datetime(result["exit_time"], errors="coerce") - pd.to_datetime(result["entry_time"], errors="coerce")
        ).dt.total_seconds() / 60.0
    else:
        result["holding_minutes"] = 0.0
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    return result.sort_values([col for col in ["entry_time", "exit_time"] if col in result.columns]).reset_index(drop=True)


def load_agent_lab_runs(root: Path | str) -> pd.DataFrame:
    base = Path(root) / "reports" / "agent_lab"
    if not base.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        comparison_path = run_dir / "comparison.csv"
        report_path = run_dir / "report.md"
        candidate_path = run_dir / "candidate_config.json"
        if not comparison_path.exists():
            continue
        comparison = pd.read_csv(comparison_path)
        pivot = comparison.set_index("metric")
        rows.append(
            {
                "run_name": run_dir.name,
                "comparison_path": str(comparison_path.resolve()),
                "report_path": str(report_path.resolve()) if report_path.exists() else "",
                "candidate_config_path": str(candidate_path.resolve()) if candidate_path.exists() else "",
                "baseline_net_points": float(pivot.loc["net_points", "baseline"]) if "net_points" in pivot.index else 0.0,
                "upgraded_net_points": float(pivot.loc["net_points", "upgraded"]) if "net_points" in pivot.index else 0.0,
                "baseline_win_rate": float(pivot.loc["win_rate", "baseline"]) if "win_rate" in pivot.index else 0.0,
                "upgraded_win_rate": float(pivot.loc["win_rate", "upgraded"]) if "win_rate" in pivot.index else 0.0,
                "baseline_trades": float(pivot.loc["trades", "baseline"]) if "trades" in pivot.index else 0.0,
                "upgraded_trades": float(pivot.loc["trades", "upgraded"]) if "trades" in pivot.index else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_agent_lab_detail(root: Path | str, run_name: str) -> dict[str, object]:
    run_dir = Path(root) / "reports" / "agent_lab" / run_name
    comparison_path = run_dir / "comparison.csv"
    candidate_path = run_dir / "candidate_config.json"
    report_path = run_dir / "report.md"
    baseline_summary_path = next(iter(sorted((run_dir / "baseline").glob("*summary.csv"))), None)
    upgraded_summary_path = next(iter(sorted((run_dir / "upgraded").glob("*summary.csv"))), None)
    baseline_trades_path = next(iter(sorted((run_dir / "baseline").glob("*trades.csv"))), None)
    upgraded_trades_path = next(iter(sorted((run_dir / "upgraded").glob("*trades.csv"))), None)
    return {
        "comparison": pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame(),
        "candidate": candidate_path.read_text() if candidate_path.exists() else "{}",
        "report": report_path.read_text() if report_path.exists() else "",
        "baseline_summary": pd.read_csv(baseline_summary_path) if baseline_summary_path else pd.DataFrame(),
        "upgraded_summary": pd.read_csv(upgraded_summary_path) if upgraded_summary_path else pd.DataFrame(),
        "baseline_trades": pd.read_csv(baseline_trades_path, parse_dates=["entry_time", "exit_time"]) if baseline_trades_path else pd.DataFrame(),
        "upgraded_trades": pd.read_csv(upgraded_trades_path, parse_dates=["entry_time", "exit_time"]) if upgraded_trades_path else pd.DataFrame(),
    }
