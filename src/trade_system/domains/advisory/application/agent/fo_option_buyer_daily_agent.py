#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import argparse
import json
import logging
from datetime import date
from typing import Any

import pandas as pd

from trade_system.domains.advisory.application.agent.fo_option_buyer_agent import (
    build_recommendations,
    validate_recommendations,
    backtest_daily_recommendations,
    save_json,
    serialize_recommendations,
)
from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.shared.notifications.telegram import TelegramNotifier
from trade_system.domains.analysis.application.analysis.option_chain_analyzer import OptionChainAnalyzer

LOGGER = logging.getLogger(__name__)


def setup_broker() -> FyersBroker:
    settings = Settings.load()
    token_file = Path(".secrets/fyers_token.json")
    if not token_file.exists():
        raise FileNotFoundError("Token not found. Run authenticate_fyers_totp.py first.")
    token_data = json.loads(token_file.read_text())
    broker = FyersBroker(
        client_id=settings.fyers.client_id,
        access_token=token_data.get("access_token", ""),
        user_id=settings.fyers.user_id,
    )
    if not broker.authenticate():
        raise RuntimeError("Fyers authentication failed")
    return broker


def fetch_market_context(broker: FyersBroker) -> dict[str, Any]:
    analyzer = OptionChainAnalyzer(broker.fyers, symbol="NIFTY50", strike_count=20)
    out = analyzer.analyze()
    pcr = None
    vix = None
    market_nature = "unknown"
    if out:
        metrics = out.get("metrics", {})
        pcr = metrics.get("pcr_oi")
        vix = metrics.get("vix")
        market_nature = out.get("market_nature", {}).get("regime", "unknown")
    return {"pcr": pcr, "vix": vix, "market_nature": market_nature}


def run_recommend(top_n: int, out_dir: Path, send_telegram: bool) -> Path:
    broker = setup_broker()
    context = fetch_market_context(broker)
    quotes = broker.get_quotes(get_fo_universe())

    recs = build_recommendations(quotes=quotes, top_n=top_n, vix=context["vix"], pcr=context["pcr"])
    payload = {
        "date": date.today().isoformat(),
        "top_n": top_n,
        "market_context": context,
        "recommendations": serialize_recommendations(recs),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fo_option_buyer_reco_{date.today().strftime('%Y%m%d')}.json"
    save_json(out_path, payload)

    if send_telegram:
        settings = Settings.load()
        telegram_cfg = settings.st_confirmed_telegram if settings.st_confirmed_telegram.enabled else settings.telegram
        if telegram_cfg.enabled:
            notifier = TelegramNotifier(telegram_cfg.bot_token, telegram_cfg.chat_id)
            lines = [
                f"🎯 <b>F&O Option Buyer Picks (Top {top_n})</b>",
                f"PCR: {context['pcr']} | VIX: {context['vix']} | Regime: {context['market_nature']}",
                "",
            ]
            for i, r in enumerate(recs, 1):
                sym = r.symbol.replace("NSE:", "").replace("-EQ", "")
                conviction = "🔥 HIGH" if r.score >= 80 else "✅ MED" if r.score >= 60 else "⚠️ LOW"
                lines.append(f"{i:02d}. <b>{sym}</b> ({r.sector}) score={r.score:.2f} [{conviction}] spot={r.spot:.2f}")
            notifier.send("\n".join(lines))

    LOGGER.info("Saved recommendations: %s", out_path)
    return out_path


def resolve_latest_reco_file(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("fo_option_buyer_reco_*.json"))
    if not files:
        raise FileNotFoundError(f"No recommendation files found in {out_dir}")
    return files[-1]


def run_validate(input_file: Path | None, out_dir: Path, send_telegram: bool) -> Path:
    broker = setup_broker()
    if input_file is None:
        input_file = resolve_latest_reco_file(out_dir)
    payload = json.loads(input_file.read_text())
    recs = payload.get("recommendations", [])
    symbols = [r["symbol"] for r in recs]
    quotes = broker.get_quotes(symbols)

    from trade_system.domains.advisory.application.agent.fo_option_buyer_agent import Recommendation
    reco_objs = [Recommendation(**r) for r in recs]

    benchmark = broker.get_quotes(["NSE:NIFTY50-INDEX"]).get("NSE:NIFTY50-INDEX")
    benchmark_change = float(benchmark.change_percent) if benchmark else 0.0

    report = validate_recommendations(
        recommendation_date=payload.get("date", date.today().isoformat()),
        recommendations=reco_objs,
        latest_quotes=quotes,
        benchmark_change_pct=benchmark_change,
    )
    report["benchmark_change_pct"] = benchmark_change

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fo_option_buyer_validation_{date.today().strftime('%Y%m%d')}.json"
    save_json(out_path, report)

    if send_telegram:
        settings = Settings.load()
        telegram_cfg = settings.st_confirmed_telegram if settings.st_confirmed_telegram.enabled else settings.telegram
        if telegram_cfg.enabled:
            notifier = TelegramNotifier(telegram_cfg.bot_token, telegram_cfg.chat_id)
            lines = [
                "📘 <b>F&O Option Buyer Validation</b>",
                f"Hit rate: {report['hit_rate']}% | Avg change: {report['avg_change_pct']}%",
                f"Benchmark (NIFTY): {benchmark_change:.2f}%",
                "",
                f"📝 <b>Analysis:</b>",
                report["analysis"],
            ]
            notifier.send("\n".join(lines))

    LOGGER.info("Saved validation report: %s", out_path)
    return out_path


def run_backtest(history_csv: Path, top_n: int, out_dir: Path) -> Path:
    df = pd.read_csv(history_csv)
    if "timestamp" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], format="mixed").dt.date.astype(str)
    if "change_pct" not in df.columns:
        df["change_pct"] = ((df["close"] - df["open"]) / df["open"]) * 100.0

    rows = df[["date", "symbol", "open", "close", "high", "low", "volume", "change_pct"]].to_dict(orient="records")
    result = backtest_daily_recommendations(rows, top_n=top_n)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fo_option_buyer_backtest_{date.today().strftime('%Y%m%d')}.json"
    save_json(out_path, result)
    LOGGER.info("Saved backtest report: %s", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily F&O option-buyer recommendation agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("recommend")
    p_rec.add_argument("--top-n", type=int, default=5)
    p_rec.add_argument("--out-dir", default="data/fo_option_buyer_agent")
    p_rec.add_argument("--send-telegram", action="store_true")

    p_val = sub.add_parser("validate")
    p_val.add_argument("--input-file")
    p_val.add_argument("--out-dir", default="data/fo_option_buyer_agent")
    p_val.add_argument("--send-telegram", action="store_true")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--history-csv", default="data/fo_performance_master.csv")
    p_bt.add_argument("--top-n", type=int, default=5)
    p_bt.add_argument("--out-dir", default="reports/fo_option_buyer_agent")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.cmd == "recommend":
        run_recommend(args.top_n, Path(args.out_dir), args.send_telegram)
        return 0
    if args.cmd == "validate":
        input_file = Path(args.input_file) if args.input_file else None
        run_validate(input_file, Path(args.out_dir), args.send_telegram)
        return 0
    if args.cmd == "backtest":
        run_backtest(Path(args.history_csv), args.top_n, Path(args.out_dir))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
