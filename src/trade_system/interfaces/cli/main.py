from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from trade_system.application.advisory import TradeAdvisor, TradeContext, TradeProposal
from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.application.backtesting import BacktestEngine
from trade_system.infrastructure.brokers.legacy import FyersAuthService, FyersBrokerClient
from trade_system.config import Settings
from trade_system.infrastructure.data import CsvDataCatalog, HistoricalDataService
from trade_system.interfaces.live import LiveMarketDataService
from trade_system.utils.logging_utils import configure_logging
from trade_system.application.research import AutonomousResearchOrchestrator
from trade_system.application.strategies import build_strategy, registered_strategies

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_parser = subparsers.add_parser("auth", help="Authenticate with FYERS")
    auth_parser.add_argument("--mode", choices=["browser", "totp"], default="browser")
    auth_parser.add_argument("--auth-code")
    auth_parser.add_argument("--no-browser", action="store_true")

    history_parser = subparsers.add_parser("fetch-history", help="Fetch historical candles from FYERS")
    history_parser.add_argument("--symbol", action="append", required=True)
    history_parser.add_argument("--resolution", default="D")
    history_parser.add_argument("--from-date", required=True)
    history_parser.add_argument("--to-date", default=date.today().isoformat())
    history_parser.add_argument("--chunk-days", type=int, default=365)
    history_parser.add_argument("--year-wise", action="store_true")

    live_parser = subparsers.add_parser("live", help="Collect live tick data after authentication")
    live_parser.add_argument("--symbol", action="append")
    live_parser.add_argument("--timeframe-minutes", type=int)

    backtest_parser = subparsers.add_parser("backtest", help="Run local backtest")
    backtest_parser.add_argument("--strategy", default="sma_cross", choices=sorted(registered_strategies()))
    backtest_parser.add_argument("--csv", required=True)
    backtest_parser.add_argument("--fast", type=int, default=5)
    backtest_parser.add_argument("--slow", type=int, default=20)
    backtest_parser.add_argument("--cash", type=float, default=100000.0)
    backtest_parser.add_argument("--quantity", type=float, default=1.0)

    suggest_parser = subparsers.add_parser(
        "suggest-trade",
        help="Evaluate a candidate option buy against the rulebook and optionally call an LLM",
    )
    suggest_parser.add_argument("--input", required=True, help="Path to a JSON file with context and proposal")
    suggest_parser.add_argument("--with-llm", action="store_true", help="Call the configured LLM after rule evaluation")

    research_parser = subparsers.add_parser(
        "research-agent",
        help="Run the experimental autonomous research loop",
    )
    research_parser.add_argument("--year", type=int, default=2026)
    research_parser.add_argument("--symbol", default="NSE_NIFTY50-INDEX")

    sanity_parser = subparsers.add_parser(
        "sanity-check",
        help="Verify and heal F&O database history",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    if args.command == "auth":
        service = FyersAuthService(settings)
        if args.mode == "browser":
            token = service.authenticate_browser(args.auth_code, open_browser=not args.no_browser)
        else:
            token = service.authenticate_totp()
        print(f"FYERS access token saved to {settings.fyers_token_path}")
        print(token)
        return 0

    if args.command == "fetch-history":
        token = _require_token(settings)
        broker = FyersBrokerClient(
            client_id=settings.fyers.client_id,
            access_token=token,
            user_id=settings.fyers.user_id,
        )
        catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
        service = HistoricalDataService(broker, catalog)
        from_date = date.fromisoformat(args.from_date)
        to_date = date.fromisoformat(args.to_date)
        for symbol in args.symbol:
            frame = service.collect(
                symbol=symbol,
                resolution=args.resolution,
                from_date=from_date,
                to_date=to_date,
                chunk_days=args.chunk_days,
                year_wise=args.year_wise,
            )
            print(f"{symbol}: collected {len(frame)} rows")
        return 0

    if args.command == "live":
        token = _require_token(settings)
        broker = FyersBrokerClient(
            client_id=settings.fyers.client_id,
            access_token=token,
            user_id=settings.fyers.user_id,
        )
        catalog = CsvDataCatalog(settings.data_dir / "fo_historical")
        symbols = args.symbol or settings.live_symbols
        timeframe = args.timeframe_minutes or settings.live_timeframe_minutes
        service = LiveMarketDataService(
            broker=broker,
            catalog=catalog,
            symbols=symbols,
            timeframe_minutes=timeframe,
            settings=settings,
        )
        service.run_forever()
        return 0

    if args.command == "sanity-check":
        from trade_system.application.agent.data_sanity_agent import DataSanityAgent
        agent = DataSanityAgent(settings)
        result = agent.ensure_data_sanity(min_candles=100)
        print(f"Sanity check complete. Healed symbols: {result.get('healed_daily', [])}")
        return 0

    if args.command == "backtest":
        candles = pd.read_csv(args.csv, parse_dates=["timestamp"])
        strategy = build_strategy(args.strategy, fast=args.fast, slow=args.slow)
        result = BacktestEngine(
            strategy=strategy,
            initial_cash=args.cash,
            quantity=args.quantity,
        ).run(candles)
        print(json.dumps(result.summary, indent=2))
        trades_path = Path(args.csv).with_name(f"{Path(args.csv).stem}_{args.strategy}_trades.csv")
        if result.trades:
            pd.DataFrame([__import__("dataclasses").asdict(trade) for trade in result.trades]).to_csv(trades_path, index=False)
            print(f"Trades saved to {trades_path}")
        return 0

    if args.command == "suggest-trade":
        payload = json.loads(Path(args.input).read_text())
        context = TradeContext(**payload["context"])
        proposal = TradeProposal(**payload["proposal"])
        advice = TradeAdvisor().evaluate(context, proposal)
        output = {
            "context": payload["context"],
            "proposal": payload["proposal"],
            "advice": {
                "verdict": advice.verdict,
                "score": advice.score,
                "summary": advice.summary,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status,
                        "detail": check.detail,
                    }
                    for check in advice.checks
                ],
                "suggested_adjustments": advice.suggested_adjustments,
            },
        }
        if args.with_llm:
            client = LlmAdvisorClient()
            output["llm_note"] = client.suggest(context=context, proposal=proposal, advice=advice)
        print(json.dumps(output, indent=2))
        return 0

    if args.command == "research-agent":
        orchestrator = AutonomousResearchOrchestrator(root=Path.cwd())
        artifacts = orchestrator.run_zone_upgrade_cycle(year=args.year, symbol=args.symbol)
        print(json.dumps(
            {
                "baseline_summary_path": str(artifacts.baseline_summary_path),
                "upgraded_summary_path": str(artifacts.upgraded_summary_path),
                "candidate_config_path": str(artifacts.candidate_config_path),
                "comparison_path": str(artifacts.comparison_path),
                "report_path": str(artifacts.report_path),
            },
            indent=2,
        ))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _require_token(settings: Settings) -> str:
    service = FyersAuthService(settings)
    token = service.read_cached_token()
    if not token:
        if settings.fyers.access_token:
            return settings.fyers.access_token
        raise RuntimeError("No FYERS token found. Run `trade-system auth` first.")
    return token


def run() -> int:
    return main()
