#!/usr/bin/env python3
"""

NSE Top Gainers Analysis Script

Fetches top N gainers from a predefined NSE stock watchlist, applies filters,
and generates a recurrence report showing how often top stocks reappear.

Usage:
    python scripts/nse_top_gainers_analysis.py --top-n 20 --lookback-days 30
    python scripts/nse_top_gainers_analysis.py --fetch-only
    python scripts/nse_top_gainers_analysis.py --generate-report --last-m-days 30
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from trade_system.config import Settings
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.brokers.factory import get_broker_manager

LOGGER = logging.getLogger(__name__)

# Predefined NSE stock watchlist (can be extended)
NSE_STOCKS = [
    "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ",
    "NSE:INFY-EQ", "NSE:SBIN-EQ", "NSE:HINDUNILVR-EQ", "NSE:ITC-EQ",
    "NSE:HDFC-EQ", "NSE:KOTAKBANK-EQ", "NSE:LT-EQ", "NSE:AXISBANK-EQ",
    "NSE:BAJFINANCE-EQ", "NSE:MARUTI-EQ", "NSE:ASIANPAINT-EQ", "NSE:ONGC-EQ",
    "NSE:WIPRO-EQ", "NSE:TITAN-EQ", "NSE:ADANIENT-EQ", "NSE:ADANIPORTS-EQ",
    "NSE:BAJAJFINSV-EQ", "NSE:NESTLEIND-EQ", "NSE:ULTRACEMCO-EQ", "NSE:TATAMOTORS-EQ",
    "NSE:POWERGRID-EQ", "NSE:COALINDIA-EQ", "NSE:GRASIM-EQ", "NSE:TECHM-EQ",
    "NSE:BRITANNIA-EQ", "NSE:NTPC-EQ", "NSE:HCLTECH-EQ", "NSE:JSWSTEEL-EQ",
    "NSE:DRREDDY-EQ", "NSE:INDUSINDBK-EQ", "NSE:CIPLA-EQ", "NSE:SBILIFE-EQ",
    "NSE:EICHERMOT-EQ", "NSE:BAJAJ-AUTO-EQ", "NSE:TATASTEEL-EQ", "NSE:APOLLOHOSP-EQ",
    "NSE:HEROMOTOCO-EQ", "NSE:BHARTIARTL-EQ", "NSE:M&M-EQ", "NSE:UPL-EQ",
    "NSE:TATACONSUM-EQ", "NSE:DIVISLAB-EQ", "NSE:HDFCLIFE-EQ", "NSE:SUNPHARMA-EQ",
    "NSE:BPCL-EQ", "NSE:IOC-EQ", "NSE:HINDALCO-EQ",
]

@dataclass
class StockData:
    """Single stock data point."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    volume: int
    day_high: float = 0.0
    day_low: float = 0.0

    @property
    def change_pct(self) -> float:
        """Calculate percentage change from previous close."""
        if self.prev_close == 0:
            return 0.0
        return ((self.close - self.prev_close) / self.prev_close) * 100

    @property
    def intraday_range_pct(self) -> float:
        """Calculate intraday range percentage."""
        if self.open == 0:
            return 0.0
        return ((self.high - self.low) / self.open) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "prev_close": self.prev_close,
            "volume": self.volume,
            "change_pct": round(self.change_pct, 2),
            "intraday_range_pct": round(self.intraday_range_pct, 2),
        }

@dataclass
class Filters:
    """Filters for top gainers selection."""
    min_price: float = 50.0  # Minimum stock price
    max_price: float = 10000.0  # Maximum stock price
    min_volume: int = 100000  # Minimum daily volume
    max_intraday_range_pct: float = 20.0  # Max intraday volatility
    min_change_pct: float = 0.0  # Minimum gain to qualify

    def apply(self, stocks: list[StockData]) -> list[StockData]:
        """Apply filters to stock list."""
        filtered = []
        for stock in stocks:
            if stock.close < self.min_price:
                continue
            if stock.close > self.max_price:
                continue
            if stock.volume < self.min_volume:
                continue
            if stock.intraday_range_pct > self.max_intraday_range_pct:
                continue
            if stock.change_pct < self.min_change_pct:
                continue
            filtered.append(stock)
        return filtered

@dataclass
class DailyTopGainers:
    """Top gainers for a specific day."""
    date: date
    top_n: int
    gainers: list[StockData] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "top_n": self.top_n,
            "gainers": [g.to_dict() for g in self.gainers],
        }

class NSETopGainersAnalyzer:
    """Main analyzer class for NSE top gainers."""

    def __init__(
        self,
        broker: FyersBroker,
        data_dir: Path = Path("data/top_gainers"),
        filters: Filters | None = None,
    ) -> None:
        self.broker = broker
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.filters = filters or Filters()
        self.history_file = self.data_dir / "history.json"

    def fetch_daily_quotes(self, symbols: list[str] | None = None) -> list[StockData]:
        """Fetch current day quotes for all stocks."""
        symbols = symbols or NSE_STOCKS
        LOGGER.info("Fetching quotes for %d stocks...", len(symbols))

        quotes = self.broker.get_quotes(symbols)
        stocks: list[StockData] = []

        for symbol, data in quotes.items():
            try:
                # Parse Fyers quote format
                stock = StockData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=float(data.get("open", 0)),
                    high=float(data.get("high", 0)),
                    low=float(data.get("low", 0)),
                    close=float(data.get("lp", 0)),  # last price
                    prev_close=float(data.get("prev_close_price", 0)),
                    volume=int(data.get("volume", 0)),
                    day_high=float(data.get("high", 0)),
                    day_low=float(data.get("low", 0)),
                )
                stocks.append(stock)
            except (KeyError, ValueError, TypeError) as e:
                LOGGER.warning("Failed to parse data for %s: %s", symbol, e)

        LOGGER.info("Successfully fetched %d stocks", len(stocks))
        return stocks

    def get_top_gainers(
        self,
        stocks: list[StockData],
        top_n: int = 20,
    ) -> DailyTopGainers:
        """Get top N gainers after applying filters."""
        # Apply filters
        filtered = self.filters.apply(stocks)
        LOGGER.info("%d stocks passed filters", len(filtered))

        # Sort by change percentage (descending)
        sorted_stocks = sorted(filtered, key=lambda s: s.change_pct, reverse=True)

        # Take top N
        top_gainers = DailyTopGainers(
            date=date.today(),
            top_n=top_n,
            gainers=sorted_stocks[:top_n],
        )

        return top_gainers

    def save_daily_data(self, daily_data: DailyTopGainers) -> None:
        """Save daily top gainers to file."""
        # Load existing history
        history: list[dict] = []
        if self.history_file.exists():
            with open(self.history_file) as f:
                history = json.load(f)

        # Append new data
        history.append(daily_data.to_dict())

        # Save back
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

        LOGGER.info("Saved data for %s", daily_data.date)

    def load_history(self, days: int | None = None) -> list[DailyTopGainers]:
        """Load historical top gainers data."""
        if not self.history_file.exists():
            return []

        with open(self.history_file) as f:
            data = json.load(f)

        result: list[DailyTopGainers] = []
        for item in data:
            gainers = [
                StockData(
                    symbol=g["symbol"],
                    timestamp=datetime.fromisoformat(g["timestamp"]),
                    open=g["open"],
                    high=g["high"],
                    low=g["low"],
                    close=g["close"],
                    prev_close=g["prev_close"],
                    volume=g["volume"],
                )
                for g in item.get("gainers", [])
            ]
            result.append(DailyTopGainers(
                date=date.fromisoformat(item["date"]),
                top_n=item["top_n"],
                gainers=gainers,
            ))

        # Filter by days if specified
        if days:
            cutoff = date.today() - timedelta(days=days)
            result = [r for r in result if r.date >= cutoff]

        return result

    def generate_recurrence_report(self, last_m_days: int = 30) -> dict[str, Any]:
        """Generate report on how often stocks reappear in top gainers."""
        history = self.load_history(days=last_m_days)

        if not history:
            return {"error": "No historical data available"}

        # Count occurrences of each stock
        stock_counts: dict[str, int] = {}
        stock_avg_gains: dict[str, list[float]] = {}

        for daily in history:
            for stock in daily.gainers:
                symbol = stock.symbol
                stock_counts[symbol] = stock_counts.get(symbol, 0) + 1

                if symbol not in stock_avg_gains:
                    stock_avg_gains[symbol] = []
                stock_avg_gains[symbol].append(stock.change_pct)

        # Calculate statistics
        total_days = len(history)
        recurring_stocks = [
            {
                "symbol": symbol,
                "appearances": count,
                "appearance_rate": round(count / total_days * 100, 2),
                "avg_gain_pct": round(sum(stock_avg_gains[symbol]) / len(stock_avg_gains[symbol]), 2),
                "max_gain_pct": round(max(stock_avg_gains[symbol]), 2),
            }
            for symbol, count in stock_counts.items()
            if count > 1  # Only include stocks that appeared more than once
        ]

        # Sort by appearance count (descending)
        recurring_stocks.sort(key=lambda x: x["appearances"], reverse=True)

        # Get today's top gainers for reference
        today_data = None
        try:
            stocks = self.fetch_daily_quotes()
            today_data = self.get_top_gainers(stocks, top_n=20)
        except Exception as e:
            LOGGER.warning("Could not fetch today's data: %s", e)

        report = {
            "report_date": date.today().isoformat(),
            "analysis_period_days": last_m_days,
            "total_trading_days_analyzed": total_days,
            "unique_stocks_in_history": len(stock_counts),
            "recurring_stocks_count": len(recurring_stocks),
            "top_recurring_stocks": recurring_stocks[:20],  # Top 20 recurring
            "all_stocks_appearance": [
                {"symbol": s, "appearances": c, "rate_pct": round(c / total_days * 100, 2)}
                for s, c in sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "todays_top_gainers": today_data.to_dict() if today_data else None,
        }

        return report

    def export_report_to_csv(self, report: dict[str, Any], output_dir: Path | None = None) -> Path:
        """Export report to CSV files."""
        output_dir = output_dir or self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = date.today().strftime("%Y%m%d")

        # Export recurring stocks
        if report.get("top_recurring_stocks"):
            df = pd.DataFrame(report["top_recurring_stocks"])
            csv_path = output_dir / f"recurring_stocks_{date_str}.csv"
            df.to_csv(csv_path, index=False)
            LOGGER.info("Exported recurring stocks to %s", csv_path)

        # Export all appearances
        if report.get("all_stocks_appearance"):
            df = pd.DataFrame(report["all_stocks_appearance"])
            csv_path = output_dir / f"all_appearances_{date_str}.csv"
            df.to_csv(csv_path, index=False)
            LOGGER.info("Exported all appearances to %s", csv_path)

        # Save JSON report
        json_path = output_dir / f"report_{date_str}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        LOGGER.info("Saved full report to %s", json_path)

        return json_path

def setup_broker() -> FyersBroker:
    """Initialize Fyers broker with settings."""
    settings = Settings.load()

    # Load token from file
    token_file = Path(".secrets/fyers_token.json")
    if not token_file.exists():
        raise FileNotFoundError("Token file not found. Run authenticate_fyers_totp.py first.")

    with open(token_file) as f:
        token_data = json.load(f)
        access_token = token_data.get("access_token", "")

    broker = FyersBroker(
        client_id=settings.fyers_client_id,
        access_token=access_token,
        user_id=settings.fyers_user_id,
    )

    if not broker.authenticate():
        raise RuntimeError("Failed to authenticate with Fyers")

    return broker

def main() -> int:
    parser = argparse.ArgumentParser(
        description="NSE Top Gainers Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch and save today's top gainers
  python scripts/nse_top_gainers_analysis.py --fetch-only

  # Generate recurrence report for last 30 days
  python scripts/nse_top_gainers_analysis.py --generate-report --last-m-days 30

  # Full analysis: fetch today + generate report
  python scripts/nse_top_gainers_analysis.py --full-analysis --top-n 20 --last-m-days 30
        """,
    )

    parser.add_argument("--top-n", type=int, default=20, help="Number of top gainers to fetch (default: 20)")
    parser.add_argument("--last-m-days", type=int, default=30, help="Lookback period for recurrence analysis (default: 30)")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch and save today's data")
    parser.add_argument("--generate-report", action="store_true", help="Only generate recurrence report")
    parser.add_argument("--full-analysis", action="store_true", help="Fetch today's data + generate report")
    parser.add_argument("--min-price", type=float, default=50.0, help="Minimum stock price filter")
    parser.add_argument("--max-price", type=float, default=10000.0, help="Maximum stock price filter")
    parser.add_argument("--min-volume", type=int, default=100000, help="Minimum volume filter")
    parser.add_argument("--data-dir", type=str, default="data/top_gainers", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="reports/top_gainers", help="Output directory for reports")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine mode
    if not (args.fetch_only or args.generate_report or args.full_analysis):
        LOGGER.error("No action specified. Use --fetch-only, --generate-report, or --full-analysis")
        return 1

    try:
        # Setup
        broker = setup_broker()
        LOGGER.info("Authenticated with Fyers successfully")

        filters = Filters(
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume,
        )

        analyzer = NSETopGainersAnalyzer(
            broker=broker,
            data_dir=Path(args.data_dir),
            filters=filters,
        )

        # Execute based on mode
        if args.fetch_only or args.full_analysis:
            LOGGER.info("Fetching today's top %d gainers...", args.top_n)
            stocks = analyzer.fetch_daily_quotes()
            top_gainers = analyzer.get_top_gainers(stocks, top_n=args.top_n)
            analyzer.save_daily_data(top_gainers)

            LOGGER.info("\nToday's Top %d Gainers:", args.top_n)
            for i, stock in enumerate(top_gainers.gainers, 1):
                LOGGER.info(
                    "%2d. %-20s Close: %8.2f | Change: %6.2f%% | Volume: %12d",
                    i,
                    stock.symbol.replace("NSE:", "").replace("-EQ", ""),
                    stock.close,
                    stock.change_pct,
                    stock.volume,
                )

        if args.generate_report or args.full_analysis:
            LOGGER.info("\nGenerating recurrence report for last %d days...", args.last_m_days)
            report = analyzer.generate_recurrence_report(last_m_days=args.last_m_days)

            LOGGER.info("\n=== RECURRENCE REPORT ===")
            LOGGER.info("Analysis Period: %d days", report["analysis_period_days"])
            LOGGER.info("Trading Days Analyzed: %d", report["total_trading_days_analyzed"])
            LOGGER.info("Unique Stocks: %d", report["unique_stocks_in_history"])
            LOGGER.info("Recurring Stocks: %d", report["recurring_stocks_count"])

            if report["top_recurring_stocks"]:
                LOGGER.info("\nTop 10 Most Frequent Gainers:")
                for i, stock in enumerate(report["top_recurring_stocks"][:10], 1):
                    LOGGER.info(
                        "%2d. %-20s | Appearances: %2d/%d (%5.1f%%) | Avg Gain: %5.2f%%",
                        i,
                        stock["symbol"].replace("NSE:", "").replace("-EQ", ""),
                        stock["appearances"],
                        report["total_trading_days_analyzed"],
                        stock["appearance_rate"],
                        stock["avg_gain_pct"],
                    )

            # Export to files
            output_path = analyzer.export_report_to_csv(report, Path(args.output_dir))
            LOGGER.info("\nReport exported to: %s", output_path)

            # Also print JSON summary
            print("\n" + "=" * 60)
            print("JSON REPORT SUMMARY")
            print("=" * 60)
            print(json.dumps(report, indent=2, default=str)[:2000] + "...")

        return 0

    except Exception as e:
        LOGGER.error("Error: %s", e, exc_info=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
