#!/usr/bin/env python3
"""

Daily Top 100 Gainers Fetcher + Recurrence Analysis

This script:
1. Fetches top 100 gainers daily (designed to run via cron/job scheduler)
2. Analyzes which stocks appeared frequently in last 7 days
3. Generates a recurrence report

Schedule this with cron (crontab -e):
    # Run at 4:30 PM IST (market close)
    30 16 * * 1-5 cd /Users/pratham/aitrade/trade_system && python scripts/daily_top_gainers_scheduler.py --analyze

Usage:
    python scripts/daily_top_gainers_scheduler.py              # Just fetch
    python scripts/daily_top_gainers_scheduler.py --analyze   # Fetch + analyze
    python scripts/daily_top_gainers_scheduler.py --report    # Generate recurrence report only
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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.infrastructure.data.nse_universe import NSE_UNIVERSE

LOGGER = logging.getLogger(__name__)

@dataclass
class StockAppearance:
    """Track stock appearances in top gainers."""
    symbol: str
    name: str
    appearances: int = 0
    total_gain_pct: float = 0.0
    avg_gain_pct: float = 0.0
    last_appearance: str = ""
    appearance_dates: list[str] = field(default_factory=list)
    daily_ranks: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "appearances": self.appearances,
            "total_gain_pct": round(self.total_gain_pct, 2),
            "avg_gain_pct": round(self.avg_gain_pct, 2),
            "last_appearance": self.last_appearance,
            "appearance_dates": self.appearance_dates,
            "daily_ranks": self.daily_ranks,
        }

@dataclass
class RecurrenceReport:
    """Report of frequently appearing stocks."""
    analysis_date: str
    days_analyzed: int
    total_stocks_analyzed: int
    total_unique_stocks_in_top100: int
    stocks_appearing_3_plus_times: list[StockAppearance]
    stocks_appearing_5_plus_times: list[StockAppearance]
    stocks_appearing_every_day: list[StockAppearance]
    day_by_day_breakdown: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_date": self.analysis_date,
            "days_analyzed": self.days_analyzed,
            "total_stocks_analyzed": self.total_stocks_analyzed,
            "total_unique_stocks_in_top100": self.total_unique_stocks_in_top100,
            "stocks_appearing_3_plus_times": [s.to_dict() for s in self.stocks_appearing_3_plus_times],
            "stocks_appearing_5_plus_times": [s.to_dict() for s in self.stocks_appearing_5_plus_times],
            "stocks_appearing_every_day": [s.to_dict() for s in self.stocks_appearing_every_day],
            "day_by_day_breakdown": self.day_by_day_breakdown,
        }

class TopGainersAnalyzer:
    """Analyzes top gainers for recurrence patterns."""

    def __init__(self, data_dir: Path = Path("data/top_gainers")) -> None:
        self.data_dir = Path(data_dir)
        self.history_file = self.data_dir / "top100_history.json"
        self.report_dir = self.data_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_history(self, last_n_days: int = 7) -> list[dict]:
        """Load historical top gainers data."""
        if not self.history_file.exists():
            LOGGER.warning(f"No history file found: {self.history_file}")
            return []

        with open(self.history_file) as f:
            all_history = json.load(f)

        # Sort by date and take last N days
        sorted_history = sorted(all_history, key=lambda x: x.get("date", ""))
        return sorted_history[-last_n_days:]

    def analyze_recurrence(self, last_n_days: int = 7) -> RecurrenceReport:
        """Analyze which stocks appear frequently in top 100."""
        LOGGER.info(f"Analyzing last {last_n_days} days of data...")

        history = self.load_history(last_n_days)
        if not history:
            LOGGER.warning("No data to analyze")
            return RecurrenceReport(
                analysis_date=date.today().isoformat(),
                days_analyzed=0,
                total_stocks_analyzed=0,
                total_unique_stocks_in_top100=0,
                stocks_appearing_3_plus_times=[],
                stocks_appearing_5_plus_times=[],
                stocks_appearing_every_day=[],
            )

        # Track stock appearances
        stock_tracker: dict[str, StockAppearance] = {}
        day_breakdown: dict[str, list[str]] = {}

        for entry in history:
            entry_date = entry.get("date", "")
            day_breakdown[entry_date] = []

            for rank, stock in enumerate(entry.get("gainers", []), 1):
                symbol = stock.get("symbol", "")
                name = stock.get("name", symbol.replace("NSE:", "").replace("-EQ", ""))
                change_pct = stock.get("change_pct", 0)

                if symbol not in stock_tracker:
                    stock_tracker[symbol] = StockAppearance(
                        symbol=symbol,
                        name=name,
                    )

                tracker = stock_tracker[symbol]
                tracker.appearances += 1
                tracker.total_gain_pct += change_pct
                tracker.appearance_dates.append(entry_date)
                tracker.daily_ranks.append(rank)
                tracker.last_appearance = entry_date
                tracker.avg_gain_pct = tracker.total_gain_pct / tracker.appearances

                day_breakdown[entry_date].append(name)

        # Categorize stocks by frequency
        all_stocks = list(stock_tracker.values())

        # 3+ appearances (frequent flyers)
        frequent_3_plus = [s for s in all_stocks if s.appearances >= 3]
        frequent_3_plus.sort(key=lambda x: (x.appearances, x.avg_gain_pct), reverse=True)

        # 5+ appearances (consistent performers)
        frequent_5_plus = [s for s in all_stocks if s.appearances >= 5]
        frequent_5_plus.sort(key=lambda x: (x.appearances, x.avg_gain_pct), reverse=True)

        # Every day (the champions)
        every_day = [s for s in all_stocks if s.appearances == last_n_days]
        every_day.sort(key=lambda x: x.avg_gain_pct, reverse=True)

        total_analyzed = sum(
            entry.get("total_stocks_analyzed", 0)
            for entry in history
        ) // len(history) if history else 0

        return RecurrenceReport(
            analysis_date=date.today().isoformat(),
            days_analyzed=len(history),
            total_stocks_analyzed=total_analyzed,
            total_unique_stocks_in_top100=len(all_stocks),
            stocks_appearing_3_plus_times=frequent_3_plus[:20],  # Top 20
            stocks_appearing_5_plus_times=frequent_5_plus[:15],  # Top 15
            stocks_appearing_every_day=every_day[:10],  # Top 10
            day_by_day_breakdown=day_breakdown,
        )

    def generate_recurrence_report(self, report: RecurrenceReport) -> Path:
        """Generate a comprehensive recurrence report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_file = self.report_dir / f"recurrence_report_{timestamp}.json"
        html_file = self.report_dir / f"recurrence_report_{timestamp}.html"

        # Save JSON
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        LOGGER.info(f"Recurrence report saved: {report_file}")

        # Generate HTML
        self._generate_html_report(report, html_file)

        return report_file

    def _generate_html_report(self, report: RecurrenceReport, output_file: Path) -> None:
        """Generate beautiful HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>NSE Top 100 Gainers - Recurrence Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-card .number {{
            font-size: 36px;
            font-weight: bold;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .symbol {{
            font-family: 'Courier New', monospace;
            font-weight: 600;
            color: #667eea;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-gold {{
            background: #FFD700;
            color: #333;
        }}
        .badge-silver {{
            background: #C0C0C0;
            color: #333;
        }}
        .badge-bronze {{
            background: #CD7F32;
            color: white;
        }}
        .positive {{
            color: #4CAF50;
            font-weight: 600;
        }}
        .frequency-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .frequency-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }}
        .day-timeline {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .day-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
        }}
        .day-card h4 {{
            margin: 0 0 10px 0;
            color: #667eea;
        }}
        .stock-tag {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 NSE Top 100 Gainers - Recurrence Analysis</h1>
        <p class="subtitle">Analysis Period: Last {report.days_analyzed} Days | Generated: {report.analysis_date}</p>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Days Analyzed</h3>
                <div class="number">{report.days_analyzed}</div>
            </div>
            <div class="stat-card">
                <h3>Stocks in Universe</h3>
                <div class="number">{report.total_stocks_analyzed}</div>
            </div>
            <div class="stat-card">
                <h3>Unique Top 100 Stocks</h3>
                <div class="number">{report.total_unique_stocks_in_top100}</div>
            </div>
            <div class="stat-card">
                <h3>Frequent Flyers (3+)</h3>
                <div class="number">{len(report.stocks_appearing_3_plus_times)}</div>
            </div>
        </div>

        <!-- Champions - Appear Every Day -->
        <div class="section">
            <h2>🏆 Champions - Appeared Every Day ({len(report.stocks_appearing_every_day)} stocks)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Symbol</th>
                        <th>Appearances</th>
                        <th>Avg Gain/Day</th>
                        <th>Total Gain</th>
                        <th>Frequency</th>
                    </tr>
                </thead>
                <tbody>
"""

        for i, stock in enumerate(report.stocks_appearing_every_day, 1):
            badge_class = "badge-gold" if i == 1 else "badge-silver" if i == 2 else "badge-bronze" if i == 3 else ""
            badge = f'<span class="badge {badge_class}">#{i}</span>'
            freq_pct = (stock.appearances / report.days_analyzed) * 100

            html += f"""
                    <tr>
                        <td>{badge}</td>
                        <td class="symbol">{stock.name}</td>
                        <td>{stock.appearances}/{report.days_analyzed}</td>
                        <td class="positive">+{stock.avg_gain_pct:.2f}%</td>
                        <td class="positive">+{stock.total_gain_pct:.2f}%</td>
                        <td>
                            <div class="frequency-bar">
                                <div class="frequency-fill" style="width: {freq_pct}%"></div>
                            </div>
                        </td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <!-- Frequent Flyers (5+ times) -->
        <div class="section">
            <h2>⭐ Consistent Performers - 5+ Appearances</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Symbol</th>
                        <th>Appearances</th>
                        <th>Avg Gain</th>
                        <th>Total Gain</th>
                        <th>Last Seen</th>
                    </tr>
                </thead>
                <tbody>
"""

        for i, stock in enumerate(report.stocks_appearing_5_plus_times, 1):
            html += f"""
                    <tr>
                        <td>#{i}</td>
                        <td class="symbol">{stock.name}</td>
                        <td>{stock.appearances}</td>
                        <td class="positive">+{stock.avg_gain_pct:.2f}%</td>
                        <td class="positive">+{stock.total_gain_pct:.2f}%</td>
                        <td>{stock.last_appearance}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <!-- Frequent Flyers (3+ times) -->
        <div class="section">
            <h2>🚀 Frequent Flyers - 3+ Appearances</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Symbol</th>
                        <th>Appearances</th>
                        <th>Avg Gain</th>
                        <th>Total Gain</th>
                        <th>Last Seen</th>
                    </tr>
                </thead>
                <tbody>
"""

        for i, stock in enumerate(report.stocks_appearing_3_plus_times[:20], 1):
            html += f"""
                    <tr>
                        <td>#{i}</td>
                        <td class="symbol">{stock.name}</td>
                        <td>{stock.appearances}</td>
                        <td class="positive">+{stock.avg_gain_pct:.2f}%</td>
                        <td class="positive">+{stock.total_gain_pct:.2f}%</td>
                        <td>{stock.last_appearance}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <!-- Day by Day Breakdown -->
        <div class="section">
            <h2>📅 Day-by-Day Top Gainers</h2>
            <div class="day-timeline">
"""

        for date_str, stocks in report.day_by_day_breakdown.items():
            stock_tags = "".join([f'<span class="stock-tag">{s}</span>' for s in stocks[:10]])
            html += f"""
                <div class="day-card">
                    <h4>{date_str}</h4>
                    <div>{stock_tags}</div>
                </div>
"""

        html += """
            </div>
        </div>

        <footer style="text-align: center; margin-top: 40px; color: #666; font-size: 12px;">
            <p>Generated by NSE Top Gainers Analyzer | Auto-refresh: Daily at Market Close</p>
        </footer>
    </div>
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html)

        LOGGER.info(f"HTML report generated: {output_file}")

def run_daily_fetch() -> bool:
    """Run the daily fetch operation."""
    try:
        # Import and run the fetcher
        from scripts.fetch_nse_top100_gainers import main as fetch_main

        sys.argv = ["fetch_nse_top100_gainers.py"]
        result = fetch_main()
        return result == 0
    except Exception as e:
        LOGGER.error(f"Daily fetch failed: {e}")
        return False

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily Top Gainers Scheduler & Recurrence Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/daily_top_gainers_scheduler.py              # Just fetch today's data
    python scripts/daily_top_gainers_scheduler.py --analyze    # Fetch + generate recurrence report
    python scripts/daily_top_gainers_scheduler.py --report     # Only generate report from existing data
    python scripts/daily_top_gainers_scheduler.py --days 14    # Analyze last 14 days
        """,
    )

    parser.add_argument("--analyze", action="store_true", help="Fetch + analyze recurrence")
    parser.add_argument("--report", action="store_true", help="Generate report only (no fetch)")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze (default: 7)")
    parser.add_argument("--data-dir", type=str, default="data/top_gainers", help="Data directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    LOGGER.info("=" * 70)
    LOGGER.info("🔄 DAILY TOP GAINERS SCHEDULER")
    LOGGER.info("=" * 70)

    # Initialize analyzer
    analyzer = TopGainersAnalyzer(data_dir=Path(args.data_dir))

    # Fetch today's data (unless --report only)
    if not args.report:
        LOGGER.info("Fetching today's top gainers...")
        if run_daily_fetch():
            LOGGER.info("✓ Fetch completed successfully")
        else:
            LOGGER.error("✗ Fetch failed")
            return 1

    # Generate recurrence report
    if args.analyze or args.report:
        LOGGER.info(f"\nAnalyzing last {args.days} days...")
        report = analyzer.analyze_recurrence(last_n_days=args.days)

        LOGGER.info(f"\n📊 Recurrence Summary:")
        LOGGER.info(f"  Days analyzed: {report.days_analyzed}")
        LOGGER.info(f"  Unique stocks in top 100: {report.total_unique_stocks_in_top100}")
        LOGGER.info(f"  Champions (every day): {len(report.stocks_appearing_every_day)}")
        LOGGER.info(f"  5+ appearances: {len(report.stocks_appearing_5_plus_times)}")
        LOGGER.info(f"  3+ appearances: {len(report.stocks_appearing_3_plus_times)}")

        # Print champions
        if report.stocks_appearing_every_day:
            LOGGER.info(f"\n🏆 CHAMPIONS (Appeared all {report.days_analyzed} days):")
            for i, stock in enumerate(report.stocks_appearing_every_day[:10], 1):
                LOGGER.info(
                    f"  {i}. {stock.name:<20} - Avg: +{stock.avg_gain_pct:.2f}% "
                    f"(Total: +{stock.total_gain_pct:.2f}%)"
                )

        # Generate report files
        report_file = analyzer.generate_recurrence_report(report)
        LOGGER.info(f"\n✓ Report saved: {report_file}")

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("Scheduler completed successfully!")
    LOGGER.info("=" * 70)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
