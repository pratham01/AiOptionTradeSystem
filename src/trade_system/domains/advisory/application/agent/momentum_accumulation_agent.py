"""Momentum Accumulation Scanner Agent.

Scans the top_gainers_stocks database to detect stocks exhibiting
multi-session accumulation patterns — the kind of sustained buying
that often precedes explosive moves (e.g. the Kirloskar +7.7% surge).

Detection Rules
───────────────
1. Multi-Session Gainer Streak (3+ sessions in top gainers, >8% cumul.)
2. Marubozu After Volume Expansion (strong body + prior vol surge)
3. Group / Sibling Correlation (2+ stocks from same conglomerate)
4. Volume Plateau After Expansion (sustained elevated volume)
5. Composite Priority Score (weighted combination of signals)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import text

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)


# ── Indian Business Group Mapping ──────────────────────────────────────────────
# Maps NSE symbols to their parent business group.  This lets us detect
# "group-wide momentum" — e.g. when KIRLOSBROS and KIRLFER move together.

BUSINESS_GROUPS: dict[str, str] = {
    # Kirloskar Group
    "NSE:KIRLOSBROS-EQ": "Kirloskar",
    "NSE:KIRLFER-EQ": "Kirloskar",
    "NSE:KIRLOSENG-EQ": "Kirloskar",
    # Tata Group
    "NSE:TCS-EQ": "Tata", "NSE:TATAMOTORS-EQ": "Tata", "NSE:TATASTEEL-EQ": "Tata",
    "NSE:TATAPOWER-EQ": "Tata", "NSE:TATACOMM-EQ": "Tata", "NSE:TATACHEM-EQ": "Tata",
    "NSE:TATAELXSI-EQ": "Tata", "NSE:TATAINVEST-EQ": "Tata", "NSE:TATACONSUM-EQ": "Tata",
    "NSE:TITAN-EQ": "Tata", "NSE:VOLTAS-EQ": "Tata", "NSE:INDHOTEL-EQ": "Tata",
    "NSE:RALLIS-EQ": "Tata", "NSE:TRENT-EQ": "Tata",
    # Adani Group
    "NSE:ADANIENT-EQ": "Adani", "NSE:ADANIPORTS-EQ": "Adani", "NSE:ADANIGREEN-EQ": "Adani",
    "NSE:ADANITRANS-EQ": "Adani", "NSE:ADANIPOWER-EQ": "Adani", "NSE:AWL-EQ": "Adani",
    "NSE:ATGL-EQ": "Adani", "NSE:ADANIENSOL-EQ": "Adani", "NSE:NDTV-EQ": "Adani",
    "NSE:AMBUJACEM-EQ": "Adani", "NSE:ACC-EQ": "Adani",
    # Reliance Group
    "NSE:RELIANCE-EQ": "Reliance", "NSE:JIOFIN-EQ": "Reliance",
    # Mahindra Group
    "NSE:M&M-EQ": "Mahindra", "NSE:TECHM-EQ": "Mahindra", "NSE:MAHLOG-EQ": "Mahindra",
    "NSE:MAHLIFE-EQ": "Mahindra", "NSE:MHRIL-EQ": "Mahindra",
    # Bajaj Group
    "NSE:BAJFINANCE-EQ": "Bajaj", "NSE:BAJAJFINSV-EQ": "Bajaj",
    "NSE:BAJAJ-AUTO-EQ": "Bajaj", "NSE:BAJAJELEC-EQ": "Bajaj",
    # Aditya Birla Group
    "NSE:GRASIM-EQ": "Aditya Birla", "NSE:ULTRACEMCO-EQ": "Aditya Birla",
    "NSE:HINDALCO-EQ": "Aditya Birla", "NSE:ABCAPITAL-EQ": "Aditya Birla",
    "NSE:ABFRL-EQ": "Aditya Birla",
    # Godrej Group
    "NSE:GODREJCP-EQ": "Godrej", "NSE:GODREJPROP-EQ": "Godrej",
    "NSE:GODREJIND-EQ": "Godrej", "NSE:GODREJAGRO-EQ": "Godrej",
    # L&T Group
    "NSE:LT-EQ": "L&T", "NSE:LTIM-EQ": "L&T", "NSE:LTTS-EQ": "L&T",
    "NSE:LTTECHFIN-EQ": "L&T",
    # Vedanta Group
    "NSE:VEDL-EQ": "Vedanta", "NSE:HINDCOPPER-EQ": "Vedanta",
    "NSE:HINDPETRO-EQ": "Vedanta",
    # JSW Group
    "NSE:JSWSTEEL-EQ": "JSW", "NSE:JSWENERGY-EQ": "JSW", "NSE:JSWINFRA-EQ": "JSW",
    # Torrent Group
    "NSE:TORNTPHARM-EQ": "Torrent", "NSE:TORNTPOWER-EQ": "Torrent",
    # Murugappa Group
    "NSE:CARBORUNIV-EQ": "Murugappa", "NSE:CHOICEIN-EQ": "Murugappa",
    "NSE:TUBEINVEST-EQ": "Murugappa", "NSE:EID-EQ": "Murugappa",
    # Schneider Group
    "NSE:SCHNEIDER-EQ": "Schneider",
    # TVS Group
    "NSE:TVSMOTOR-EQ": "TVS", "NSE:TVSSCS-EQ": "TVS",
    # Hero Group
    "NSE:HEROMOTOCO-EQ": "Hero",
    # RPG Group
    "NSE:KEC-EQ": "RPG", "NSE:CEAT-EQ": "RPG",
    # Sun Pharma
    "NSE:SUNPHARMA-EQ": "Sun", "NSE:SPARC-EQ": "Sun",
}


@dataclass
class MomentumSignal:
    """One flagged stock with its signals and priority score."""

    symbol: str
    name: str
    score: float = 0.0
    signals: list[str] = field(default_factory=list)
    # Raw metrics for the report
    streak_days: int = 0
    cumulative_change: float = 0.0
    latest_change: float = 0.0
    latest_volume: int = 0
    vol_ratio_latest: float = 0.0
    has_marubozu: bool = False
    group_name: str | None = None
    group_siblings: list[str] = field(default_factory=list)
    daily_data: list[dict[str, Any]] = field(default_factory=list)


class MomentumAccumulationAgent:
    """Scans top_gainers_stocks history to detect multi-session accumulation.

    Usage::

        agent = MomentumAccumulationAgent()
        signals = agent.scan()          # returns list[MomentumSignal]
        agent.send_alerts(signals)       # sends Telegram notification
    """

    # ── Thresholds (tweakable) ─────────────────────────────────────────────
    MIN_STREAK_DAYS: int = 3
    MIN_CUMULATIVE_CHANGE: float = 6.0       # %
    MARUBOZU_BODY_PCT: float = 2.5           # % body
    MARUBOZU_WICK_RATIO: float = 0.35        # upper_wick / body < this
    VOL_EXPANSION_RATIO: float = 2.0         # vs previous session
    VOL_PLATEAU_RATIO: float = 0.7           # min ratio to sustain plateau
    GROUP_MIN_MEMBERS: int = 2
    GROUP_MIN_CHANGE: float = 3.0            # % for at least one sibling
    LOOKBACK_SESSIONS: int = 5               # how many recent sessions to scan

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.engine = get_engine()

    # ── Data Loading ───────────────────────────────────────────────────────

    def _load_recent_snapshots(self) -> pd.DataFrame:
        """Load the last N snapshot dates and all their stocks."""
        query = text("""
            SELECT
                s.id   AS snapshot_id,
                s.date AS session_date,
                t.symbol, t.name,
                t.open, t.high, t.low, t.close, t.prev_close,
                t.volume, t.change, t.change_pct, t.timestamp
            FROM top_gainers_stocks t
            JOIN top_gainers_snapshots s ON t.snapshot_id = s.id
            WHERE s.id IN (
                SELECT id FROM top_gainers_snapshots
                ORDER BY date DESC
                LIMIT :n_sessions
            )
            ORDER BY s.date ASC, t.change_pct DESC
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"n_sessions": self.LOOKBACK_SESSIONS})
        return df

    # ── Rule Detectors ─────────────────────────────────────────────────────

    def _detect_streaks(self, df: pd.DataFrame) -> dict[str, MomentumSignal]:
        """Rule 1: Multi-session gainer streak."""
        dates = sorted(df["session_date"].unique())
        symbol_dates: dict[str, list[dict]] = {}

        for _, row in df.iterrows():
            sym = row["symbol"]
            symbol_dates.setdefault(sym, []).append({
                "date": row["session_date"],
                "change_pct": row["change_pct"],
                "volume": row["volume"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "prev_close": row["prev_close"],
                "name": row["name"],
            })

        signals: dict[str, MomentumSignal] = {}

        for sym, entries in symbol_dates.items():
            unique_dates = sorted(set(e["date"] for e in entries))
            # Find consecutive streak ending at the latest date
            streak = 0
            for i in range(len(dates) - 1, -1, -1):
                if dates[i] in unique_dates:
                    streak += 1
                else:
                    break

            if streak < self.MIN_STREAK_DAYS:
                continue

            cumulative = sum(e["change_pct"] for e in entries)
            if cumulative < self.MIN_CUMULATIVE_CHANGE:
                continue

            latest = max(entries, key=lambda e: e["date"])
            sig = MomentumSignal(
                symbol=sym,
                name=latest["name"],
                streak_days=streak,
                cumulative_change=round(cumulative, 2),
                latest_change=latest["change_pct"],
                latest_volume=latest["volume"],
                daily_data=sorted(entries, key=lambda e: e["date"]),
            )
            sig.signals.append(
                f"🔥 {streak}-session gainer streak ({cumulative:+.1f}% cumulative)"
            )
            sig.score += streak * 2 + cumulative * 0.5
            signals[sym] = sig

        return signals

    def _detect_marubozu(self, signals: dict[str, MomentumSignal], df: pd.DataFrame) -> None:
        """Rule 2: Marubozu (strong body, tiny wick) after volume expansion."""
        for sym, sig in signals.items():
            for i, day in enumerate(sig.daily_data):
                body_pct = (day["close"] - day["open"]) / day["open"] * 100
                upper_wick = day["high"] - max(day["open"], day["close"])
                body_abs = abs(day["close"] - day["open"])

                if body_abs == 0:
                    continue

                wick_ratio = upper_wick / body_abs if body_abs > 0 else 999

                # Check if previous session had volume expansion
                vol_expanded = False
                if i > 0:
                    prev_vol = sig.daily_data[i - 1]["volume"]
                    if prev_vol > 0:
                        vol_ratio = day["volume"] / prev_vol
                        if vol_ratio >= self.VOL_EXPANSION_RATIO:
                            vol_expanded = True
                        sig.vol_ratio_latest = round(vol_ratio, 1)

                if (body_pct >= self.MARUBOZU_BODY_PCT
                        and wick_ratio < self.MARUBOZU_WICK_RATIO):
                    sig.has_marubozu = True
                    label = f"🕯️ Marubozu on {day['date']} (body {body_pct:+.1f}%, wick ratio {wick_ratio:.2f})"
                    if vol_expanded:
                        label += " after vol expansion"
                        sig.score += 8
                    else:
                        sig.score += 4
                    sig.signals.append(label)

    def _detect_group_correlation(
        self, signals: dict[str, MomentumSignal], df: pd.DataFrame
    ) -> None:
        """Rule 3: Multiple stocks from the same business group in top gainers."""
        # Get latest session's stocks
        latest_date = df["session_date"].max()
        latest_df = df[df["session_date"] == latest_date]

        # Also include second-latest to catch pre-move signals
        all_dates = sorted(df["session_date"].unique())
        if len(all_dates) >= 2:
            check_dates = all_dates[-2:]
        else:
            check_dates = [latest_date]

        check_df = df[df["session_date"].isin(check_dates)]

        # Group stocks by business group
        group_stocks: dict[str, list[dict]] = {}
        for _, row in check_df.iterrows():
            group = BUSINESS_GROUPS.get(row["symbol"])
            if group:
                group_stocks.setdefault(group, []).append({
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "change_pct": row["change_pct"],
                    "date": row["session_date"],
                })

        for group_name, members in group_stocks.items():
            unique_symbols = set(m["symbol"] for m in members)
            if len(unique_symbols) < self.GROUP_MIN_MEMBERS:
                continue

            max_change = max(m["change_pct"] for m in members)
            if max_change < self.GROUP_MIN_CHANGE:
                continue

            sibling_names = [m["name"] for m in members]
            for sym in unique_symbols:
                if sym in signals:
                    sig = signals[sym]
                else:
                    # Create new signal for group member not yet flagged
                    latest_member = next(
                        (m for m in members if m["symbol"] == sym), None
                    )
                    if latest_member is None:
                        continue
                    sig = MomentumSignal(
                        symbol=sym,
                        name=latest_member["name"],
                        latest_change=latest_member["change_pct"],
                    )
                    signals[sym] = sig

                others = [m["name"] for m in members if m["symbol"] != sym]
                sig.group_name = group_name
                sig.group_siblings = others
                sig.signals.append(
                    f"👥 {group_name} group momentum ({len(unique_symbols)} members: "
                    + ", ".join(f"{m['name']} {m['change_pct']:+.1f}%" for m in members if m['symbol'] != sym)
                    + ")"
                )
                sig.score += len(unique_symbols) * 3 + max_change

    def _detect_volume_plateau(self, signals: dict[str, MomentumSignal]) -> None:
        """Rule 4: Volume stays elevated across multiple sessions (no collapse)."""
        for sym, sig in signals.items():
            if len(sig.daily_data) < 3:
                continue

            volumes = [d["volume"] for d in sig.daily_data]

            # Check if volume expanded on day N and stayed elevated on N+1, N+2
            for i in range(1, len(volumes)):
                if i == 0 or volumes[i - 1] == 0:
                    continue
                ratio = volumes[i] / volumes[i - 1]
                if ratio >= self.VOL_EXPANSION_RATIO:
                    # Check if subsequent days maintained volume
                    plateau_days = 0
                    for j in range(i + 1, len(volumes)):
                        if volumes[j] >= volumes[i] * self.VOL_PLATEAU_RATIO:
                            plateau_days += 1
                        else:
                            break
                    if plateau_days >= 1:
                        sig.signals.append(
                            f"📊 Volume plateau: elevated for {plateau_days + 1} sessions "
                            f"after {ratio:.1f}x expansion"
                        )
                        sig.score += plateau_days * 3
                        break  # Only report the first plateau

    # ── Main Scan ──────────────────────────────────────────────────────────

    def scan(self) -> list[MomentumSignal]:
        """Run all detection rules and return scored signals."""
        LOGGER.info("Starting Momentum Accumulation scan...")

        df = self._load_recent_snapshots()
        if df.empty:
            LOGGER.warning("No snapshot data found in database.")
            return []

        n_dates = df["session_date"].nunique()
        n_stocks = df["symbol"].nunique()
        LOGGER.info(f"Loaded {len(df)} records across {n_dates} sessions, {n_stocks} unique stocks")

        # Run rules
        signals = self._detect_streaks(df)
        self._detect_marubozu(signals, df)
        self._detect_group_correlation(signals, df)
        self._detect_volume_plateau(signals)

        # Sort by composite score descending
        ranked = sorted(signals.values(), key=lambda s: s.score, reverse=True)

        # Filter out low-score noise
        ranked = [s for s in ranked if s.score >= 5]

        LOGGER.info(f"Found {len(ranked)} momentum accumulation signals")
        return ranked

    # ── Report Formatting ──────────────────────────────────────────────────

    def format_report(self, signals: list[MomentumSignal]) -> str:
        """Format signals as a Telegram-friendly text report."""
        if not signals:
            return "📊 Momentum Accumulation Scanner: No signals detected today."

        lines = [
            "🔥 *MOMENTUM ACCUMULATION SCANNER*",
            f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            f"Stocks with sustained multi-session buying patterns\n",
        ]

        for i, sig in enumerate(signals[:10], 1):  # Top 10
            clean = sig.symbol.replace("NSE:", "").replace("-EQ", "")
            priority = "🔴" if sig.score >= 20 else "🟡" if sig.score >= 10 else "⚪"

            lines.append(f"{priority} *{i}. {clean}* (Score: {sig.score:.0f})")

            if sig.streak_days > 0:
                lines.append(
                    f"   📈 {sig.streak_days}-day streak: {sig.cumulative_change:+.1f}% cumulative"
                )

            # Daily breakdown
            if sig.daily_data:
                daily_str = " → ".join(
                    f"{d['change_pct']:+.1f}%" for d in sig.daily_data
                )
                lines.append(f"   📅 {daily_str}")

            for s in sig.signals:
                if not s.startswith("🔥"):  # Skip streak (already shown)
                    lines.append(f"   {s}")

            lines.append("")

        lines.append("─" * 30)
        lines.append(
            "🟢 ≥20 pts: High conviction | 🟡 10-19: Watch | ⚪ <10: Early signal"
        )

        return "\n".join(lines)

    def format_markdown_report(self, signals: list[MomentumSignal]) -> str:
        """Format signals as a detailed markdown report for file storage."""
        if not signals:
            return "# Momentum Accumulation Scanner\n\nNo signals detected.\n"

        lines = [
            f"# 🔥 Momentum Accumulation Scanner — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**{len(signals)} stocks** detected with sustained multi-session buying patterns.",
            "",
            "| # | Symbol | Score | Streak | Cumul. Chg | Latest Vol | Signals |",
            "|---|--------|-------|--------|-----------|------------|---------|",
        ]

        for i, sig in enumerate(signals[:15], 1):
            clean = sig.symbol.replace("NSE:", "").replace("-EQ", "")
            vol_str = f"{sig.latest_volume/100000:.1f}L" if sig.latest_volume >= 100000 else str(sig.latest_volume)
            sig_summary = "; ".join(s[:50] for s in sig.signals[:3])
            lines.append(
                f"| {i} | **{clean}** | {sig.score:.0f} | {sig.streak_days}d | "
                f"{sig.cumulative_change:+.1f}% | {vol_str} | {sig_summary} |"
            )

        lines.append("")

        # Detailed breakdown for top 5
        lines.append("## Detailed Analysis")
        lines.append("")
        for i, sig in enumerate(signals[:5], 1):
            clean = sig.symbol.replace("NSE:", "").replace("-EQ", "")
            lines.append(f"### {i}. {clean} (Score: {sig.score:.0f})")
            lines.append("")
            if sig.daily_data:
                lines.append("| Date | Open | Close | Change | Volume |")
                lines.append("|------|------|-------|--------|--------|")
                for d in sig.daily_data:
                    vol_str = f"{d['volume']/100000:.1f}L" if d['volume'] >= 100000 else str(d['volume'])
                    lines.append(
                        f"| {d['date']} | ₹{d['open']:.1f} | ₹{d['close']:.1f} | "
                        f"{d['change_pct']:+.2f}% | {vol_str} |"
                    )
                lines.append("")
            lines.append("**Signals:**")
            for s in sig.signals:
                lines.append(f"- {s}")
            lines.append("")

        return "\n".join(lines)

    # ── Notification ───────────────────────────────────────────────────────

    def send_alerts(self, signals: list[MomentumSignal]) -> bool:
        """Send the scan results to Telegram."""
        if not signals:
            LOGGER.info("No signals to send.")
            return True

        report = self.format_report(signals)

        try:
            bot_token = self.settings.telegram.bot_token
            chat_id = self.settings.telegram.chat_id
            if bot_token and chat_id:
                notifier = TelegramNotifier(token=bot_token, chat_id=chat_id)
                if not notifier.send(report, parse_mode="Markdown"):
                    # Retry without markdown if it fails
                    notifier.send(report, parse_mode=None)
                LOGGER.info("Momentum scan sent to Telegram.")
                return True
            else:
                LOGGER.warning("Telegram not configured. Printing report to stdout.")
                print(report)
                return False
        except Exception as e:
            LOGGER.error(f"Failed to send Telegram notification: {e}")
            print(report)
            return False

    # ── Convenience ────────────────────────────────────────────────────────

    def run(self) -> list[MomentumSignal]:
        """Scan + alert in one call."""
        signals = self.scan()
        self.send_alerts(signals)

        # Save markdown report
        if signals:
            md = self.format_markdown_report(signals)
            report_path = f"scratch/momentum_scan_{date.today().strftime('%Y%m%d')}.md"
            try:
                from pathlib import Path
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(report_path).write_text(md)
                LOGGER.info(f"Report saved to {report_path}")
            except Exception as e:
                LOGGER.warning(f"Could not save report: {e}")

        return signals


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    agent = MomentumAccumulationAgent()
    results = agent.run()

    if results:
        print(f"\n{'='*60}")
        print(f"Found {len(results)} momentum signals:")
        print(f"{'='*60}")
        for sig in results[:10]:
            clean = sig.symbol.replace("NSE:", "").replace("-EQ", "")
            print(f"\n  {clean} (Score: {sig.score:.0f})")
            for s in sig.signals:
                print(f"    {s}")
    else:
        print("No momentum accumulation signals detected.")
