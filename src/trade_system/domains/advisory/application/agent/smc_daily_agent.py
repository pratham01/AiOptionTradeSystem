"""
SMC Daily Scanner Agent — Institutional Smart Money Concept Post-Market Scanner
==============================================================================
Scans the entire F&O stock universe on the Daily timeframe to identify:
1. Order Block (OB) demand & supply retests
2. Fair Value Gap (FVG) imbalance mitigation
3. Buy-Side & Sell-Side Liquidity Sweeps
4. Market Structure breaks (BOS / CHoCH) in Discount / Premium zones

Generates high-conviction BUY CALL & BUY PUT trade recommendations.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.strategy.application.strategies.smc_strategy import (
    SmartMoneyConceptStrategy, SmcTradeSetup
)
from trade_system.domains.analysis.application.analysis.sector_conflict_resolver import (
    SectorConflictResolver, SectorAlignmentResult
)
from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)


class SmcDailyScannerAgent:
    """
    Automated Post-Market Agent for Smart Money Concepts (SMC).
    Analyzes Daily candles across the F&O universe and generates structured trade setups.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        db_path: Optional[str] = None,
        min_score: float = 50.0,
    ) -> None:
        self.settings = settings or Settings.load()
        self.engine = get_engine()
        self.strategy = SmartMoneyConceptStrategy()
        self.conflict_resolver = SectorConflictResolver()
        self.min_score = min_score

        # Telegram Notifier
        telegram_cfg = getattr(self.settings, "telegram", None)
        if telegram_cfg and getattr(telegram_cfg, "bot_token", None):
            self.notifier = TelegramNotifier(
                token=telegram_cfg.bot_token,
                chat_id=telegram_cfg.chat_id,
            )
        else:
            self.notifier = None

    def scan_universe(
        self,
        symbols: Optional[List[str]] = None,
        direction: str = "both",      # "bullish", "bearish", "both"
        lookback_bars: int = 150,
        min_score: Optional[float] = None,
    ) -> List[SmcTradeSetup]:
        """
        Scan all F&O universe stocks on the Daily timeframe.
        Returns a sorted list of high-conviction SMC trade setups.
        """
        threshold = min_score if min_score is not None else self.min_score
        target_symbols = symbols or self._get_universe_symbols()
        LOGGER.info("🔍 SMC Daily Scanner: Scanning %d symbols on Daily timeframe...", len(target_symbols))

        setups: List[SmcTradeSetup] = []

        with self.engine.connect() as conn:
            for symbol in target_symbols:
                try:
                    df = pd.read_sql(
                        text("""
                            SELECT timestamp, open, high, low, close, volume
                            FROM ohlcv_daily
                            WHERE symbol = :sym
                            ORDER BY timestamp ASC
                        """),
                        conn,
                        params={"sym": symbol},
                    )

                    if df.empty or len(df) < 30:
                        continue

                    # Take the most recent lookback bars
                    if len(df) > lookback_bars:
                        df = df.iloc[-lookback_bars:].copy()

                    setup = self.strategy.analyze_symbol(df, symbol=symbol)
                    if setup and setup.confluence_score >= threshold:
                        # Filter by direction
                        if direction == "bullish" and setup.direction != 1:
                            continue
                        elif direction == "bearish" and setup.direction != -1:
                            continue

                        setups.append(setup)

                except Exception as exc:
                    LOGGER.debug("SMC analysis error for %s: %s", symbol, exc)

        # Sort by confluence score descending
        setups.sort(key=lambda s: s.confluence_score, reverse=True)
        LOGGER.info("✅ SMC Daily Scanner: Found %d high-conviction setups.", len(setups))
        return setups

    def send_telegram_report(self, setups: List[SmcTradeSetup], top_n: int = 10) -> bool:
        """Sends formatted Telegram message with top verified SMC setups, suppressing sector conflicts."""
        if not self.notifier or not setups:
            return False

        # 1. Enforce Sector Conflict Resolution
        verified_setups, suppressed_conflicts = self.conflict_resolver.filter_setups_for_telegram(setups)

        if not verified_setups:
            LOGGER.info("SMC Daily Agent: All setups were suppressed due to active sector conflicts.")
            return False

        today_str = date.today().strftime("%d %b %Y")
        top_setups = verified_setups[:top_n]

        lines = [
            f"🏛️ <b>Smart Money Concept (SMC) Daily Report</b>",
            f"<i>Institutional Order Blocks, FVGs & Sector-Verified Flow ({today_str})</i>",
            "",
        ]

        bullish = [s for s in top_setups if s.direction == 1]
        bearish = [s for s in top_setups if s.direction == -1]

        if bullish:
            lines.append("🟢 <b>BULLISH DEMAND & DISCOUNT REVERSALS (Sector Confirmed):</b>")
            for s in bullish:
                sym = s.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                reasons_str = "; ".join(s.reasons[:2])
                lines.append(
                    f"• <b>{sym}</b> (Score: {s.confluence_score:.0f}/100)\n"
                    f"  Entry: ₹{s.entry_price:.2f} | SL: ₹{s.stop_loss:.2f} | T1: ₹{s.target_1:.2f} (1:{s.risk_reward_ratio:.1f} RRR)\n"
                    f"  <i>{reasons_str}</i>"
                )
            lines.append("")

        if bearish:
            lines.append("🔴 <b>BEARISH SUPPLY & PREMIUM REVERSALS (Sector Confirmed):</b>")
            for s in bearish:
                sym = s.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                reasons_str = "; ".join(s.reasons[:2])
                lines.append(
                    f"• <b>{sym}</b> (Score: {s.confluence_score:.0f}/100)\n"
                    f"  Entry: ₹{s.entry_price:.2f} | SL: ₹{s.stop_loss:.2f} | T1: ₹{s.target_1:.2f} (1:{s.risk_reward_ratio:.1f} RRR)\n"
                    f"  <i>{reasons_str}</i>"
                )
            lines.append("")

        if suppressed_conflicts:
            lines.append(f"<i>🛡️ Filtered Out: {len(suppressed_conflicts)} setups suppressed due to sector headwinds.</i>")

        try:
            return self.notifier.send_message("\n".join(lines))
        except Exception as exc:
            LOGGER.error("Failed to send Telegram SMC report: %s", exc)
            return False

    def generate_report_markdown(self, setups: List[SmcTradeSetup]) -> str:
        """Generates markdown report for documentation and artifacts."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# 🏛️ Smart Money Concepts (SMC) Daily Scan Report",
            f"**Generated:** {now_str} IST  ",
            f"**Universe:** 212 F&O Universe Stocks & Major Indices  ",
            f"**Total Setups Found:** {len(setups)}  \n",
            "| Symbol | Sector | Action | Type | Score | Spot / Entry | Stop Loss | Target 1 | RRR | Sector Status |",
            "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for s in setups:
            sym = s.symbol.replace("NSE:", "").replace("-EQ", "")
            icon = "🟢 CALL" if s.direction == 1 else "🔴 PUT"
            sec_eval = self.conflict_resolver.evaluate_alignment(s.symbol, s.direction)
            sec_badge = "✅ ALIGNED" if sec_eval.is_aligned else f"🚫 HEADWIND ({sec_eval.sector_pct:+.1f}%)"
            lines.append(
                f"| **{sym}** | `{sec_eval.sector}` | {icon} | {s.setup_type} | {s.confluence_score:.0f} | ₹{s.entry_price:.2f} | ₹{s.stop_loss:.2f} | ₹{s.target_1:.2f} | 1:{s.risk_reward_ratio:.1f} | {sec_badge} |"
            )

        lines.append("\n## Detailed Setups Rationale\n")
        for idx, s in enumerate(setups, 1):
            sym = s.symbol.replace("NSE:", "").replace("-EQ", "")
            sec_eval = self.conflict_resolver.evaluate_alignment(s.symbol, s.direction)
            lines.append(f"### {idx}. {sym} ({s.action}) — Sector: {sec_eval.sector}")
            lines.append(f"- **Sector Alignment:** {sec_eval.reason}")
            lines.append(f"- **Confluence Score:** {s.confluence_score:.0f}/100")
            lines.append(f"- **Trade Levels:** Entry ₹{s.entry_price:.2f} | Stop Loss ₹{s.stop_loss:.2f} | Target ₹{s.target_1:.2f}")
            lines.append(f"- **Risk-to-Reward:** 1:{s.risk_reward_ratio:.1f}")
            lines.append(f"- **Dealing Range:** {s.equilibrium_status} Zone")
            lines.append(f"- **Institutional Factors:**")
            for r in s.reasons:
                lines.append(f"  - {r}")
            lines.append("")

        return "\n".join(lines)

    def _get_universe_symbols(self) -> List[str]:
        try:
            fo_syms = get_fo_universe()
        except Exception:
            fo_syms = []
        indices = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
        return list(dict.fromkeys(indices + fo_syms))
