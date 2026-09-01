"""
Wyckoff Daily Scanner Agent — Institutional Accumulation & Distribution Scanner
================================================================================
Scans the F&O universe on the Daily timeframe to detect:
- Phase C Springs (Accumulation shakeouts)
- Phase C UTADs (Distribution traps)
- Sign of Strength (SOS) & Sign of Weakness (SOW) volume breakouts
- Stopping volume & No Supply/Demand test confirmations
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import text

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.market_data.infrastructure.data.fo_universe import get_fo_universe
from trade_system.domains.strategy.application.strategies.wyckoff_strategy import (
    WyckoffVsaStrategy, WyckoffSetup
)
from trade_system.domains.analysis.application.analysis.sector_conflict_resolver import (
    SectorConflictResolver, SectorAlignmentResult
)
from trade_system.shared.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)


class WyckoffDailyScannerAgent:
    """
    Automated Post-Market Agent for Wyckoff Method & Volume-Spread Analysis.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        min_score: float = 60.0,
    ) -> None:
        self.settings = settings or Settings.load()
        self.engine = get_engine()
        self.strategy = WyckoffVsaStrategy()
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
        direction: str = "both",
        lookback_bars: int = 120,
        min_score: Optional[float] = None,
    ) -> List[WyckoffSetup]:
        """Scans the F&O universe for active Wyckoff setups."""
        threshold = min_score if min_score is not None else self.min_score
        target_symbols = symbols or self._get_universe_symbols()
        LOGGER.info("🔍 Wyckoff Scanner: Auditing %d symbols on Daily timeframe...", len(target_symbols))

        setups: List[WyckoffSetup] = []

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

                    if len(df) > lookback_bars:
                        df = df.iloc[-lookback_bars:].copy()

                    setup = self.strategy.analyze_setup(df, symbol=symbol)
                    if setup and setup.confluence_score >= threshold:
                        if direction == "bullish" and setup.direction != 1:
                            continue
                        elif direction == "bearish" and setup.direction != -1:
                            continue

                        setups.append(setup)

                except Exception as exc:
                    LOGGER.debug("Wyckoff analysis error for %s: %s", symbol, exc)

        setups.sort(key=lambda s: s.confluence_score, reverse=True)
        LOGGER.info("✅ Wyckoff Scanner: Identified %d active setups.", len(setups))
        return setups

    def send_telegram_report(self, setups: List[WyckoffSetup], top_n: int = 10) -> bool:
        """Dispatches top verified Wyckoff setups to Telegram, suppressing sector conflicts."""
        if not self.notifier or not setups:
            return False

        # 1. Filter out sector conflicts (e.g. BUY CALL on a stock whose sector is down)
        verified_setups, suppressed_conflicts = self.conflict_resolver.filter_setups_for_telegram(setups)

        if not verified_setups:
            LOGGER.info("Wyckoff Daily Agent: All setups were suppressed due to active sector conflicts.")
            return False

        today_str = date.today().strftime("%d %b %Y")
        top_setups = verified_setups[:top_n]

        lines = [
            f"📈 <b>Wyckoff & VSA Institutional Daily Report</b>",
            f"<i>Phase C Springs, UTADs & Sector-Verified Volume ({today_str})</i>",
            "",
        ]

        bullish = [s for s in top_setups if s.direction == 1]
        bearish = [s for s in top_setups if s.direction == -1]

        if bullish:
            lines.append("🟢 <b>ACCUMULATION SPRINGS & SOS (Sector Confirmed):</b>")
            for s in bullish:
                sym = s.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                reasons_str = "; ".join(s.reasons[:2])
                lines.append(
                    f"• <b>{sym}</b> [{s.pattern_type}] (Score: {s.confluence_score:.0f}/100)\n"
                    f"  Entry: ₹{s.entry_price:.2f} | SL: ₹{s.stop_loss:.2f} | T1: ₹{s.target_1:.2f} (1:{s.risk_reward_ratio:.1f} RRR)\n"
                    f"  <i>{reasons_str}</i>"
                )
            lines.append("")

        if bearish:
            lines.append("🔴 <b>DISTRIBUTION UTADS & SOW (Sector Confirmed):</b>")
            for s in bearish:
                sym = s.symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")
                reasons_str = "; ".join(s.reasons[:2])
                lines.append(
                    f"• <b>{sym}</b> [{s.pattern_type}] (Score: {s.confluence_score:.0f}/100)\n"
                    f"  Entry: ₹{s.entry_price:.2f} | SL: ₹{s.stop_loss:.2f} | T1: ₹{s.target_1:.2f} (1:{s.risk_reward_ratio:.1f} RRR)\n"
                    f"  <i>{reasons_str}</i>"
                )
            lines.append("")

        if suppressed_conflicts:
            lines.append(f"<i>🛡️ Filtered Out: {len(suppressed_conflicts)} setups suppressed due to sector headwinds.</i>")

        try:
            return self.notifier.send_message("\n".join(lines))
        except Exception as exc:
            LOGGER.error("Failed to send Telegram Wyckoff report: %s", exc)
            return False

    def generate_report_markdown(self, setups: List[WyckoffSetup]) -> str:
        """Generates markdown report."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# 📈 Wyckoff & Volume-Spread Analysis (VSA) Daily Scan Report",
            f"**Generated:** {now_str} IST  ",
            f"**Universe:** 212 F&O Stocks & Indices  ",
            f"**Total Setups Found:** {len(setups)}  \n",
            "| Symbol | Sector | Action | Pattern | Phase | Score | Spot / Entry | Stop Loss | Target 1 | RRR | Sector Status |",
            "| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for s in setups:
            sym = s.symbol.replace("NSE:", "").replace("-EQ", "")
            icon = "🟢 CALL" if s.direction == 1 else "🔴 PUT"
            sec_eval = self.conflict_resolver.evaluate_alignment(s.symbol, s.direction)
            sec_badge = "✅ ALIGNED" if sec_eval.is_aligned else f"🚫 HEADWIND ({sec_eval.sector_pct:+.1f}%)"
            lines.append(
                f"| **{sym}** | `{sec_eval.sector}` | {icon} | `{s.pattern_type}` | `{s.phase}` | {s.confluence_score:.0f} | ₹{s.entry_price:.2f} | ₹{s.stop_loss:.2f} | ₹{s.target_1:.2f} | 1:{s.risk_reward_ratio:.1f} | {sec_badge} |"
            )

        lines.append("\n## Detailed Wyckoff Setups Rationale\n")
        for idx, s in enumerate(setups, 1):
            sym = s.symbol.replace("NSE:", "").replace("-EQ", "")
            sec_eval = self.conflict_resolver.evaluate_alignment(s.symbol, s.direction)
            lines.append(f"### {idx}. {sym} — {s.action} [{s.pattern_type}] (Sector: {sec_eval.sector})")
            lines.append(f"- **Sector Alignment:** {sec_eval.reason}")
            lines.append(f"- **Phase:** {s.phase}")
            lines.append(f"- **Trading Range Bounds:** Support ₹{s.trading_range_low:.2f} ➔ Resistance ₹{s.trading_range_high:.2f}")
            lines.append(f"- **Levels:** Entry ₹{s.entry_price:.2f} | Invalidation SL ₹{s.stop_loss:.2f} | Target ₹{s.target_1:.2f}")
            lines.append(f"- **Risk-to-Reward:** 1:{s.risk_reward_ratio:.1f}")
            lines.append(f"- **Volume Activity:** {s.volume_surge_ratio:.1f}x vs 20 SMA")
            lines.append(f"- **Key Factors:**")
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
