"""
MarketSynthesisAgent — Swarm Intelligence Market Synthesizer.
Collects daily trade statistics, sector rotation, volatility coiling (VCP),
technical candle patterns, and multi-timeframe Supertrend levels to generate
a unified daily market report and trade outlook using LLMs.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List

from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.config import Settings
from trade_system.infrastructure.database import log_agent_thought, get_db_session

LOGGER = logging.getLogger(__name__)

class MarketSynthesisAgent:
    def __init__(self, llm_client: LlmAdvisorClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.llm_client = llm_client or LlmAdvisorClient()

    async def generate_daily_synthesis(
        self,
        stats: Dict[str, Any],
        missed: List[str],
        correlation: str,
        skills: List[str],
        patterns: List[Dict[str, Any]],
        st_touches: Dict[str, List[Any]],
        rotation_df: Any,
        vcp: List[Dict[str, Any]] = None
    ) -> str:
        """
        Collects all indicators and stats from today's session and prompts the LLM
        to synthesize a unified market newsletter and next-day outlook.
        """
        if not self.llm_client.configured():
            LOGGER.warning("MarketSynthesisAgent: LLM Client is not configured. Returning default summary.")
            return self._generate_fallback_text(stats, missed, correlation, skills)

        # 1. Process VCP Squeezes
        vcp_list = []
        if vcp:
            for v in vcp[:10]:
                vcp_list.append(f"- {v.get('symbol', 'Unknown')}: BBW={v.get('bbw', 0.0):.3f} | Close={v.get('close', 0.0)}")
        vcp_str = "\n".join(vcp_list) if vcp_list else "None detected."

        # 2. Process Candle Patterns
        patterns_list = []
        if patterns:
            from collections import defaultdict
            p_map = defaultdict(list)
            for p in patterns[:20]:
                for pat_name in p.get('patterns', []):
                    p_map[pat_name].append(p['symbol'].split(':')[-1].replace('-EQ', ''))
            for pat, syms in p_map.items():
                patterns_list.append(f"- {pat}: {', '.join(syms)}")
        patterns_str = "\n".join(patterns_list) if patterns_list else "None detected."

        # 3. Process ST Touches
        touches_list = []
        if st_touches:
            for timeframe, items in st_touches.items():
                syms = [item.symbol.split(':')[-1].replace('-EQ', '') for item in items[:8]]
                touches_list.append(f"- {timeframe} Touches: {', '.join(syms)}")
        touches_str = "\n".join(touches_list) if touches_list else "None detected."

        # 4. Process Sector Rotation
        rotation_str = "None available."
        if rotation_df is not None and not rotation_df.empty:
            leaders = rotation_df[rotation_df['Status'].str.contains('LEADING', na=False)]['Sector'].tolist()[:3]
            laggards = rotation_df[rotation_df['Status'].str.contains('LAGGING', na=False)]['Sector'].tolist()[:3]
            rotation_str = f"Leaders (🔥): {', '.join(leaders)}\nLaggards (❄️): {', '.join(laggards)}"

        # 5. Format prompt
        prompt = f"""You are an elite, institutional-grade market strategist and risk-manager.
Analyze today's quantitative trading data and compile a unified Daily Market Synthesis Report and Trade Outlook for the next session.

### TODAY'S SESSION DATA (Date: {date.today().strftime('%d %b %Y')})

1. Trade Performance Statistics:
- Suggested Trades: {stats.get('total', 0)}
- Win Rate: {stats.get('wins', 0) / (stats.get('total') or 1):.1%} (Wins: {stats.get('wins', 0)} / Losses: {stats.get('losses', 0)})

2. Sector Rotation Metrics:
{rotation_str}

3. Volatility Squeezes (VCP Coiling Candidates):
{vcp_str}

4. Candlestick Technical Patterns (Daily):
{patterns_str}

5. Key Retest Zones (Supertrend touches):
{touches_str}

6. Correlation Analysis:
{correlation}

7. System Performance Critic (Missed Opportunities):
{chr(10).join('- ' + m for m in missed[:3])}

8. Swarm Learned Skills (Weight Evolution):
{chr(10).join('- ' + s for s in skills[:2])}

---

### INSTRUCTIONS:
Create a beautifully structured, premium, and highly actionable market synthesis newsletter. Use rich markdown formatting. Do NOT repeat raw numbers blindly; analyze their implications. The report must contain:
1. **Executive Commentary**: A high-level overview of the day's market regime, bullish/bearish bias, and structural shifts.
2. **Sector Rotation Focus**: What sectors are gaining momentum (where smart money is moving) and which are weakening.
3. **Volatility & Breakout Watch**: Highlight the coiling stocks (VCP) and their breakout potential.
4. **Key Tactical Retests**: Interpret the Supertrend retests. Which support/resistance lines are critical for next-day entries.
5. **Actionable Next-Day Outlook**: Provide 2-3 specific strategy scenarios (e.g. "If Nifty opens flat, watch sector X breakouts; if we gap down, focus on ST retests on Y").
Keep it professional, concise, and easy to read. Output only the report (do not add conversational intro/outro text).
"""
        LOGGER.info("MarketSynthesisAgent: Submitting data to LLM client...")
        try:
            report = await self.llm_client.complete(prompt)
            # Log to DB
            session = get_db_session()
            log_agent_thought(
                session,
                agent_name="MarketSynthesizer",
                action="SYNTHESIS",
                message=report
            )
            session.close()
            LOGGER.info("MarketSynthesisAgent: Synthesis report generated and persisted to database.")
            return report
        except Exception as e:
            LOGGER.error(f"MarketSynthesisAgent: Failed to generate LLM report: {e}")
            return self._generate_fallback_text(stats, missed, correlation, skills)

    def _generate_fallback_text(self, stats: Dict[str, Any], missed: List[str], correlation: str, skills: List[str]) -> str:
        """Fallback in case LLM is unconfigured or errors."""
        lines = [
            "### 🏁 Daily Swarm Synthesis Report",
            f"📅 Session: {date.today().strftime('%d %b %Y')}",
            "",
            "📊 **Today's Stats:**",
            f"- Suggested Trades: {stats.get('total', 0)}",
            f"- Win Rate: {stats.get('wins', 0) / (stats.get('total') or 1):.1%}" if stats.get('total', 0) > 0 else "- No trades taken today.",
            "",
            "🧬 **Correlations:**",
            correlation,
            "",
            "📉 **Performance Critic:**",
        ]
        for m in missed[:3]:
            lines.append(f"- {m}")
        lines.append("")
        lines.append("🧠 **Learned Swarm Skills:**")
        for s in skills[:2]:
            lines.append(f"- {s}")
        return "\n".join(lines)
