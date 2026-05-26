"""
Swing Trading Forensics — Alpha Hunter Edition.
Scans the full F&O universe for multi-day institutional setups.
"""
import asyncio
import logging
from datetime import datetime
import pandas as pd

from trade_system.application.agent.orchestrator import TradeOrchestrator
from trade_system.core import TradeHorizon, TradeDirection
from trade_system.config import Settings
from trade_system.infrastructure.notifications.telegram import TelegramNotifier

async def generate_swing_report():
    logging.basicConfig(level=logging.INFO)
    print("🌊 GENERATING INSTITUTIONAL SWING FORENSICS REPORT...")
    
    settings = Settings.load()
    orc = TradeOrchestrator()
    notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
    
    # 1. Run Session Deliberation
    plan = await orc.run_session()
    
    # 2. Extract Swing Candidates
    swing_trades = [s for s in plan.all_suggestions() if s.horizon == TradeHorizon.SWING]
    
    # 3. Build Detailed Report
    report = [
        "🌊 <b>Institutional Swing Forensics</b>",
        f"📅 Date: {datetime.now().strftime('%d %b %Y')}",
        f"🧬 Regime: {plan.market_context.regime.value}",
        "-----------------------------------",
        ""
    ]
    
    if not swing_trades:
        report.append("⚠️ No high-conviction swing setups passed the Alpha Hunter merge today.")
    else:
        for i, s in enumerate(swing_trades, 1):
            m = s.setup_features.metadata if s.setup_features else {}
            hurst = m.get("hurst_exponent", "N/A")
            
            report += [
                f"{i}. 🚀 <b>{s.symbol.split(':')[-1]} — {s.direction.value} SWING</b>",
                f"   • <b>Conviction:</b> {s.confidence:.0%}",
                f"   • <b>Regime Memory (Hurst):</b> {hurst}",
                f"   • <b>Entry Zone:</b> ₹{s.entry_zone_low:.0f} - ₹{s.entry_zone_high:.0f}",
                f"   • <b>Target:</b> ₹{s.target:.0f} (RR 1:{s.risk_reward():.1f})",
                f"   • <b>Forensics:</b> {s.narrative}",
                ""
            ]
            
        report.append("<i>💡 Logic: These stocks show multi-day institutional persistence (Hurst > 0.55) and have successfully cleared the GEX safety wall.</i>")

    final_msg = "\n".join(report)
    print("\n" + final_msg)
    
    # Send to Telegram
    notifier.send(final_msg)
    print("\n✅ Swing Report dispatched to Telegram.")

if __name__ == "__main__":
    asyncio.run(generate_swing_report())
