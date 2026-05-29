"""
LiveAlertAgent — Specialized agent for real-time market monitoring and alerting.
Handles OB/OS detection, unusual volume, and SMC signals with configurable thresholds.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd

from trade_system.config import Settings
from trade_system.infrastructure.notifications.telegram import TelegramNotifier

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

class LiveAlertAgent:
    """
    Monitors live data streams and dispatches proactive technical alerts.
    """

    def __init__(self, settings: Settings | None = None, notifier: TelegramNotifier | None = None):
        self.settings = settings or Settings.load()
        self.notifier = notifier or TelegramNotifier(
            self.settings.telegram.bot_token, 
            self.settings.telegram.chat_id
        )
        self.last_alert_time: dict[str, datetime] = {}
        self.config = self.settings.indicator_config
        # OI History: symbol -> list of (timestamp, OptionChainAnalysis)
        self.oi_history: dict[str, list[tuple[datetime, Any]]] = {}
        
        self.squeeze_targets: dict[str, float] = {}
        self._load_squeeze_targets()

    def _load_squeeze_targets(self):
        from trade_system.infrastructure.database.connection import get_engine
        from trade_system.infrastructure.database.repository import get_latest_consolidation_watchlist
        from sqlalchemy.orm import Session
        
        try:
            engine = get_engine()
            with Session(engine) as session:
                stocks = get_latest_consolidation_watchlist(session)
                for item in stocks:
                    self.squeeze_targets[item["symbol"]] = item["resistance"]
            LOGGER.info(f"Loaded {len(self.squeeze_targets)} squeeze targets from database.")
        except Exception as e:
            LOGGER.error(f"Failed to load squeeze targets from database: {e}")

    def monitor_squeeze_breakout(self, symbol: str, df: pd.DataFrame) -> None:
        if symbol not in self.squeeze_targets or df.empty:
            return
            
        resistance = self.squeeze_targets[symbol]
        latest = df.iloc[-1]
        price = float(latest["close"])
        
        if price > resistance:
            # Check volume
            if len(df) >= 20:
                vol_ma = df["volume"].rolling(20).mean().iloc[-1]
                vol_surge = float(latest["volume"]) / float(vol_ma) if vol_ma > 0 else 1.0
                if vol_surge >= 1.5:  # require 1.5x volume surge for breakout
                    self.alert_squeeze_breakout(symbol, price, resistance, vol_surge)
                    # Remove from targets to prevent spamming
                    del self.squeeze_targets[symbol]

    def monitor_technical_extremes(self, symbol: str, df: pd.DataFrame) -> None:
        """Analyze latest bar for RSI extremes and volume surges."""
        if df.empty: return
        
        latest = df.iloc[-1]
        price = float(latest["close"])
        short_sym = symbol.split(':')[-1].replace('-INDEX', '')

        # 1. OB/OS Detection
        rsi_val = latest.get("rsi")
        if rsi_val is None and len(df) >= 15:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi_val = 100 - (100 / (1 + rs.iloc[-1]))

        if rsi_val:
            if rsi_val >= self.config.rsi_overbought:
                self._debounce_send(symbol, "OB", f"⚠️ <b>{short_sym} Overbought</b>\nRSI: {rsi_val:.1f}\nPrice: ₹{price:.2f}\n<i>Potential reversal zone.</i>")
            elif rsi_val <= self.config.rsi_oversold:
                self._debounce_send(symbol, "OS", f"🛡️ <b>{short_sym} Oversold</b>\nRSI: {rsi_val:.1f}\nPrice: ₹{price:.2f}\n<i>Potential bounce zone.</i>")

        # 2. Unusual Volume Detection
        if len(df) >= 20:
            vol_ma = df["volume"].rolling(20).mean().iloc[-1]
            vol_surge = latest["volume"] / vol_ma if vol_ma > 0 else 1.0
            if vol_surge >= self.config.volume_surge_multiplier:
                self._debounce_send(symbol, "VOL", f"🚨 <b>{short_sym} Abnormal Volume</b>\nSurge: <b>{vol_surge:.1f}x</b> average\nPrice: ₹{price:.2f}\n<i>Institutional interest detected.</i>")

    def process_smc_signal(self, symbol: str, signal: dict[str, Any]) -> None:
        """Alert on Smart Money Concept structural breaks."""
        short_sym = symbol.split(':')[-1].replace('-INDEX', '')
        direction = int(signal.get("direction", 0))
        sig_name = signal.get("signal_name", "SMC Alert")
        action = "BUY CALL" if direction == 1 else "BUY PUT"
        color = "🟦" if direction == 1 else "🟧"
        
        message = (
            f"{color} <b>{short_sym} SMC Alert</b>\n"
            f"Signal: <b>{sig_name}</b>\n"
            f"Price: ₹{float(signal.get('close', 0)):.2f}\n"
            f"Note: Institutional structure break detected.\n"
            f"Action: <b>{action}</b>"
        )
        self._debounce_send(symbol, f"SMC_{sig_name}", message)

    def alert_retest(self, symbol: str, level_name: str, price: float, log_only: bool = False):
        """Alert when price retests a major technical level."""
        short_sym = symbol.split(':')[-1].replace('-INDEX', '')
        msg = f"🧭 <b>{short_sym} Level Retest</b>\nLevel: <b>{level_name}</b>\nPrice: ₹{price:.2f}\n<i>Institutional re-entry zone.</i>"
        self._debounce_send(symbol, f"RETEST_{level_name}", msg, log_only=log_only)

    def alert_mwpl_trigger(self, symbol: str, setup_type: str, price: float, mwpl_pct: float):
        """High-priority alert for institutional squeezes."""
        short_sym = symbol.split(':')[-1].replace('-INDEX', '')
        icon = "🚀" if setup_type == "SQUEEZE" else "⚠️"
        message = (
            f"{icon} <b>{short_sym} MWPL {setup_type}</b>\n"
            f"Position Limit: <b>{mwpl_pct:.1f}%</b>\n"
            f"Trigger: Price crossed VWAP with high institutional heat.\n"
            f"Price: ₹{price:.2f}\n"
            f"Action: <b>WATCH FOR EXPLOSIVE MOVE</b>"
        )
        # 30-min debounce for these explosive alerts
        self._debounce_send(symbol, f"MWPL_{setup_type}", message, custom_debounce=1800)

    def alert_gap_and_go(self, symbol: str, direction: str, price: float, sl: float):
        """High-priority alert for Gap & Go momentum continuation."""
        short_sym = symbol.split(':')[-1].replace('-EQ', '')
        icon = "🚀" if direction == "CALL" else "🩸"
        message = (
            f"{icon} <b>GAP & GO DETECTED: {short_sym}</b>\n"
            f"Direction: <b>{direction}</b>\n"
            f"Trigger Price: ₹{price:.2f}\n"
            f"Stop Loss: ₹{sl:.2f}\n"
            f"<i>Institutional momentum continuation.</i>"
        )
        self._debounce_send(symbol, "GAP_AND_GO", message, custom_debounce=1800)

    def alert_orb(self, symbol: str, direction: str, price: float, sl: float):
        """High-priority alert for 15-minute Opening Range Breakout."""
        short_sym = symbol.split(':')[-1].replace('-EQ', '')
        icon = "💥" if direction == "CALL" else "🌪️"
        message = (
            f"{icon} <b>15M ORB BREAKOUT: {short_sym}</b>\n"
            f"Direction: <b>{direction}</b>\n"
            f"Breakout Price: ₹{price:.2f}\n"
            f"Stop Loss: ₹{sl:.2f}\n"
            f"<i>Institutional range expansion.</i>"
        )
        self._debounce_send(symbol, "ORB", message, custom_debounce=3600)

    def alert_sniper_reversal(self, symbol: str, setup_type: str, price: float, sl: float, narrative: str):
        """High-priority alert for precise institutional reversals."""
        short_sym = symbol.split(':')[-1].replace('-EQ', '')
        icon = "🎯"
        message = (
            f"{icon} <b>SNIPER REVERSAL: {short_sym}</b>\n"
            f"Type: <b>{setup_type}</b>\n"
            f"Entry Price: ₹{price:.2f}\n"
            f"Stop Loss: ₹{sl:.2f}\n"
            f"<i>Forensics: {narrative}</i>"
        )
        # 1-hour debounce for these rare sniper setups
        self._debounce_send(symbol, "SNIPER", message, custom_debounce=3600)

    def alert_zone_proximity(self, symbol: str, name: str, zone_type: str, price: float, level: float, dist_pct: float, log_only: bool = False):
        """Alert if price enters a Supply/Demand zone buffer."""
        short_sym = symbol.split(':')[-1].replace('-INDEX', '')
        icon = "🔵" if zone_type == "Support" else "🔴"
        message = (
            f"{icon} <b>{short_sym} Near Zone</b>\n"
            f"Price is near <b>{name}</b> ({zone_type})\n"
            f"Current: ₹{price:.2f}\n"
            f"Level: ₹{level:.2f} ({dist_pct*100:.2f}% dist)"
        )
        self._debounce_send(symbol, f"ZONE_{name}", message, custom_debounce=300, log_only=log_only)

    def alert_squeeze_breakout(self, symbol: str, price: float, resistance: float, vol_surge: float):
        """High-priority alert for volatility contraction (squeeze) breakouts."""
        short_sym = symbol.split(':')[-1].replace('-EQ', '')
        message = (
            f"🚨 <b>SQUEEZE BREAKOUT: {short_sym}</b>\n"
            f"Price crossed Resistance: ₹{resistance:.2f}\n"
            f"Current Price: ₹{price:.2f}\n"
            f"Volume Surge: <b>{vol_surge:.1f}x</b> average\n"
            f"<i>Explosive momentum likely from contraction zone.</i>"
        )
        self._debounce_send(symbol, "SQUEEZE_BREAKOUT", message, custom_debounce=3600)


    def monitor_option_chain_changes(self, symbol: str, analysis: Any):
        """
        Check for sudden changes in Option Chain (15-min window).
        Always logs, but sends Telegram if shifts are significant.
        """
        now = datetime.now()
        if symbol not in self.oi_history:
            self.oi_history[symbol] = []
        
        self.oi_history[symbol].append((now, analysis))
        self.oi_history[symbol] = [item for item in self.oi_history[symbol] if (now - item[0]).total_seconds() <= 1800]
        
        past_15 = None
        for ts, old_analysis in reversed(self.oi_history[symbol]):
            if 13 * 60 <= (now - ts).total_seconds() <= 17 * 60:
                past_15 = old_analysis
                break
        
        if not past_15: return

        short_sym = symbol.split(':')[-1].replace('-INDEX', '')
        alerts = []
        try:
            def _get(obj, key):
                if isinstance(obj, dict): return obj.get(key)
                return getattr(obj, key, None)

            curr_pcr = _get(analysis, 'pcr')
            prev_pcr = _get(past_15, 'pcr')
            if curr_pcr is not None and prev_pcr is not None:
                pcr_delta = curr_pcr - prev_pcr
                if abs(pcr_delta) >= 0.08:
                    direction = "Bullish" if pcr_delta > 0 else "Bearish"
                    alerts.append(f"• PCR: {direction} {pcr_delta:+.2f} (Now: {curr_pcr:.2f})")

            curr_mp = _get(analysis, 'max_pain')
            prev_mp = _get(past_15, 'max_pain')
            if curr_mp and prev_mp and curr_mp != prev_mp:
                alerts.append(f"• Max Pain Shift: {curr_mp - prev_mp:+.0f} pts")

            # Speculative Heat
            try:
                inst = _get(analysis, 'institutional')
                if inst:
                    atm_data = inst.get('zones', {}).get('ATM', {})
                    ce_voi = atm_data.get('CE', {}).get('voi_ratio', 0)
                    pe_voi = atm_data.get('PE', {}).get('voi_ratio', 0)
                    max_voi = max(ce_voi, pe_voi)
                    if max_voi >= 10.0:
                        side = "Calls" if ce_voi > pe_voi else "Puts"
                        alerts.append(f"• Speculative Heat: {side} VOI {max_voi:.1f}x")
            except: pass

            # Always log these thoughts for EOD analysis
            if alerts:
                from trade_system.infrastructure.database.connection import get_engine
                from trade_system.infrastructure.database.repository import log_agent_thought
                from sqlalchemy.orm import Session
                with Session(get_engine()) as session:
                    log_agent_thought(session, "OptionChainAgent", f"Institutional shift: {', '.join(alerts)}", symbol, "FORENSICS")

            # Send Telegram if significant
            if alerts:
                msg = f"📊 <b>{short_sym} Sudden OI Change</b>\n15-min forensics:\n\n" + "\n".join(alerts)
                self._debounce_send(symbol, "SUDDEN_OI", msg, custom_debounce=900)
        except Exception as e:
            LOGGER.debug(f"OI Change detection error: {e}")

    def _debounce_send(self, symbol: str, alert_type: str, message: str, custom_debounce: int | None = None, log_only: bool = False) -> None:
        """Centralized dispatcher for alerts. Handles Telegram and/or DB logging."""
        key = f"{symbol}_{alert_type}"
        now = datetime.now(IST)
        last_sent = self.last_alert_time.get(key)
        debounce = custom_debounce or self.config.alert_debounce_seconds
        
        # Always log to Thought Stream for EOD Analysis
        try:
            from trade_system.infrastructure.database.connection import get_engine
            from trade_system.infrastructure.database.repository import log_agent_thought
            from sqlalchemy.orm import Session
            clean_msg = message.replace('<b>','').replace('</b>','').replace('<i>','').replace('</i>','')
            with Session(get_engine()) as session:
                log_agent_thought(session, "LiveAlertAgent", clean_msg, symbol, alert_type)
        except: pass

        if log_only:
            return

        if last_sent is None or (now - last_sent).total_seconds() > debounce:
            self.notifier.send(message)
            self.last_alert_time[key] = now
            LOGGER.info(f"Live Alert Sent: {key}")
