"""Configurable Telegram notification template manager and formatter."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from trade_system.shared.exceptions import TemplateRenderError

LOGGER = logging.getLogger(__name__)

# Built-in fallback template dictionary in case yaml is missing
_DEFAULT_CONFIG: Dict[str, Any] = {
    "defaults": {
        "parse_mode": "HTML",
        "cooldown_seconds": 60,
        "enable_fo_alerts": False,
        "enable_index_alerts": True,
    },
    "strategies": {
        "supertrend_flip": {
            "enabled": True,
            "channel": "main",
            "cooldown_seconds": 120,
            "template": (
                "{color} <b>{symbol_short} {time}</b>\n"
                "Direction changed to <b>{direction}</b> ({timeframe}m)\n"
                "15m Trend: <b>{trend_15m}</b>\n"
                "Close: ₹{close:.2f}\n"
                "Supertrend: ₹{supertrend:.2f}\n"
                "Confluence: <b>{confluence}</b>\n"
                "Action: <b>{action}</b>{strikes_block}"
            ),
        },
        "supertrend_confirmed": {
            "enabled": True,
            "channel": "confirmed",
            "cooldown_seconds": 180,
            "template": (
                "{color} <b>⚡ ST FLIP — {symbol_short} {time}</b>\n"
                "Timeframe: <b>{timeframe}m Supertrend</b>\n"
                "New Direction: <b>{direction_verbose}</b>\n"
                "15m Trend Alignment: <b>{trend_15m}</b>\n"
                "Close: ₹{close:.2f}  |  ST Level: ₹{supertrend:.2f}\n"
                "Confluence: <b>{confluence}</b>\n"
                "Suggested Action: <b>{action}</b>{strikes_block}"
            ),
        },
        "orb_breakout": {
            "enabled": True,
            "channel": "main",
            "cooldown_seconds": 300,
            "template": (
                "🚀 <b>ORB BREAKOUT — {symbol_short} {time}</b>\n"
                "Range: ₹{orb_low:.2f} – ₹{orb_high:.2f} (15m ORB)\n"
                "Breakout Price: ₹{price:.2f} ({direction})\n"
                "Volume Surge: <b>{volume_surge:.1f}x</b>\n"
                "VWAP Status: <b>{vwap_status}</b>\n"
                "Action: <b>{action}</b>"
            ),
        },
        "gamma_blast": {
            "enabled": True,
            "channel": "main",
            "cooldown_seconds": 180,
            "template": (
                "⚡ <b>0DTE GAMMA BLAST — {symbol_short}</b>\n"
                "Strike: <b>{strike_name}</b> @ ₹{option_ltp:.2f}\n"
                "Underlying: ₹{spot_price:.2f}\n"
                "Gamma Momentum: <b>{momentum_score:.1f}/100</b>\n"
                "Suggested Entry: ₹{entry:.2f} | SL: ₹{stop_loss:.2f} | TGT: ₹{target:.2f}\n"
                "Action: <b>{action}</b>"
            ),
        },
        "sniper_reversal": {
            "enabled": True,
            "channel": "main",
            "cooldown_seconds": 300,
            "template": (
                "🎯 <b>SNIPER REVERSAL — {symbol_short} {time}</b>\n"
                "Zone: <b>{zone_name}</b> @ ₹{zone_level:.2f}\n"
                "Exhaustion Wick: <b>{wick_pct:.0%}</b>\n"
                "Price: ₹{price:.2f} ({direction})\n"
                "Action: <b>{action}</b>"
            ),
        },
        "intraday_edge": {
            "enabled": True,
            "channel": "main",
            "cooldown_seconds": 600,
            "min_score": 75.0,
            "template": (
                "🏆 <b>HIGH CONFLUENCE EDGE — {symbol_short}</b>\n"
                "Edge Score: <b>{final_score:.1f}/100</b> ({direction})\n"
                "Sector Momentum: <b>{sector}</b> ({sector_rank})\n"
                "Key Trigger: <b>{trigger_status}</b>\n"
                "Entry: ₹{entry:.2f} | SL: ₹{stop_loss:.2f} | TGT: ₹{target:.2f}"
            ),
        },
        "premarket_report": {
            "enabled": True,
            "channel": "main",
            "template": (
                "📊 <b>Premarket Report — {symbol_short}</b>\n\n"
                "{summary}\n"
                "{zone_info}\n\n"
                "🧭 <b>Current Trend:</b> {trend_icon}"
            ),
        },
        "eod_summary": {
            "enabled": True,
            "channel": "main",
            "template": (
                "🏁 <b>EOD Session Summary — {date}</b>\n\n"
                "📈 Total Signals: <b>{total_signals}</b>\n"
                "🎯 Target Hits: <b>{target_hits}</b> | 🛑 Stop Losses: <b>{stop_losses}</b>\n"
                "💰 Day PnL: <b>{day_pnl:+.2f} pts</b>"
            ),
        },
    }
}


class TelegramTemplateManager:
    """Thread-safe manager for configurable Telegram alert templates and routing."""
    _instance: Optional[TelegramTemplateManager] = None
    _lock = threading.Lock()

    def __new__(cls, config_path: str | Path | None = None) -> TelegramTemplateManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: str | Path | None = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self.config_path = Path(config_path) if config_path else Path("config/alerts_config.yaml")
        self._config: Dict[str, Any] = {}
        self._last_alert_times: Dict[str, Dict[str, datetime]] = {}
        self._state_lock = threading.Lock()
        self.reload_config()
        self._initialized = True

    def reload_config(self) -> None:
        """Reload configuration from disk or fall back to defaults."""
        with self._state_lock:
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        self._config = yaml.safe_load(f) or {}
                    LOGGER.info("Loaded Telegram alert configuration from %s", self.config_path)
                except Exception as exc:
                    LOGGER.warning("Failed to parse %s: %s. Using default templates.", self.config_path, exc)
                    self._config = _DEFAULT_CONFIG
            else:
                LOGGER.info("Alert config %s not found. Using default templates.", self.config_path)
                self._config = _DEFAULT_CONFIG

    def is_strategy_enabled(self, strategy_name: str, is_index: bool = True) -> bool:
        """Check if alerts for a given strategy and instrument type are enabled."""
        defaults = self._config.get("defaults", {})
        if is_index and not defaults.get("enable_index_alerts", True):
            return False
        if not is_index and not defaults.get("enable_fo_alerts", False):
            return False

        strat_cfg = self._config.get("strategies", {}).get(strategy_name, {})
        return bool(strat_cfg.get("enabled", True))

    def should_debounce(self, strategy_name: str, symbol: str, current_time: datetime | None = None) -> bool:
        """Check if an alert should be suppressed due to active cooldown."""
        strat_cfg = self._config.get("strategies", {}).get(strategy_name, {})
        cooldown = strat_cfg.get("cooldown_seconds", self._config.get("defaults", {}).get("cooldown_seconds", 60))
        now = current_time or datetime.now()

        with self._state_lock:
            strat_history = self._last_alert_times.setdefault(strategy_name, {})
            last_time = strat_history.get(symbol)
            if last_time is not None:
                elapsed = (now - last_time).total_seconds()
                if elapsed < cooldown:
                    return True  # Should debounce
            strat_history[symbol] = now
            return False

    def get_channel(self, strategy_name: str) -> str:
        """Get the target channel routing for this strategy ('main' or 'confirmed')."""
        strat_cfg = self._config.get("strategies", {}).get(strategy_name, {})
        return str(strat_cfg.get("channel", "main"))

    def format_message(self, strategy_name: str, context: Dict[str, Any]) -> str:
        """Render the message template for the given strategy and context parameters."""
        strat_cfg = self._config.get("strategies", {}).get(strategy_name)
        if not strat_cfg or "template" not in strat_cfg:
            fallback = _DEFAULT_CONFIG["strategies"].get(strategy_name, {}).get("template")
            if not fallback:
                raise TemplateRenderError(f"No template registered for strategy: {strategy_name}")
            template_str = fallback
        else:
            template_str = strat_cfg["template"]

        # Ensure default keys exist to prevent KeyError
        safe_ctx = {
            "symbol_short": context.get("symbol_short", context.get("symbol", "")),
            "color": context.get("color", "⚪"),
            "time": context.get("time", datetime.now().strftime("%H:%M")),
            "timeframe": context.get("timeframe", 3),
            "direction": context.get("direction", "NEUTRAL"),
            "direction_verbose": context.get("direction_verbose", context.get("direction", "")),
            "trend_15m": context.get("trend_15m", "UNKNOWN"),
            "close": float(context.get("close", 0.0)),
            "supertrend": float(context.get("supertrend", 0.0)),
            "confluence": context.get("confluence", "Standard Signal"),
            "action": context.get("action", "HOLD"),
            "strikes_block": context.get("strikes_block", ""),
            "orb_high": float(context.get("orb_high", 0.0)),
            "orb_low": float(context.get("orb_low", 0.0)),
            "price": float(context.get("price", context.get("close", 0.0))),
            "volume_surge": float(context.get("volume_surge", 1.0)),
            "vwap_status": context.get("vwap_status", "Neutral"),
            "strike_name": context.get("strike_name", ""),
            "option_ltp": float(context.get("option_ltp", 0.0)),
            "spot_price": float(context.get("spot_price", context.get("close", 0.0))),
            "momentum_score": float(context.get("momentum_score", 0.0)),
            "entry": float(context.get("entry", 0.0)),
            "stop_loss": float(context.get("stop_loss", 0.0)),
            "target": float(context.get("target", 0.0)),
            "zone_name": context.get("zone_name", ""),
            "zone_level": float(context.get("zone_level", 0.0)),
            "wick_pct": float(context.get("wick_pct", 0.0)),
            "final_score": float(context.get("final_score", 0.0)),
            "sector": context.get("sector", ""),
            "sector_rank": context.get("sector_rank", ""),
            "trigger_status": context.get("trigger_status", ""),
            "summary": context.get("summary", ""),
            "zone_info": context.get("zone_info", ""),
            "trend_icon": context.get("trend_icon", "⚪"),
            "date": context.get("date", str(date.today())),
            "total_signals": context.get("total_signals", 0),
            "target_hits": context.get("target_hits", 0),
            "stop_losses": context.get("stop_losses", 0),
            "day_pnl": float(context.get("day_pnl", 0.0)),
        }
        safe_ctx.update(context)

        try:
            return template_str.strip().format(**safe_ctx)
        except Exception as exc:
            LOGGER.error("Failed to render template for %s: %s", strategy_name, exc)
            raise TemplateRenderError(f"Error rendering {strategy_name} template: {exc}") from exc
