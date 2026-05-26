"""Signals module for signal generation and management."""

from trade_system.application.signals.generator import SignalRegistry, register_signal_generator
from trade_system.application.signals.manager import SignalManager

__all__ = ["SignalRegistry", "register_signal_generator", "SignalManager"]
