#!/usr/bin/env python3
"""
F&O Stock Universe - Most liquid stocks for options trading with sector mapping.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Dynamic config file located in the config folder at the project root
config_path = Path(__file__).resolve().parents[4] / "config" / "fo_universe.json"

if not config_path.exists():
    raise FileNotFoundError(f"Required F&O universe configuration not found at: {config_path}")

try:
    with open(config_path, "r") as f:
        FO_METADATA = json.load(f)
except Exception as e:
    logger.error(f"Failed to load fo_universe.json: {e}")
    raise RuntimeError(f"Failed to parse F&O universe config: {e}")

def get_fo_universe() -> list[str]:
    """Return all F&O stock symbols."""
    return list(FO_METADATA.keys())

def get_sector_mapping() -> dict[str, str]:
    """Return the symbol to sector mapping."""
    return FO_METADATA

def get_stocks_by_sector(sector: str) -> list[str]:
    """Return all stocks belonging to a specific sector."""
    return [sym for sym, sec in FO_METADATA.items() if sec == sector]

def get_sector_summary() -> dict[str, int]:
    """Return count of stocks in each sector."""
    summary = {}
    for sec in FO_METADATA.values():
        summary[sec] = summary.get(sec, 0) + 1
    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    print(f"F&O Universe contains {len(FO_METADATA)} stocks across {len(set(FO_METADATA.values()))} sectors.")
    print("\nSector Distribution:")
    for sector, count in get_sector_summary().items():
        print(f"• {sector}: {count}")
