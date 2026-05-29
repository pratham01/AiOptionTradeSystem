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

import requests
import csv
from datetime import datetime, timedelta

def update_fo_universe_from_nse():
    """Fetches the latest F&O universe from NSE and updates the JSON config."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        res = requests.get("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv", headers=headers, timeout=10)
        
        if res.status_code != 200:
            logger.error(f"Failed to fetch FO universe from NSE. Status: {res.status_code}")
            return False
            
        lines = res.text.split('\n')
        symbols = []
        reader = csv.reader(lines)
        header_skipped = False
        for row in reader:
            if not header_skipped:
                if len(row) > 1 and ('UNDERLYING' in row[0].upper() or 'SYMBOL' in row[1].upper()):
                    header_skipped = True
                continue
            if len(row) > 1:
                sym = row[1].strip()
                if sym and not sym.startswith('Symbol') and sym != 'SYMBOL':
                    symbols.append(sym)
                    
        indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50']
        stock_symbols = [s for s in symbols if s not in indices]
        
        if len(stock_symbols) < 150:
            logger.error("Parsed too few symbols, aborting update.")
            return False
            
        # Load existing json
        with open(config_path, 'r') as f:
            old_data = json.load(f)
            
        new_data = {}
        for s in stock_symbols:
            fyers_sym = f"NSE:{s}-EQ"
            if fyers_sym in old_data:
                new_data[fyers_sym] = old_data[fyers_sym]
            else:
                new_data[fyers_sym] = "UNKNOWN"
                
        with open(config_path, 'w') as f:
            json.dump(new_data, f, indent=4)
            
        logger.info(f"Dynamically updated F&O universe. Total stocks: {len(stock_symbols)}")
        
        # Reload FO_METADATA
        global FO_METADATA
        FO_METADATA = new_data
        
        return True
    except Exception as e:
        logger.error(f"Exception during FO universe update: {e}")
        return False
