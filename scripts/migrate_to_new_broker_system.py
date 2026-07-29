#!/usr/bin/env python3
"""

Migration script to update existing code to use new broker system.

This script:
1. Updates imports in existing scripts
2. Creates backward compatibility shims
3. Updates configuration files
4. Tests the new system
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import json
import logging
import shutil
from typing import Any

LOGGER = logging.getLogger(__name__)

def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a Python file to use new broker system."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        original = content

        # Update imports
        content = content.replace(
            'from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker',
            'from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker'
        )
        content = content.replace(
            'from trade_system.shared.config.settings import Settings',
            'from trade_system.shared.config import Settings'
        )
        content = content.replace(
            'from trade_system.domains.market_data.infrastructure.data.nse_universe import NSE_UNIVERSE',
            'from trade_system.domains.market_data.infrastructure.data.nse_universe import NSE_UNIVERSE'
        )

        # Update broker instantiation
        if 'FyersBroker(' in content and 'get_broker_manager' not in content:
            # Add broker manager import
            if 'from trade_system.domains.trading.infrastructure.brokers.factory import' not in content:
                content = content.replace(
                    'from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker',
                    'from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker\nfrom trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager'
                )

        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            LOGGER.info(f"Updated imports in {file_path}")
            return True

    except Exception as e:
        LOGGER.error(f"Failed to update {file_path}: {e}")

    return False

def create_backward_compatibility_shim() -> None:
    """Create backward compatibility shim for old imports."""
    shim_path = Path("trade_system/__init__.py")
    
    shim_content = '''"""
Backward compatibility shim for existing imports.

This file redirects old imports to new locations while maintaining compatibility.
"""

# Re-export for backward compatibility
from trade_system.shared.config import Settings
from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker
from trade_system.domains.market_data.infrastructure.data.nse_universe import NSE_UNIVERSE

# Legacy aliases
__all__ = [
    "Settings",
    "FyersBroker", 
    "NSE_UNIVERSE",
]
'''

    shim_path.parent.mkdir(exist_ok=True)
    with open(shim_path, 'w') as f:
        f.write(shim_content)
    
    LOGGER.info(f"Created backward compatibility shim: {shim_path}")

def update_environment_file() -> None:
    """Add Dhan configuration to environment file."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    # Read existing env file
    env_vars = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value

    # Add new Dhan variables
    new_vars = {
        "PRIMARY_BROKER": "fyers",
        "BACKUP_BROKER": "dhan",
        "DHAN_CLIENT_ID": "",
        "DHAN_API_KEY": "",
        "DHAN_ACCESS_TOKEN": "",
        "DHAN_ENABLED": "false",
        "DHAN_PRIORITY": "2",
        "FYERS_ENABLED": "true",
        "FYERS_PRIORITY": "1",
    }

    # Update with new vars (don't overwrite existing)
    for key, value in new_vars.items():
        if key not in env_vars:
            env_vars[key] = value

    # Write updated env file
    with open(env_file, 'w') as f:
        f.write("# Trading System Configuration\n")
        f.write("# =============================\n\n")
        
        # Fyers config
        f.write("# Fyers Configuration\n")
        f.write(f"FYERS_CLIENT_ID={env_vars.get('FYERS_CLIENT_ID', '')}\n")
        f.write(f"FYERS_USER_ID={env_vars.get('FYERS_USER_ID', '')}\n")
        f.write(f"FYERS_SECRET_KEY={env_vars.get('FYERS_SECRET_KEY', '')}\n")
        f.write(f"FYERS_ACCESS_TOKEN={env_vars.get('FYERS_ACCESS_TOKEN', '')}\n")
        f.write(f"FYERS_ENABLED={env_vars.get('FYERS_ENABLED', 'true')}\n")
        f.write(f"FYERS_PRIORITY={env_vars.get('FYERS_PRIORITY', '1')}\n\n")
        
        # Dhan config
        f.write("# Dhan Configuration (Backup)\n")
        f.write(f"DHAN_CLIENT_ID={env_vars.get('DHAN_CLIENT_ID', '')}\n")
        f.write(f"DHAN_API_KEY={env_vars.get('DHAN_API_KEY', '')}\n")
        f.write(f"DHAN_ACCESS_TOKEN={env_vars.get('DHAN_ACCESS_TOKEN', '')}\n")
        f.write(f"DHAN_ENABLED={env_vars.get('DHAN_ENABLED', 'false')}\n")
        f.write(f"DHAN_PRIORITY={env_vars.get('DHAN_PRIORITY', '2')}\n\n")
        
        # Broker selection
        f.write("# Broker Selection\n")
        f.write(f"PRIMARY_BROKER={env_vars.get('PRIMARY_BROKER', 'fyers')}\n")
        f.write(f"BACKUP_BROKER={env_vars.get('BACKUP_BROKER', 'dhan')}\n\n")
        
        # Other existing vars
        for key, value in env_vars.items():
            if not any(key.startswith(prefix) for prefix in ["FYERS_", "DHAN_", "PRIMARY_", "BACKUP_"]):
                f.write(f"{key}={value}\n")

    LOGGER.info("Updated .env file with new broker configuration")

def test_new_broker_system() -> bool:
    """Test the new broker system."""
    try:
        # Test imports
        from trade_system.shared.config import Settings
        from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
        from trade_system.domains.trading.infrastructure.brokers.fyers.client import FyersBroker
        from trade_system.domains.trading.infrastructure.brokers.dhan.client import DhanBroker

        # Test settings
        settings = Settings.load()
        LOGGER.info(f"Loaded settings: Primary={settings.primary_broker}, Backup={settings.backup_broker}")

        # Test broker manager
        manager = get_broker_manager(settings)
        LOGGER.info(f"Broker manager created: {manager}")

        # Test health status
        health = manager.get_health_status()
        LOGGER.info(f"Broker health: {health}")

        return True

    except Exception as e:
        LOGGER.error(f"Failed to test new broker system: {e}")
        return False

def main() -> int:
    """Run migration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    LOGGER.info("=" * 70)
    LOGGER.info("🔄 MIGRATING TO NEW BROKER SYSTEM")
    LOGGER.info("=" * 70)

    # 1. Create backward compatibility shim
    LOGGER.info("\n1. Creating backward compatibility shim...")
    create_backward_compatibility_shim()

    # 2. Update scripts
    LOGGER.info("\n2. Updating scripts...")
    scripts_dir = Path("scripts")
    updated_count = 0

    for script_file in scripts_dir.glob("*.py"):
        if update_imports_in_file(script_file):
            updated_count += 1

    LOGGER.info(f"Updated {updated_count} script files")

    # 3. Update environment
    LOGGER.info("\n3. Updating environment configuration...")
    update_environment_file()

    # 4. Test new system
    LOGGER.info("\n4. Testing new broker system...")
    if test_new_broker_system():
        LOGGER.info("✅ New broker system test passed!")
    else:
        LOGGER.error("❌ New broker system test failed!")
        return 1

    # 5. Create summary
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("MIGRATION COMPLETE")
    LOGGER.info("=" * 70)
    LOGGER.info("""
Next steps:
1. Configure Dhan credentials in .env file (optional)
2. Test your existing scripts to ensure they work
3. Gradually update scripts to use broker manager for failover support
4. Remove old trade_system/ directory once fully migrated

Example usage with new system:
    from trade_system.domains.trading.infrastructure.brokers.factory import get_broker_manager
    
    manager = get_broker_manager()
    quotes = manager.get_quotes(["NSE:RELIANCE-EQ"])  # Auto failover!
""")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
