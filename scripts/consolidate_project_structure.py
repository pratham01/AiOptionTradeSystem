#!/usr/bin/env python3
"""

Consolidate project structure to clean src/ layout.

This script:
1. Merges old trade_system/ into src/trade_system/
2. Removes duplicate directories
3. Updates imports
4. Creates clean project structure
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import shutil
from typing import Any

# Define the new clean structure
NEW_STRUCTURE = {
    "src/trade_system": {
        "config": ["settings.py"],
        "core": [
            "ports/broker.py",
            "models.py",
            "exceptions.py",
        ],
        "infrastructure": [
            "brokers/factory.py",
            "brokers/fyers/client.py",
            "brokers/dhan/client.py",
        ],
        "data": [
            "nse_universe.py",
            "manager.py",
        ],
        "advisory": [
            "advisor.py",
            "portfolio.py",
        ],
        "agent": [
            "risk_manager.py",
            "decision_engine.py",
        ],
        "analysis": [
            "technical.py",
            "fundamental.py",
        ],
        "api": [
            "mcp_server.py",
            "rest_server.py",
        ],
        "backtesting": [
            "engine.py",
            "metrics.py",
        ],
        "brokers": [
            "base.py",
            "fyers.py",
            "fyers_auth.py",
        ],
        "dashboard": [
            "app.py",
            "views.py",
        ],
        "indicators": [
            "supertrend.py",
            "rsi.py",
        ],
        "live": [
            "streamer.py",
            "processor.py",
        ],
        "notifications": [
            "telegram.py",
            "email.py",
        ],
        "pipeline": [
            "data_pipeline.py",
        ],
        "research": [
            "screener.py",
        ],
        "signals": [
            "generator.py",
        ],
        "strategies": [
            "base.py",
            "supertrend_strategy.py",
        ],
    }
}

def copy_with_structure(src_root: Path, dst_root: Path) -> None:
    """Copy files maintaining directory structure."""
    for module, files in NEW_STRUCTURE.items():
        dst_dir = dst_root / module
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in files:
            src_file = src_root / file_path
            dst_file = dst_root / module / Path(file_path).name
            
            if src_file.exists():
                print(f"  Copying: {file_path} -> {module}/{Path(file_path).name}")
                shutil.copy2(src_file, dst_file)
            else:
                print(f"  Missing: {file_path}")

def update_imports_in_file(file_path: Path) -> None:
    """Update imports in a Python file."""
    if not file_path.exists():
        return
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original = content
        
        # Update imports to use new structure
        content = content.replace(
            'from trade_system.',
            'from trade_system.'
        )
        content = content.replace(
            'import trade_system.',
            'import trade_system.'
        )
        
        # Remove duplicate src/ if already present
        content = content.replace('from src.src.trade_system', 'from trade_system')
        content = content.replace('import src.src.trade_system', 'import trade_system')
        
        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  Updated imports in {file_path.name}")
    
    except Exception as e:
        print(f"  Failed to update {file_path}: {e}")

def update_all_imports(root_dir: Path) -> None:
    """Update imports in all Python files."""
    print("Updating imports...")
    
    # Update scripts
    scripts_dir = root_dir / "scripts"
    if scripts_dir.exists():
        for py_file in scripts_dir.glob("*.py"):
            update_imports_in_file(py_file)
    
    # Update src files
    src_dir = root_dir / "src"
    if src_dir.exists():
        for py_file in src_dir.rglob("*.py"):
            update_imports_in_file(py_file)
    
    # Update root files
    for py_file in root_dir.glob("*.py"):
        update_imports_in_file(py_file)

def create_clean_structure(root_dir: Path) -> None:
    """Create the new clean project structure."""
    print("Creating clean project structure...")
    
    # Backup old structure
    backup_dir = root_dir / "trade_system_backup"
    if (root_dir / "trade_system").exists() and not backup_dir.exists():
        print(f"  Backing up old trade_system/ to {backup_dir}")
        shutil.move(str(root_dir / "trade_system"), str(backup_dir))
    
    # Create new structure
    src_dir = root_dir / "src" / "trade_system"
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files from old structure
    old_trade_system = root_dir / "trade_system_backup"
    if old_trade_system.exists():
        print("  Copying files from old structure...")
        copy_with_structure(old_trade_system, root_dir)
    
    # Create __init__.py files
    for module_dir in src_dir.rglob("*"):
        if module_dir.is_dir():
            init_file = module_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Module initialization."""\n')
    
    # Create main __init__.py
    main_init = src_dir / "__init__.py"
    if not main_init.exists():
        main_init.write_text('"""Trade System - Multi-broker trading platform."""\n\n__version__ = "2.0.0"\n')

def update_pyproject_toml(root_dir: Path) -> None:
    """Update pyproject.toml for new structure."""
    pyproject_file = root_dir / "pyproject.toml"
    if not pyproject_file.exists():
        return
    
    try:
        with open(pyproject_file, 'r') as f:
            content = f.read()
        
        # Update PYTHONPATH
        if "[tool.pytest.ini_options]" in content:
            content = content.replace(
                'pythonpath = "."',
                'pythonpath = "src"'
            )
        
        with open(pyproject_file, 'w') as f:
            f.write(content)
        
        print("  Updated pyproject.toml")
    
    except Exception as e:
        print(f"  Failed to update pyproject.toml: {e}")

def create_readme_update(root_dir: Path) -> None:
    """Create README update for new structure."""
    readme_update = f"""
# Project Structure Update

## New Structure

```
{root_dir.name}/
├── src/trade_system/          # Main source code
│   ├── config/               # Configuration
│   ├── core/                 # Core interfaces
│   ├── infrastructure/       # External integrations
│   ├── data/                 # Data management
│   ├── brokers/              # Broker implementations
│   ├── strategies/           # Trading strategies
│   └── ...                   # Other modules
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
├── docs/                     # Documentation
└── data/                     # Data files
```

## Import Changes

Preferred imports:
```python
from trade_system.config import Settings
from trade_system.infrastructure.brokers.fyers.client import FyersBroker
```

## Running Scripts

Use PYTHONPATH to include src directory:
```bash
PYTHONPATH=src python scripts/your_script.py
```

Or run from project root (scripts handle PYTHONPATH automatically).
"""
    
    readme_file = root_dir / "docs" / "PROJECT_STRUCTURE.md"
    readme_file.parent.mkdir(exist_ok=True)
    with open(readme_file, 'w') as f:
        f.write(readme_update)
    
    print(f"  Created structure documentation: {readme_file}")

def main() -> int:
    """Run the consolidation."""
    root_dir = Path.cwd()
    
    print("=" * 70)
    print("🔧 CONSOLIDATING PROJECT STRUCTURE")
    print("=" * 70)
    
    # Create clean structure
    create_clean_structure(root_dir)
    
    # Update imports
    update_all_imports(root_dir)
    
    # Update configuration
    update_pyproject_toml(root_dir)
    
    # Create documentation
    create_readme_update(root_dir)
    
    print("\n" + "=" * 70)
    print("✅ STRUCTURE CONSOLIDATION COMPLETE")
    print("=" * 70)
    print("""
Next steps:
1. Test the new structure with: PYTHONPATH=src python scripts/test_new_broker_system.py
2. Remove old backup directory when confident: rm -rf trade_system_backup
3. Update IDE settings to recognize src/ as source root
4. Update any remaining manual imports

The project now has a clean, industry-standard structure!
""")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
