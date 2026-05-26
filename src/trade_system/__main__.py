"""
Main entry point for the trade_system package.
Redirects to the CLI interface.
"""

import sys
from trade_system.interfaces.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
