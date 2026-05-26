import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.interfaces.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["auth", *sys.argv[1:]]))
