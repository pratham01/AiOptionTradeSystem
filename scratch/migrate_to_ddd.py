import os
import shutil
import re

SRC_DIR = "/Users/pratham/aitrade/trade_system_v2/src/trade_system"

DOMAINS = ["trading", "market_data", "strategy", "analysis", "advisory"]
SHARED = "shared"

def create_dirs():
    for d in DOMAINS:
        os.makedirs(os.path.join(SRC_DIR, "domains", d, "domain"), exist_ok=True)
        os.makedirs(os.path.join(SRC_DIR, "domains", d, "application"), exist_ok=True)
        os.makedirs(os.path.join(SRC_DIR, "domains", d, "infrastructure"), exist_ok=True)
        
        # init files
        with open(os.path.join(SRC_DIR, "domains", "__init__.py"), "a") as f: pass
        with open(os.path.join(SRC_DIR, "domains", d, "__init__.py"), "a") as f: pass
        with open(os.path.join(SRC_DIR, "domains", d, "domain", "__init__.py"), "a") as f: pass
        with open(os.path.join(SRC_DIR, "domains", d, "application", "__init__.py"), "a") as f: pass
        with open(os.path.join(SRC_DIR, "domains", d, "infrastructure", "__init__.py"), "a") as f: pass

    os.makedirs(os.path.join(SRC_DIR, SHARED), exist_ok=True)
    with open(os.path.join(SRC_DIR, SHARED, "__init__.py"), "a") as f: pass

def move_dir(src_rel, dst_rel):
    src = os.path.join(SRC_DIR, src_rel)
    dst = os.path.join(SRC_DIR, dst_rel)
    if os.path.exists(src):
        if os.path.exists(dst) and os.path.isdir(dst):
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                    shutil.rmtree(s)
                else:
                    shutil.move(s, d)
            os.rmdir(src)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        print(f"Moved {src_rel} to {dst_rel}")
    else:
        print(f"Warning: {src_rel} not found")

def move_file(src_rel, dst_rel):
    src = os.path.join(SRC_DIR, src_rel)
    dst = os.path.join(SRC_DIR, dst_rel)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"Moved {src_rel} to {dst_rel}")
    else:
        print(f"Warning: {src_rel} not found")

def execute_moves():
    # Shared
    move_dir("config", f"{SHARED}/config")
    move_dir("utils", f"{SHARED}/utils")
    move_dir("core/events", f"{SHARED}/events")
    move_dir("infrastructure/notifications", f"{SHARED}/notifications")
    os.makedirs(os.path.join(SRC_DIR, SHARED, "ports"), exist_ok=True)
    with open(os.path.join(SRC_DIR, SHARED, "ports", "__init__.py"), "a") as f: pass
    move_file("core/ports/repository.py", f"{SHARED}/ports/repository.py")
    move_file("core/ports/notifier.py", f"{SHARED}/ports/notifier.py")
    move_file("core/ports/weights.py", f"{SHARED}/ports/weights.py") # Missing previously

    # Trading Domain
    os.makedirs(os.path.join(SRC_DIR, "domains/trading/domain/models"), exist_ok=True)
    with open(os.path.join(SRC_DIR, "domains/trading/domain/models", "__init__.py"), "a") as f: pass
    move_file("core/models/domain.py", "domains/trading/domain/models/domain.py")
    move_file("core/models/legacy.py", "domains/trading/domain/models/legacy.py")
    
    os.makedirs(os.path.join(SRC_DIR, "domains/trading/domain/ports"), exist_ok=True)
    with open(os.path.join(SRC_DIR, "domains/trading/domain/ports", "__init__.py"), "a") as f: pass
    move_file("core/ports/broker.py", "domains/trading/domain/ports/broker.py")
    
    move_dir("application/execution", "domains/trading/application/execution")
    move_dir("application/pipeline", "domains/trading/application/pipeline")
    move_dir("infrastructure/brokers", "domains/trading/infrastructure/brokers")

    # Market Data Domain
    os.makedirs(os.path.join(SRC_DIR, "domains/market_data/domain/schemas"), exist_ok=True)
    with open(os.path.join(SRC_DIR, "domains/market_data/domain/schemas", "__init__.py"), "a") as f: pass
    move_file("core/schemas/market_data.py", "domains/market_data/domain/schemas/market_data.py")
    
    move_dir("infrastructure/data", "domains/market_data/infrastructure/data")
    move_dir("infrastructure/database", "domains/market_data/infrastructure/database")

    # Strategy Domain
    move_file("core/signals.py", "domains/strategy/domain/signals.py")
    move_dir("application/strategies", "domains/strategy/application/strategies")
    move_dir("application/signals", "domains/strategy/application/signals")
    move_dir("application/indicators", "domains/strategy/application/indicators")

    # Analysis Domain
    move_dir("application/analysis", "domains/analysis/application/analysis")
    move_dir("application/backtesting", "domains/analysis/application/backtesting")
    move_dir("application/evolution", "domains/analysis/application/evolution")
    move_dir("application/research", "domains/analysis/application/research")

    # Advisory Domain
    move_dir("core/agents", "domains/advisory/domain/agents")
    move_dir("application/advisory", "domains/advisory/application/advisory")
    move_dir("application/agent", "domains/advisory/application/agent")
    
    # Clean up empty dirs
    for d in ["core/models", "core/ports", "core/schemas", "core", "application", "infrastructure"]:
        d_path = os.path.join(SRC_DIR, d)
        if os.path.exists(d_path) and not os.listdir(d_path):
            os.rmdir(d_path)
            print(f"Removed empty dir {d}")

def update_imports():
    replacements = [
        # Shared
        (r"trade_system\.config", r"trade_system.shared.config"),
        (r"trade_system\.utils", r"trade_system.shared.utils"),
        (r"trade_system\.core\.events", r"trade_system.shared.events"),
        (r"trade_system\.infrastructure\.notifications", r"trade_system.shared.notifications"),
        (r"trade_system\.core\.ports\.repository", r"trade_system.shared.ports.repository"),
        (r"trade_system\.core\.ports\.notifier", r"trade_system.shared.ports.notifier"),
        (r"trade_system\.core\.ports\.weights", r"trade_system.shared.ports.weights"),
        
        # Trading
        (r"trade_system\.core\.models\.domain", r"trade_system.domains.trading.domain.models.domain"),
        (r"trade_system\.core\.models\.legacy", r"trade_system.domains.trading.domain.models.legacy"),
        (r"trade_system\.core\.ports\.broker", r"trade_system.domains.trading.domain.ports.broker"),
        (r"trade_system\.application\.execution", r"trade_system.domains.trading.application.execution"),
        (r"trade_system\.application\.pipeline", r"trade_system.domains.trading.application.pipeline"),
        (r"trade_system\.infrastructure\.brokers", r"trade_system.domains.trading.infrastructure.brokers"),

        # Market Data
        (r"trade_system\.core\.schemas\.market_data", r"trade_system.domains.market_data.domain.schemas.market_data"),
        (r"trade_system\.infrastructure\.data", r"trade_system.domains.market_data.infrastructure.data"),
        (r"trade_system\.infrastructure\.database", r"trade_system.domains.market_data.infrastructure.database"),

        # Strategy
        (r"trade_system\.core\.signals", r"trade_system.domains.strategy.domain.signals"),
        (r"trade_system\.application\.strategies", r"trade_system.domains.strategy.application.strategies"),
        (r"trade_system\.application\.signals", r"trade_system.domains.strategy.application.signals"),
        (r"trade_system\.application\.indicators", r"trade_system.domains.strategy.application.indicators"),

        # Analysis
        (r"trade_system\.application\.analysis", r"trade_system.domains.analysis.application.analysis"),
        (r"trade_system\.application\.backtesting", r"trade_system.domains.analysis.application.backtesting"),
        (r"trade_system\.application\.evolution", r"trade_system.domains.analysis.application.evolution"),
        (r"trade_system\.application\.research", r"trade_system.domains.analysis.application.research"),

        # Advisory
        (r"trade_system\.core\.agents", r"trade_system.domains.advisory.domain.agents"),
        (r"trade_system\.application\.advisory", r"trade_system.domains.advisory.application.advisory"),
        (r"trade_system\.application\.agent", r"trade_system.domains.advisory.application.agent"),
        
        # Catch-alls for moved roots (if folks imported the root)
        (r"from trade_system\.core import", r"from trade_system.shared import"),
        (r"import trade_system\.core\.", r"import trade_system.shared."),
    ]

    for root, _, files in os.walk("/Users/pratham/aitrade/trade_system_v2"):
        if ".git" in root or "__pycache__" in root or "venv" in root:
            continue
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = content
                for old, new in replacements:
                    new_content = re.sub(old, new, new_content)
                
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated imports in {path}")
            except Exception as e:
                print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    create_dirs()
    execute_moves()
    update_imports()
    print("Migration complete.")
