import re

with open("src/trade_system/interfaces/live/collector.py", "r") as f:
    content = f.read()

# Insert the orchestrator import
if "WorkflowEngine" not in content:
    content = content.replace(
        "from trade_system.interfaces.live.health_server import start_health_server",
        "from trade_system.interfaces.live.health_server import start_health_server\nfrom trade_system.core.orchestration.workflow_engine import WorkflowEngine\nimport asyncio"
    )

with open("src/trade_system/interfaces/live/collector.py", "w") as f:
    f.write(content)
print("Patched imports")
