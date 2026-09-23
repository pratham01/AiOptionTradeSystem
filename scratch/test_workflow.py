import asyncio
import sys
import os
import logging

sys.path.append(os.path.abspath('src'))
logging.basicConfig(level=logging.INFO)

from trade_system.core.orchestration.workflow_engine import WorkflowEngine

async def main():
    engine = WorkflowEngine("config/workflow.yaml")
    print(f"Execution Order: {engine.execution_order}")
    final_context = await engine.run()
    
    print("\n--- FINAL CONTEXT ---")
    print(final_context)

if __name__ == "__main__":
    asyncio.run(main())
