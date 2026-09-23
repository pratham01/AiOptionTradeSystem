import yaml
import logging
import asyncio
from typing import Any, Dict, List
import importlib

LOGGER = logging.getLogger(__name__)

class WorkflowNode:
    def __init__(self, node_config: dict):
        self.id = node_config["id"]
        self.module_path = node_config["module"]
        self.depends_on = node_config.get("depends_on", [])
        self.config = node_config.get("config", {})
        self.description = node_config.get("description", "")
        self.module_instance = None

    def load_module(self):
        try:
            mod = importlib.import_module(self.module_path)
            if hasattr(mod, "Module"):
                self.module_instance = mod.Module(self.config)
            else:
                LOGGER.error(f"Module {self.module_path} missing 'Module' class.")
        except Exception as e:
            LOGGER.error(f"Failed to load module {self.module_path}: {e}")

    async def execute(self, context: Dict[str, Any]) -> None:
        if not self.module_instance:
            self.load_module()
        if self.module_instance:
            LOGGER.info(f"[Orchestrator] Executing node: {self.id}")
            await self.module_instance.execute(context)


class WorkflowEngine:
    """
    DAG-based AI orchestration engine.
    Reads a YAML configuration, resolves dependencies, and executes nodes.
    """
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.config = self._load_yaml()
        self.nodes: Dict[str, WorkflowNode] = {}
        self.execution_order: List[str] = []
        self._initialize_nodes()

    def _load_yaml(self) -> dict:
        try:
            with open(self.yaml_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            LOGGER.error(f"Failed to load {self.yaml_path}: {e}")
            return {}

    def _initialize_nodes(self):
        if "nodes" not in self.config:
            return
        
        for n_conf in self.config["nodes"]:
            self.nodes[n_conf["id"]] = WorkflowNode(n_conf)
            
        self.execution_order = self._topological_sort()

    def _topological_sort(self) -> List[str]:
        """Resolves node dependencies and returns execution order."""
        visited = set()
        temp_mark = set()
        order = []

        def visit(node_id):
            if node_id in temp_mark:
                raise RuntimeError(f"Circular dependency detected at node {node_id}")
            if node_id not in visited:
                temp_mark.add(node_id)
                node = self.nodes.get(node_id)
                if node:
                    for dep in node.depends_on:
                        visit(dep)
                temp_mark.remove(node_id)
                visited.add(node_id)
                order.append(node_id)

        for n_id in self.nodes:
            if n_id not in visited:
                visit(n_id)

        return order

    async def run(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Executes the workflow."""
        context = initial_context or {}
        LOGGER.info(f"Starting Workflow: {self.config.get('name', 'Unnamed')}")
        
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            try:
                await node.execute(context)
            except Exception as e:
                LOGGER.error(f"Error executing node {node_id}: {e}")
                # Depending on strictness, we might raise or continue
                raise e
        
        LOGGER.info("Workflow execution completed.")
        return context
