from abc import ABC, abstractmethod
from typing import Dict, Any

class ModuleInterface(ABC):
    """
    Base interface for all orchestration modules (nodes).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the module logic.
        Data flows via the mutable `context` dictionary.
        """
        pass
