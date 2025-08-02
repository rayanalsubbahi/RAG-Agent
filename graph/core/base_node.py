from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from graph.state import GraphState


class BaseNode(ABC):
    """Abstract base class for all graph nodes providing a standard interface."""
    
    def __init__(self, name: str, dependencies: Optional[Dict[str, Any]] = None):
        self.name = name
        self.dependencies = dependencies or {}
        
    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        """Execute the node's logic and return updated state."""
        pass
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that required inputs are present in state."""
        return True
    
    def get_required_dependencies(self) -> List[str]:
        """Return list of required dependency keys."""
        return []
    
    def setup_dependencies(self, **kwargs):
        """Setup node dependencies (LLM, retriever, etc.)."""
        self.dependencies.update(kwargs)
    
    def log(self, message: str):
        """Log node execution information."""
        print(f"----{self.name}: {message}----")


class LLMNode(BaseNode):
    """Base class for nodes that require an LLM."""
    
    def __init__(self, name: str, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(name, dependencies)
        
    @property
    def llm(self):
        return self.dependencies.get('llm')
    
    def get_required_dependencies(self) -> List[str]:
        return ['llm']


class RetrieverNode(BaseNode):
    """Base class for nodes that require a retriever."""
    
    def __init__(self, name: str, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(name, dependencies)
        
    @property
    def retriever(self):
        return self.dependencies.get('retriever')
    
    def get_required_dependencies(self) -> List[str]:
        return ['retriever']