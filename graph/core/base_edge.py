from abc import ABC, abstractmethod
from typing import Callable, Dict, Union, Optional
from graph.state import GraphState


class BaseEdge(ABC):
    """Abstract base class for edge conditions."""
    
    def __init__(self, name: str, from_node: Optional[str] = None):
        self.name = name
        self.from_node = from_node
    
    @abstractmethod
    def evaluate(self, state: GraphState) -> str:
        """Evaluate the edge condition and return next node name."""
        pass


class ConditionalEdge(BaseEdge):
    """Edge that evaluates a condition and returns one of multiple paths."""
    
    def __init__(self, name: str, condition_fn: Callable[[GraphState], str], 
                 paths: Dict[str, str], from_node: Optional[str] = None, 
                 default_path: str = "end"):
        super().__init__(name, from_node)
        self.condition_fn = condition_fn
        self.paths = paths
        self.default_path = default_path
    
    def evaluate(self, state: GraphState) -> str:
        result = self.condition_fn(state)
        return self.paths.get(result, self.default_path)


class SimpleEdge(BaseEdge):
    """Simple edge that always leads to a specific node."""
    
    def __init__(self, name: str, target: str, from_node: Optional[str] = None):
        super().__init__(name, from_node)
        self.target = target
    
    def evaluate(self, state: GraphState) -> str:
        return self.target