from typing import Dict, List, Callable, Any, Optional
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.core.base_node import BaseNode
from graph.core.base_edge import BaseEdge, ConditionalEdge, SimpleEdge


class WorkflowBuilder:
    """Builder class for creating composable LangGraph workflows."""
    
    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: List[BaseEdge] = []
        self.entry_point: Optional[str] = None
        self.dependencies: Dict[str, Any] = {}
    
    def add_node(self, node: BaseNode) -> 'WorkflowBuilder':
        """Add a node to the workflow."""
        self.nodes[node.name] = node
        return self
    
    def add_nodes(self, nodes: List[BaseNode]) -> 'WorkflowBuilder':
        """Add multiple nodes to the workflow."""
        for node in nodes:
            self.add_node(node)
        return self
    
    def add_conditional_edge(self, from_node: str, condition_fn: Callable[[GraphState], str], 
                           paths: Dict[str, str]) -> 'WorkflowBuilder':
        """Add a conditional edge between nodes."""
        edge = ConditionalEdge(f"{from_node}_conditional", condition_fn, paths, from_node)
        self.edges.append(edge)
        return self
    
    def add_simple_edge(self, from_node: str, to_node: str) -> 'WorkflowBuilder':
        """Add a simple edge between nodes."""
        edge = SimpleEdge(f"{from_node}_to_{to_node}", to_node, from_node)
        self.edges.append(edge)
        return self
    
    def set_entry_point(self, node_name: str) -> 'WorkflowBuilder':
        """Set the entry point for the workflow."""
        self.entry_point = node_name
        return self
    
    def add_dependencies(self, **kwargs) -> 'WorkflowBuilder':
        """Add dependencies (LLM, retriever, etc.) for nodes."""
        self.dependencies.update(kwargs)
        return self
    
    def build(self) -> StateGraph:
        """Build and return the LangGraph StateGraph."""
        workflow = StateGraph(GraphState)
        
        # Setup dependencies for all nodes
        for node in self.nodes.values():
            required_deps = node.get_required_dependencies()
            node_deps = {dep: self.dependencies[dep] for dep in required_deps if dep in self.dependencies}
            node.setup_dependencies(**node_deps)
        
        # Add nodes to workflow
        for node_name, node in self.nodes.items():
            workflow.add_node(node_name, node.execute)
        
        # Add edges to workflow
        for edge in self.edges:
            if isinstance(edge, ConditionalEdge):
                # Convert "end" to END in conditional paths
                processed_paths = {}
                for key, target in edge.paths.items():
                    if target == "end":
                        processed_paths[key] = END
                    else:
                        processed_paths[key] = target
                
                workflow.add_conditional_edges(
                    edge.from_node,
                    edge.condition_fn,
                    processed_paths
                )
            elif isinstance(edge, SimpleEdge):
                if edge.target == "end":
                    workflow.add_edge(edge.from_node, END)
                else:
                    workflow.add_edge(edge.from_node, edge.target)
        
        # Set entry point
        if self.entry_point:
            workflow.set_entry_point(self.entry_point)
        
        return workflow.compile()
    
    def validate(self) -> List[str]:
        """Validate the workflow configuration and return any errors."""
        errors = []
        
        # Check if entry point is set
        if not self.entry_point:
            errors.append("No entry point set")
        
        # Check if entry point exists in nodes
        if self.entry_point and self.entry_point not in self.nodes:
            errors.append(f"Entry point '{self.entry_point}' not found in nodes")
        
        # Check if all edge targets exist
        for edge in self.edges:
            if hasattr(edge, 'from_node') and edge.from_node not in self.nodes:
                errors.append(f"Edge from node '{edge.from_node}' not found")
            
            if isinstance(edge, SimpleEdge) and edge.target not in self.nodes and edge.target != END and edge.target != "end":
                errors.append(f"Edge target '{edge.target}' not found")
            
            if isinstance(edge, ConditionalEdge):
                for path_target in edge.paths.values():
                    if path_target not in self.nodes and path_target != END and path_target != "end":
                        errors.append(f"Conditional edge target '{path_target}' not found")
        
        # Check node dependencies
        for node_name, node in self.nodes.items():
            required_deps = node.get_required_dependencies()
            missing_deps = [dep for dep in required_deps if dep not in self.dependencies]
            if missing_deps:
                errors.append(f"Node '{node_name}' missing dependencies: {missing_deps}")
        
        return errors
    
    def clone(self) -> 'WorkflowBuilder':
        """Create a copy of this workflow builder."""
        new_builder = WorkflowBuilder(self.name)
        new_builder.nodes = self.nodes.copy()
        new_builder.edges = self.edges.copy()
        new_builder.entry_point = self.entry_point
        new_builder.dependencies = self.dependencies.copy()
        return new_builder
    
    def extend(self, other: 'WorkflowBuilder') -> 'WorkflowBuilder':
        """Extend this workflow with nodes and edges from another workflow."""
        self.nodes.update(other.nodes)
        self.edges.extend(other.edges)
        self.dependencies.update(other.dependencies)
        return self