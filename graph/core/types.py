"""Core types and enums for the RAG system."""

from enum import Enum


class WorkflowType:
    """Enum-like class for workflow types."""
    ALL = 'all'           # Full RAG workflow with all features
    WEB = 'web'           # Web search only workflow  
    RETRIEVE = 'retrieve' # Local retrieval only workflow


class NodeCategory(Enum):
    """Categories for organizing nodes."""
    RETRIEVAL = "retrieval"     # Document retrieval and search
    GENERATION = "generation"   # Text and code generation
    PROCESSING = "processing"   # Query processing and validation
    EXECUTION = "execution"     # Code execution and error handling


class EdgeType(Enum):
    """Types of edges in the workflow."""
    SIMPLE = "simple"           # Direct connection between nodes
    CONDITIONAL = "conditional" # Conditional routing based on state
    PREDICATE = "predicate"     # Boolean-based routing