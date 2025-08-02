"""Reusable edge conditions for code execution-related decisions."""

from graph.state import GraphState
from graph.core.config import get_config

def decide_execution_path(state: GraphState) -> str:
    """Decide whether to execute code or end."""
    is_gen_code = state.get("is_gen_code", False)
    
    if is_gen_code:
        return "execute_code"
    else:
        return "end"

def decide_error_handling_path(state: GraphState) -> str:
    """Decide whether to handle stack trace error or end."""
    stack_trace = state.get("stack_trace", "")
    n_iterations = state.get("n_iterations", 0)
    max_iterations = get_config().max_execution_iterations
    
    if stack_trace and n_iterations < max_iterations:
        return "generate_based_on_error"
    else:
        return "end"

