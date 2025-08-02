"""Reusable edge conditions for generation-related decisions."""

from graph.state import GraphState


def decide_generation_type(state: GraphState) -> str:
    """Decide which type of generation to perform."""
    is_gen_code = state.get("is_gen_code", False)
    search_type = state.get("search_type", "")
    
    if is_gen_code:
        return "generate_code"
    elif "custom_knowledge_base" in search_type or "web" in search_type:
        return "generate_context"
    elif "own" in search_type:
        return "generate"
    else:
        return "end"


