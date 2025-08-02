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


def decide_to_generate(state: GraphState) -> str:
    """Decide to transform query or proceed with generation (from old logic)."""
    print("----Deciding to generate or search----")
    
    is_transform_query = state.get("is_transform_query", False)
    
    if is_transform_query:
        # Perform transform query then search
        return "transform_query"
    else:
        return "check_code_generation"


def decide_to_generate_web(state: GraphState) -> str:
    """Decide to transform query or proceed with generation for web workflow."""
    print("----Deciding to generate or search (web workflow)----")
    
    is_transform_query = state.get("is_transform_query", False)
    
    if is_transform_query:
        # Perform transform query then search
        return "transform_query"
    else:
        return "generate_context"


