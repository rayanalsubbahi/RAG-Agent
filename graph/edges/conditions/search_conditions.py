"""Reusable edge conditions for search-related decisions."""

from graph.state import GraphState


def decide_search_type(state: GraphState) -> str:
    """Decide which type of search to perform."""
    search_type = state.get("search_type", "").lower()
    print(f"DEBUG: search_type = '{search_type}'")
    
    if "web" in search_type:
        return "web_search"
    elif "custom_knowledge_base" in search_type or "knowledge_base" in search_type:
        return "retrieve"
    elif "own" in search_type or "model" in search_type:
        return "generate"
    else:
        # Default to generate with own knowledge if unclear
        print(f"WARNING: Unknown search_type '{search_type}', defaulting to generate")
        return "generate"
    
def decide_rag_path(state: GraphState) -> str:
    """Decide whether to skip RAG or continue."""
    query_type = state.get("query_type", "")
    
    if query_type == "statement":
        return "generate"
    else:
        return "check_required_search"
