"""Reusable edge conditions for search-related decisions."""

from graph.state import GraphState
from graph.core.config import get_search_config


def decide_search_type(state: GraphState) -> str:
    """Decide which type of search to perform."""
    search_type = state.get("search_type", "").lower()
    print(f"----Deciding to search the web or the knowledge base for more documents----")
    print(f"DEBUG: search_type = '{search_type}'")
    
    if "web" in search_type:
        return "web_search"
    elif "custom_knowledge_base" in search_type or "knowledge_base" in search_type:
        return "retrieve"
    elif "own" in search_type or "model" in search_type:
        return "check_code_generation"
    else:
        # Default to generate with own knowledge if unclear
        print(f"WARNING: Unknown search_type '{search_type}', defaulting to check_code_generation")
        return "check_code_generation"
    

def decide_rag_path(state: GraphState) -> str:
    """Decide whether to skip RAG or continue."""
    print("----Deciding to skip RAG and continue directly with generation----")
    query_type = state.get("query_type", "")
    
    if query_type == "statement":
        return "generate"
    else:
        return "check_required_search"


def transform_query_search(state: GraphState) -> str:
    """Direct transformed query based on search type and iterations."""
    n_iterations = state.get("n_iterations", 0)
    search_config = get_search_config()
    
    if n_iterations < search_config.max_transform_iterations:
        # check type of search required
        return "check_required_search"
    else:
        # Continue with existing answer
        return "end"


def transform_query_search_web(state: GraphState) -> str:
    """Direct transformed query for web workflow specifically."""
    n_iterations = state.get("n_iterations", 0)
    search_config = get_search_config()
    
    if n_iterations < search_config.max_transform_iterations:
        # go back to web search directly
        return "web_search"
    else:
        # Continue with existing answer
        return "end"


def transform_query_search_retrieval(state: GraphState) -> str:
    """Direct transformed query for retrieval workflow specifically."""
    n_iterations = state.get("n_iterations", 0)
    search_config = get_search_config()
    
    if n_iterations < search_config.max_transform_iterations:
        # go back to knowledge base retrieval directly
        return "retrieve"
    else:
        # Continue with existing answer
        return "end"


def decide_rag_path_retrieval(state: GraphState) -> str:
    """Decide whether to skip RAG or continue for retrieval workflow."""
    print("----Deciding to skip RAG and continue directly with generation----")
    query_type = state.get("query_type", "")
    
    if query_type == "statement":
        return "generate"
    else:
        return "retrieve"
