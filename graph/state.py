from typing import TypedDict, Dict, List

# Define the GraphState
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: List[str]
    documents: List[str]
    search_type: str
    query_type: str
    stack_trace: str
    generation: str
    is_gen_code: bool
    is_transform_query: bool
    
    n_iterations: int
    parse_str_output: bool