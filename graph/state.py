from typing import TypedDict, Dict, List, Any, Optional


class GraphState(TypedDict, total=False):
    """State representation for the RAG graph."""
    # Core message handling
    messages: List[Dict[str, Any]]      # Chat messages with role and content
    
    # Document handling
    documents: List[Any]                # Retrieved documents
    cleaned_documents: List[Any]        # Processed/cleaned documents
    
    # Search and query management
    search_type: str                    # Type of search: 'web', 'custom_knowledge_base', 'own'
    query_type: str                     # Type of query: 'question', 'statement'
    original_query: str                 # Original user query before transformation
    transformed_query: str              # Query after transformation
    
    # Generation control
    generation: Any                     # Generated response from LLM
    is_gen_code: bool                  # Whether code generation is required
    is_transform_query: bool           # Whether query transformation is needed
    
    # Execution and error handling
    stack_trace: Optional[str]         # Error stack trace from code execution
    execution_result: Any              # Result of code execution
    
    # Workflow control
    n_iterations: int                  # Number of iterations/retries
    max_iterations: int                # Maximum allowed iterations
    workflow_type: str                 # Type of workflow being executed
    
    # Configuration
    parse_str_output: bool             # Whether to parse string output from LLM
    
    # Metadata and tracking
    node_history: List[str]            # History of nodes executed
    start_time: float                  # Workflow start time
    total_tokens_used: int             # Total tokens consumed
    
    # Custom extensions (for future use)
    custom_context: Dict[str, Any]     # Custom context data
    user_preferences: Dict[str, Any]   # User-specific preferences