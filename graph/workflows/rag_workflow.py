"""Complete RAG workflow with all features enabled."""

from graph.core.workflow_builder import WorkflowBuilder
from graph.nodes.retrieval.retrieve import RetrieveNode
from graph.nodes.retrieval.web_search import WebSearchNode
from graph.nodes.retrieval.grade_documents import GradeDocumentsNode
from graph.nodes.generation.generate import GenerateNode
from graph.nodes.generation.generate_code import GenerateCodeNode
from graph.nodes.generation.generate_context import GenerateContextNode
from graph.nodes.processing.rephrase_follow_up import RephraseFollowUpNode
from graph.nodes.processing.check_required_search import CheckRequiredSearchNode
from graph.nodes.processing.check_code_generation import CheckCodeGenerationNode
from graph.nodes.processing.clean_documents import CleanDocumentsNode
from graph.nodes.processing.transform_query import TransformQueryNode
from graph.nodes.execution.execute_code import ExecuteCodeNode
from graph.nodes.execution.generate_based_error import GenerateBasedErrorNode

from graph.edges.conditions.search_conditions import decide_rag_path, decide_search_type
from graph.edges.conditions.generation_conditions import decide_generation_type
from graph.edges.conditions.execution_conditions import decide_execution_path, decide_error_handling_path


class RAGWorkflowBuilder:
    """Builder for the complete RAG workflow."""
    
    @staticmethod
    def create_full_workflow() -> WorkflowBuilder:
        """Create the complete RAG workflow with all nodes."""
        builder = WorkflowBuilder("full_rag_workflow")
        
        # Add all nodes
        builder.add_nodes([
            RephraseFollowUpNode(),
            CheckRequiredSearchNode(), 
            CheckCodeGenerationNode(),
            RetrieveNode(),
            WebSearchNode(),
            GradeDocumentsNode(),
            CleanDocumentsNode(),
            TransformQueryNode(),
            GenerateCodeNode(),
            GenerateContextNode(),
            GenerateNode(),
            ExecuteCodeNode(),
            GenerateBasedErrorNode()
        ])
        
        # Define the complete workflow structure following the exact provided flow
        builder.set_entry_point("rephrase_follow_up_question")
        
        # Entry decision: Skip RAG or continue
        builder.add_conditional_edge(
            "rephrase_follow_up_question",
            decide_rag_path,
            {
                "generate": "generate",
                "check_required_search": "check_required_search"
            }
        )
        
        # Search type decision
        builder.add_conditional_edge(
            "check_required_search", 
            decide_search_type,
            {
                "retrieve": "retrieve",
                "web_search": "web_search",
                "check_code_generation": "check_code_generation"
            }
        )
        
        # Web search flow
        builder.add_simple_edge("web_search", "clean_documents")
        builder.add_simple_edge("clean_documents", "grade_documents")
        
        # Knowledge base retrieval flow
        builder.add_simple_edge("retrieve", "grade_documents")
        
        # Grading decision
        builder.add_conditional_edge(
            "grade_documents",
            decide_generation_type,
            {
                "transform_query": "transform_query",
                "check_code_generation": "check_code_generation"
            }
        )
        
        # Transform query loop back
        builder.add_conditional_edge(
            "transform_query",
            decide_search_type,
            {
                "check_required_search": "check_required_search",
                "end": "end"
            }
        )
        
        # Generation type decision
        builder.add_conditional_edge(
            "check_code_generation",
            decide_generation_type,
            {
                "generate_code": "generate_code",
                "generate_context": "generate_context",
                "generate": "generate"
            }
        )
        
        # Code execution decision
        builder.add_conditional_edge(
            "generate_code",
            decide_execution_path,
            {
                "execute_code": "execute_code",
                "end": "end"
            }
        )
        
        # Error handling after execution
        builder.add_conditional_edge(
            "execute_code",
            decide_error_handling_path,
            {
                "generate_based_on_error": "generate_based_on_error",
                "end": "end"
            }
        )
        
        # Retry execution after error correction
        builder.add_simple_edge("generate_based_on_error", "execute_code")
        
        # End paths for other generators
        builder.add_simple_edge("generate_context", "end")
        builder.add_simple_edge("generate", "end")
        
        return builder
    
    @staticmethod
    def create_simple_rag_workflow() -> WorkflowBuilder:
        """Create a simplified RAG workflow for testing."""
        builder = WorkflowBuilder("simple_rag_workflow")
        
        # Add basic nodes
        builder.add_nodes([
            CheckRequiredSearchNode(),
            RetrieveNode(),
            WebSearchNode(),
            GenerateContextNode(),
            GenerateNode()
        ])
        
        # Simple flow
        builder.set_entry_point("check_required_search")
        
        builder.add_conditional_edge(
            "check_required_search",
            decide_search_type,
            {
                "web_search": "web_search",
                "retrieve": "retrieve",
                "generate": "generate",
                "end": "end"
            }
        )
        
        builder.add_simple_edge("retrieve", "generate_context")
        builder.add_simple_edge("web_search", "generate_context")
        builder.add_simple_edge("generate_context", "end")
        builder.add_simple_edge("generate", "end")
        
        return builder