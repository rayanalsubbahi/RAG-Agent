"""Document retrieval-only workflow with full original logic."""

from graph.core.workflow_builder import WorkflowBuilder
from graph.nodes.retrieval.retrieve import RetrieveNode
from graph.nodes.processing.grade_documents import GradeDocumentsNode
from graph.nodes.processing.rephrase_follow_up import RephraseFollowUpNode
from graph.nodes.processing.transform_query import TransformQueryNode
from graph.nodes.generation.generate_context import GenerateContextNode
from graph.nodes.generation.generate import GenerateNode

from graph.edges.conditions.generation_conditions import decide_to_generate_web
from graph.edges.conditions.search_conditions import decide_rag_path_retrieval, transform_query_search_retrieval


class RetrievalWorkflowBuilder:
    """Builder for knowledge base retrieval-only workflow."""
    
    @staticmethod
    def create_retrieval_workflow() -> WorkflowBuilder:
        """Create a knowledge base retrieval-only workflow matching original logic."""
        builder = WorkflowBuilder("retrieval_workflow")
        
        # Add all nodes needed for retrieval workflow
        builder.add_nodes([
            RephraseFollowUpNode(),
            RetrieveNode(),
            GradeDocumentsNode(),
            TransformQueryNode(),
            GenerateContextNode(),
            GenerateNode()
        ])
        
        # Set entry point like original
        builder.set_entry_point("rephrase_follow_up_question")
        
        # Entry decision: Skip RAG or continue (original logic)
        builder.add_conditional_edge(
            "rephrase_follow_up_question",
            decide_rag_path_retrieval,
            {
                "generate": "generate",
                "retrieve": "retrieve",
            }
        )
        
        # Knowledge base retrieval flow (original logic)
        builder.add_simple_edge("retrieve", "grade_documents")
        
        # Grading decision (original logic)
        builder.add_conditional_edge(
            "grade_documents",
            decide_to_generate_web,
            {
                "transform_query": "transform_query",
                "generate_context": "generate_context"
            }
        )
        
        # Transform query loop back (original logic)
        builder.add_conditional_edge(
            "transform_query",
            transform_query_search_retrieval,
            {
                "retrieve": "retrieve",
                "end": "end",
            }
        )
        
        # End paths
        builder.add_simple_edge("generate_context", "end")
        builder.add_simple_edge("generate", "end")
        
        return builder