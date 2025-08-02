"""Web-only search workflow."""

from graph.core.workflow_builder import WorkflowBuilder
from graph.nodes.retrieval.web_search import WebSearchNode
from graph.nodes.processing.clean_documents import CleanDocumentsNode
from graph.nodes.generation.generate_context import GenerateContextNode


class WebWorkflowBuilder:
    """Builder for web-only search workflow."""
    
    @staticmethod
    def create_web_workflow() -> WorkflowBuilder:
        """Create a web-only search workflow."""
        builder = WorkflowBuilder("web_workflow")
        
        # Add nodes
        builder.add_nodes([
            WebSearchNode(search_method="tavily", max_results=3),
            CleanDocumentsNode(),
            GenerateContextNode()
        ])
        
        builder.set_entry_point("web_search")
        
        # Workflow: web_search -> clean_documents -> generate_context -> end
        builder.add_simple_edge("web_search", "clean_documents")
        builder.add_simple_edge("clean_documents", "generate_context")
        builder.add_simple_edge("generate_context", "end")
        
        return builder