from typing import Any, Dict
from langgraph.graph import StateGraph
from graph.core.retrieval.vector_retriever import VectorRetriever
from graph.core.workflow_builder import WorkflowBuilder
from graph.workflows.rag_workflow import RAGWorkflowBuilder
from graph.workflows.web_workflow import WebWorkflowBuilder
from graph.core.types import WorkflowType


class RAGGraph:
    """RAG Graph with modular architecture."""
    
    def __init__(self, llm, workflow_type: WorkflowType, parse_str_output: bool = False):
        self.llm = llm
        self.workflow_type = workflow_type
        self.parse_str_output = parse_str_output
        self.retriever = VectorRetriever()
        
        # Build the appropriate workflow
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self) -> StateGraph:
        """Create the appropriate pipeline based on workflow type."""
        builder = self._get_workflow_builder()
        
        # Add common dependencies
        builder.add_dependencies(
            llm=self.llm,
            retriever=self.retriever
        )
        
        # Validate workflow before building
        errors = builder.validate()
        if errors:
            print(f"Workflow validation errors: {errors}")
        
        return builder.build()
    
    def _get_workflow_builder(self) -> WorkflowBuilder:
        """Get the appropriate workflow builder based on type."""
        if self.workflow_type == WorkflowType.ALL:
            return RAGWorkflowBuilder.create_full_workflow()
        elif self.workflow_type == WorkflowType.WEB:
            return WebWorkflowBuilder.create_web_workflow()
        elif self.workflow_type == WorkflowType.RETRIEVE:
            return RAGWorkflowBuilder.create_simple_rag_workflow()
        else:
            # Default to full workflow for maximum functionality
            return RAGWorkflowBuilder.create_full_workflow()
    
    def invoke_pipeline(self, messages: list) -> Dict[str, Any]:
        """Invoke the pipeline with the given messages."""
        inputs = {
            "messages": messages,
            "documents": [],
            "search_type": "",
            "query_type": "",
            "stack_trace": "",
            "generation": "",
            "is_gen_code": False,
            "is_transform_query": False,
            "n_iterations": 0,
            "parse_str_output": self.parse_str_output
        }
        
        # Stream through the pipeline
        final_output = None
        for output in self.pipeline.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
                final_output = value  # Keep track of the final state
        
        # Extract final generation from the last output
        if final_output:
            if 'generation' in final_output and hasattr(final_output['generation'], 'content'):
                answer = final_output["generation"].content
            elif 'generation' in final_output:
                answer = str(final_output["generation"])
            else:
                answer = "No response generated"
        else:
            answer = "Pipeline failed to execute"
        
        ai_message = {"role": "assistant", "content": answer}
        messages.append(ai_message)
        return ai_message
    
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the current workflow."""
        return {
            "workflow_type": self.workflow_type,
            "parse_str_output": self.parse_str_output,
            "has_retriever": self.retriever is not None,
            "llm_model": str(self.llm) if self.llm else None
        }