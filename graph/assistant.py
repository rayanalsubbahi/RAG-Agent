try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Environment variables from .env file won't be loaded.")

from graph.graph import RAGGraph
from graph.core.types import WorkflowType


class Assistant:
    """Main Assistant interface for the RAG system."""
    
    def __init__(self, llm, workflow_type: WorkflowType = WorkflowType.ALL, parse_str_output: bool = True):
        """Initialize the Assistant."""
        self.llm = llm
        self.workflow_type = workflow_type
        self.parse_str_output = parse_str_output
        
        self.pipeline = RAGGraph(llm, workflow_type, parse_str_output)
        print(f"🚀 Assistant initialized (workflow: {workflow_type})")
    
    def chat(self, messages: list) -> dict:
        """Process chat conversation and return response."""
        return self.pipeline.invoke_pipeline(messages)
    
    def update_workflow(self, new_workflow_type: WorkflowType):
        """Update the workflow type and rebuild the pipeline."""
        if new_workflow_type != self.workflow_type:
            print(f"🔄 Switching workflow from {self.workflow_type} to {new_workflow_type}")
            self.workflow_type = new_workflow_type
            
            # Rebuild the pipeline with the new workflow type
            self.pipeline = RAGGraph(self.llm, new_workflow_type, self.parse_str_output)
            print(f"✅ Workflow updated successfully")
    
    def get_workflow_info(self) -> dict:
        """Get information about the current workflow configuration."""
        return {
            "workflow_type": self.workflow_type,
            "parse_str_output": self.parse_str_output,
            "pipeline_info": self.pipeline.get_workflow_info() if hasattr(self.pipeline, 'get_workflow_info') else None
        }
    
