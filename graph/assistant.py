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
    
