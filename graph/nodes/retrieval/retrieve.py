from graph.core.base_node import RetrieverNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message


class RetrieveNode(RetrieverNode):
    """Node for retrieving content from vector store documents."""
    
    def __init__(self):
        super().__init__("retrieve")
    
    def execute(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents from vector store."""
        self.log("Retrieving documents")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        
        # Retrieve documents using the vector store
        documents = self.retriever.vectorStore.get_relevant_documents(question)
        
        print(f"Retrieved {len(documents)} documents")
        state["documents"] = documents
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages are present in state."""
        return "messages" in state and len(state["messages"]) > 0


