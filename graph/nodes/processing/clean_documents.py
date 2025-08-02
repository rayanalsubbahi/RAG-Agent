from graph.core.base_node import LLMNode
from graph.state import GraphState


class CleanDocumentsNode(LLMNode):
    """Node for cleaning and preprocessing retrieved documents."""
    
    def __init__(self):
        super().__init__("clean_documents")
    
    def execute(self, state: GraphState) -> GraphState:
        """Clean and preprocess documents."""
        self.log("Cleaning documents")
        
        documents = state.get("documents", [])
        
        # Basic document cleaning - remove empty docs, normalize whitespace
        cleaned_documents = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                content = doc.page_content.strip()
                if content:  # Only keep non-empty documents
                    # Normalize whitespace
                    content = ' '.join(content.split())
                    doc.page_content = content
                    cleaned_documents.append(doc)
            elif isinstance(doc, str) and doc.strip():
                cleaned_documents.append(doc.strip())
        
        state["documents"] = cleaned_documents
        print(f"Cleaned {len(documents)} documents to {len(cleaned_documents)} valid documents")
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that documents are present in state."""
        return "documents" in state


