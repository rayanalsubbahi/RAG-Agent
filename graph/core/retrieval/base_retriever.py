"""Base retriever interface for different retrieval methods."""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional


class BaseRetriever(ABC):
    """Abstract base class for all retrieval methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Any]:
        """Retrieve relevant documents for a query."""
        pass
    
    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs) -> List[Any]:
        """Get relevant documents (alias for retrieve for compatibility)."""
        pass
    
    def validate_query(self, query: str) -> bool:
        """Validate if the query is suitable for retrieval."""
        return bool(query and query.strip())
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query before retrieval."""
        return query.strip()
    
    def postprocess_documents(self, documents: List[Any]) -> List[Any]:
        """Postprocess retrieved documents."""
        return documents