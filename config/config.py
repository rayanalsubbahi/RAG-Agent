"""Centralized configuration settings for the RAG system."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    
    # Web search settings
    web_search_method: str = "tavily"
    web_max_results: int = 3
    google_api_key: Optional[str] = None
    google_csi_id: Optional[str] = None
    
    # Document processing settings  
    relevance_threshold: float = 0.5
    max_transform_iterations: int = 3


@dataclass 
class AsyncConfig:
    """Configuration for async processing."""
    
    # Concurrency settings
    max_concurrent_documents: int = 10
    
    # Adaptive concurrency thresholds
    small_batch_size: int = 5
    medium_batch_size: int = 20
    small_batch_concurrency: int = 3
    medium_batch_concurrency: int = 8
    
    def get_optimal_concurrency(self, num_documents: int) -> int:
        """Get optimal concurrency based on document count."""
        if num_documents <= self.small_batch_size:
            return min(num_documents, self.small_batch_concurrency)
        elif num_documents <= self.medium_batch_size:
            return min(num_documents, self.medium_batch_concurrency)
        else:
            return self.max_concurrent_documents


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    
    max_execution_iterations: int = 3


@dataclass
class WorkflowConfig:
    """Master configuration containing all sub-configurations."""
    
    # Sub-configurations
    search: SearchConfig = field(default_factory=SearchConfig)
    async_processing: AsyncConfig = field(default_factory=AsyncConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Direct access for legacy compatibility
    max_execution_iterations: int = 3
    
    def __post_init__(self):
        """Load from environment and validate."""
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration values from environment variables."""
        if os.getenv("GOOGLE_SEARCH_API_KEY"):
            self.search.google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        if os.getenv("GOOGLE_CSI_ID"):
            self.search.google_csi_id = os.getenv("GOOGLE_CSI_ID")
    
    
# Global configuration instance
_GLOBAL_CONFIG: Optional[WorkflowConfig] = None


def get_config() -> WorkflowConfig:
    """Get the current workflow configuration."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = WorkflowConfig()
    return _GLOBAL_CONFIG


def get_search_config() -> SearchConfig:
    """Get search configuration."""
    return get_config().search


def get_async_config() -> AsyncConfig:
    """Get async processing configuration."""
    return get_config().async_processing