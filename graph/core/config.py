"""Configuration settings for the RAG system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution parameters."""
    
    # Iteration limits
    max_execution_iterations: int = 3
    max_search_iterations: int = 3
    max_transform_iterations: int = 3
    
    # Timeout settings
    execution_timeout: Optional[int] = None
    search_timeout: Optional[int] = None
    
    # Debug settings
    debug_mode: bool = False
    verbose_logging: bool = False
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_execution_iterations <= 0:
            raise ValueError("max_execution_iterations must be positive")
        if self.max_search_iterations <= 0:
            raise ValueError("max_search_iterations must be positive")
        if self.max_transform_iterations <= 0:
            raise ValueError("max_transform_iterations must be positive")


# Default configuration instance
DEFAULT_CONFIG = WorkflowConfig()


def get_config() -> WorkflowConfig:
    """Get the current workflow configuration."""
    return DEFAULT_CONFIG


def set_config(config: WorkflowConfig) -> None:
    """Set the global workflow configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


def update_config(**kwargs) -> None:
    """Update specific configuration values."""
    global DEFAULT_CONFIG
    for key, value in kwargs.items():
        if hasattr(DEFAULT_CONFIG, key):
            setattr(DEFAULT_CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")