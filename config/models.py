"""Model configuration and factory for different LLM providers."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import LLM classes with error handling
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_cohere import ChatCohere
except ImportError:
    ChatCohere = None

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:
    ChatNVIDIA = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_xai import ChatXAI
except ImportError:
    ChatXAI = None


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    provider: str
    model_id: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    default_params: Optional[Dict[str, Any]] = None


# Available model configurations
AVAILABLE_MODELS = {
    # Anthropic Models
    "claude-3-5-haiku": ModelConfig(
        name="Claude 3.5 Haiku",
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        api_key_env="ANTHROPIC_API_KEY"
    ),
    "claude-3-5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet", 
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY"
    ),
    
    # OpenAI Models
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        provider="openai",
        model_id="gpt-4o-mini-2024-07-18",
        api_key_env="OPENAI_API_KEY"
    ),
    
    # DeepSeek
    "deepseek-chat": ModelConfig(
        name="DeepSeek Chat",
        provider="openai",  # Uses OpenAI-compatible API
        model_id="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com"
    ),
    
    # Cohere
    "command-r": ModelConfig(
        name="Command R",
        provider="cohere",
        model_id="command-r-08-2024",
        api_key_env="COHERE_API_KEY"
    ),
    
    # NVIDIA
    "nemotron-4": ModelConfig(
        name="Nemotron 4 340B",
        provider="nvidia",
        model_id="nvidia/nemotron-4-340b-instruct",
        api_key_env="NVIDIA_API_KEY"
    ),
    
    # Google
    "gemini-2-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        provider="google",
        model_id="gemini-2.0-flash-thinking-exp-01-21",
        api_key_env="GEMINI_API_KEY"
    ),
    
    # xAI
    "grok-2": ModelConfig(
        name="Grok 2",
        provider="xai", 
        model_id="grok-2-1212",
        api_key_env="GROK_API_KEY"
    ),
    
    # Ollama (local)
    "ollama-local": ModelConfig(
        name="Ollama Local",
        provider="ollama",
        model_id=":latest",
        base_url="https://196.219.58.150:11434/"
    )
}


def get_available_models() -> Dict[str, ModelConfig]:
    """Get all available model configurations."""
    return AVAILABLE_MODELS


def create_llm(model_key: str, **kwargs) -> Any:
    """Create an LLM instance based on the model key."""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    config = AVAILABLE_MODELS[model_key]
    
    # Get API key if required
    api_key = None
    if config.api_key_env:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            print(f"⚠️  Warning: {config.api_key_env} not found in environment variables")
    
    # Create the appropriate LLM instance
    if config.provider == "anthropic" and ChatAnthropic:
        return ChatAnthropic(
            model=config.model_id,
            anthropic_api_key=api_key,
            **kwargs
        )
    
    elif config.provider == "openai" and ChatOpenAI:
        params = {
            "model": config.model_id,
            **kwargs
        }
        if api_key:
            params["api_key"] = api_key
        if config.base_url:
            params["base_url"] = config.base_url
        return ChatOpenAI(**params)
    
    elif config.provider == "cohere" and ChatCohere:
        return ChatCohere(
            model=config.model_id,
            cohere_api_key=api_key,
            **kwargs
        )
    
    elif config.provider == "nvidia" and ChatNVIDIA:
        return ChatNVIDIA(
            model=config.model_id,
            nvidia_api_key=api_key,
            **kwargs
        )
    
    elif config.provider == "google" and ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=config.model_id,
            api_key=api_key,
            **kwargs
        )
    
    elif config.provider == "xai" and ChatXAI:
        return ChatXAI(
            model=config.model_id,
            api_key=api_key,
            **kwargs
        )
    
    elif config.provider == "ollama" and ChatOllama:
        return ChatOllama(
            model=config.model_id,
            base_url=config.base_url,
            **kwargs
        )
    
    else:
        available_providers = [cfg.provider for cfg in AVAILABLE_MODELS.values()]
        raise ValueError(f"Provider '{config.provider}' not available. Check if the required package is installed. Available providers: {set(available_providers)}")


def get_default_model() -> str:
    """Get the default model key."""
    return "command-r"  # Use Cohere Command R