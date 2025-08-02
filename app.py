#!/usr/bin/env python3
"""
Main application entry point for the RAG Assistant.
"""

from config.models import create_llm, get_default_model
from graph.core.types import WorkflowType
from ui.streamlit_app import create_streamlit_app

# Configuration - Change these values as needed
MODEL_NAME = None  # Set to None to use default, or specify model name like "gpt-4", "ollama-llama3", etc.
WORKFLOW_TYPE = WorkflowType.ALL  # Options: WorkflowType.ALL, WorkflowType.WEB, WorkflowType.RETRIEVE


def main():
    """Main application entry point."""
    
    # Determine which model to use
    try:
        if MODEL_NAME:
            # Use specified model name
            model_key = MODEL_NAME
            print(f"Using specified model: {model_key}")
        else:
            # Use default model from configuration
            model_key = get_default_model()
            print(f"Using default model: {model_key}")
        
        llm = create_llm(model_key)
        print(f"✅ Model loaded successfully")
        
    except Exception as e:
        print(f"⚠️  Error creating model '{model_key}': {e}")
        print("Please check your model configuration and dependencies.")
        return
    
    # Create and run the Streamlit app
    create_streamlit_app(
        llm=llm,
        workflow_type=WORKFLOW_TYPE,
        parse_str_output=True
    )


if __name__ == "__main__":
    main()