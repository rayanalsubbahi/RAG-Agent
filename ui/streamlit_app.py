"""Main Streamlit application class."""

import streamlit as st
import time
from typing import Any

from graph.assistant import Assistant
from graph.core.types import WorkflowType
from ui.components import ChatInterface, HistoryManager, ModelSelector


class StreamlitApp:
    """Main Streamlit application for the RAG Assistant."""
    
    def __init__(self, llm: Any, workflow_type: WorkflowType = WorkflowType.ALL, parse_str_output: bool = True):
        """
        Initialize the Streamlit app.
        
        Args:
            llm: Language model instance
            workflow_type: Workflow type to use
            parse_str_output: Whether to parse string output
        """
        self.llm = llm
        self.workflow_type = workflow_type
        self.parse_str_output = parse_str_output
        
        # Setup page configuration
        ChatInterface.setup_page_config()
        ChatInterface.add_custom_css()
    
    def _initialize_assistant(self) -> Assistant:
        """Initialize or get the assistant from session state."""
        if 'model' not in st.session_state:
            assistant = Assistant(self.llm, self.workflow_type, self.parse_str_output)
            st.session_state['model'] = assistant
        return st.session_state['model']
    
    def _send_message(self, messages: list, assistant: Assistant) -> dict:
        """Send message to assistant and measure response time."""
        start_time = time.time()
        answer = assistant.chat(messages)
        response_time = time.time() - start_time
        print(f"Time taken for response: {response_time:.2f}s")
        return answer
    
    def _handle_chat_input(self, message: str, assistant: Assistant) -> dict:
        """Handle new chat input from user."""
        # Add user message to history
        user_message = {"role": "user", "content": message}
        st.session_state["history"].append(user_message)
        HistoryManager.write_to_file(message, "user")
        
        # Get assistant response
        response = self._send_message(st.session_state["history"], assistant)
        
        # Write assistant response to history file
        HistoryManager.write_to_file(response["content"], "assistant")
        
        return response
    
    def _display_current_chat(self):
        """Display the current chat session."""
        HistoryManager.initialize_history()
        
        # Display chat history
        for message in st.session_state["history"]:
            if message["role"] == "user" and not message.get("ai_message", False):
                ChatInterface.display_chat_message("user", message["content"])
            elif message["role"] == "assistant":
                ChatInterface.display_chat_message("assistant", message["content"])
    
    
    def run(self):
        """Run the main Streamlit application."""
        # Initialize assistant
        assistant = self._initialize_assistant()
        
        # Display organized sidebar - New Chat first
        history_file = HistoryManager.display_new_chat_button()
        
        # Display sidebar sections
        ModelSelector.display_workflow_selector(assistant)
        history_file = HistoryManager.display_history_selector()
        
        # Main content area
        if history_file and history_file != "None":
            # Display historical chat
            st.caption(history_file)
            chat_data = HistoryManager.read_chat_from_file(history_file)
            HistoryManager.display_chat_history(chat_data)
        else:
            # Display current chat session
            ChatInterface.display_main_header()
            self._display_current_chat()
        
        # Chat input
        message = st.chat_input(placeholder="Ask me anything...")
        if message:
            # Display user message immediately
            ChatInterface.display_chat_message("user", message)
            
            # Get assistant response with spinner
            with st.spinner("🤔 Thinking... "):
                response = self._handle_chat_input(message, assistant)
            
            # Display assistant response with typing effect
            formatted_response = response["content"].replace('\n', '  \n')
            st.chat_message(name="Assistant", avatar="✨").write_stream(
                ChatInterface.response_generator(formatted_response)
            )
            
            print(f"Response: {response['content']}")


def create_streamlit_app(llm: Any, workflow_type: WorkflowType = WorkflowType.ALL, parse_str_output: bool = True):
    """
    Factory function to create and run the Streamlit app.
    
    Args:
        llm: Language model instance
        workflow_type: Workflow type to use
        parse_str_output: Whether to parse string output
    """
    app = StreamlitApp(llm, workflow_type, parse_str_output)
    app.run()