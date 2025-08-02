"""Reusable UI components for the Streamlit app."""

import os
import time
import datetime
import streamlit as st
from typing import List, Tuple, Dict, Any


class ChatInterface:
    """Handles chat interface functionality."""
    
    @staticmethod
    def setup_page_config():
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="RAG Assistant",
            page_icon="🤖", 
            layout="wide"
        )
    
    @staticmethod
    def add_custom_css():
        """Add custom CSS styling."""
        st.markdown("""
        <style>
            .stButton>button {
                background: linear-gradient(to right, #FF4B4B, #7E56D9);
                border: none;
                color: white;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_sidebar_header():
        """Display sidebar header only."""
        with st.sidebar:
            # Header with logo
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='color: #FF4B4B; margin: 0;'>🤖</h1>
                <h2 style='margin: 0.5rem 0; color: #262730;'>RAG Assistant</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
    
    @staticmethod
    def display_sidebar_capabilities():
        """Display capabilities at the end of sidebar."""
        with st.sidebar:
            st.divider()
            
            # Features section
            st.markdown("""
            **🚀 Capabilities**
            
            📚 **Document Search** - Query your knowledge base  
            🌐 **Web Search** - Real-time internet search  
            💻 **Code Generation** - AI-powered coding help  
            🔍 **Context-Aware** - Intelligent responses
            """)
    
    @staticmethod
    def display_main_header():
        """Display main page header."""
        st.title("✨ RAG Assistant")
        st.caption("Your AI-powered search companion")
        st.markdown("---")
    
    @staticmethod
    def display_chat_message(role: str, content: str):
        """Display a chat message with appropriate styling."""
        if role == "user":
            if "|img|" in content:
                text = content.split("|img|")[0]
                st.chat_message(name="User", avatar="👤").write(text)
                img = content.split("|img|")[1]
                st.image(img)
            else:
                st.chat_message(name="User", avatar="👤").write(content)
        elif role == "assistant":
            # Format response with line breaks
            formatted_content = content.replace('\n', '  \n')
            st.chat_message(name="Assistant", avatar="✨").write(formatted_content)
    
    @staticmethod
    def response_generator(response: str):
        """Generate response text with typing effect."""
        start = 0
        for i, char in enumerate(response):
            if char in ' \n':
                yield response[start:i+1]
                start = i + 1
            if char == ' ':
                time.sleep(0.05)
            elif char == '\n':
                time.sleep(0.05)
        
        # Yield the last word or remaining part
        if start < len(response):
            yield response[start:]


class HistoryManager:
    """Manages chat history functionality."""
    
    @staticmethod
    def initialize_history():
        """Initialize chat history in session state."""
        if "history" not in st.session_state:
            st.session_state["history"] = []
            dt = datetime.datetime.now()
            history_name = "history_" + dt.strftime("%y-%m-%d %H:%M:%S") + ".txt"
            st.session_state["history_name"] = history_name
    
    @staticmethod
    def get_history_files() -> List[str]:
        """Get list of available history files."""
        if not os.path.exists('History'):
            return ["None"]
        
        history_files = [file for file in os.listdir('History') if file.endswith(".txt")]
        history_files.sort(key=lambda x: os.path.getmtime(os.path.join('History', x)), reverse=True)
        history_files.insert(0, "None")
        
        return history_files
    
    @staticmethod
    def display_new_chat_button():
        """Display new chat button at the top."""
        with st.sidebar:
            # New chat button at the very top
            if st.button("➕ New Chat", type='primary', use_container_width=True):
                st.session_state["history"] = []
                if 'historySelect' in st.session_state:
                    st.session_state.historySelect = "None"
                st.rerun()
            
            st.divider()
    
    @staticmethod
    def display_history_selector():
        """Display history selector section."""
        with st.sidebar:
            st.markdown("### 📄 Chat History")
            
            # Get history files and selector
            history_files = HistoryManager.get_history_files()
            selected_history = st.selectbox(
                "Previous Conversations",
                options=history_files,
                key='historySelect',
                help="Select a previous conversation to view"
            )
            
            # Show number of available conversations
            if len(history_files) > 1:  # Exclude "None"
                st.caption(f"📊 {len(history_files) - 1} conversations available")
            
            return selected_history
    
    @staticmethod
    def read_chat_from_file(file_path: str) -> List[Tuple[str, str]]:
        """Read chat history from file."""
        chat_data = []
        try:
            with open(os.path.join('History', file_path), "r") as file:
                lines = file.read().split('""')
                for line in lines:
                    if line.strip():
                        try:
                            speaker, message = line.split(": ", 1)
                            speaker = speaker.replace('"', "")
                            message = message.strip('"')
                            chat_data.append((speaker.strip(), message.strip()))
                        except ValueError:
                            continue
        except FileNotFoundError:
            st.error(f"History file not found: {file_path}")
        except Exception as e:
            st.error(f"Error reading history file: {e}")
        
        return chat_data
    
    @staticmethod
    def write_to_file(text: str, role: str):
        """Write message to history file."""
        try:
            if not os.path.exists('History'):
                os.makedirs('History')
            with open(os.path.join('History', st.session_state["history_name"]), "a") as f:
                f.write(f'"{role}": "{text}""')
        except Exception as e:
            st.error(f"Error writing to history file: {e}")
    
    @staticmethod
    def display_chat_history(chat_data: List[Tuple[str, str]]):
        """Display chat history from data."""
        for speaker, message in chat_data:
            ChatInterface.display_chat_message(speaker, message)


class ModelSelector:
    """Handles model selection and configuration."""
    
    @staticmethod
    def display_workflow_selector(assistant):
        """Display workflow type selector with better UI."""
        from graph.core.types import WorkflowType
        
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            workflow_options = {
                "🎯 Complete RAG": WorkflowType.ALL,
                "🌐 Web Search Only": WorkflowType.WEB,
                "📚 Document Retrieval Only": WorkflowType.RETRIEVE
            }
            
            current_workflow = getattr(assistant, 'workflow_type', WorkflowType.ALL)
            
            selected = st.selectbox(
                "Workflow Mode",
                options=list(workflow_options.keys()),
                index=list(workflow_options.values()).index(current_workflow),
                help="Choose how the assistant should process your queries"
            )
            
            if workflow_options[selected] != current_workflow:
                if hasattr(assistant, 'update_workflow'):
                    assistant.update_workflow(workflow_options[selected])
                    st.rerun()
            
            st.divider()