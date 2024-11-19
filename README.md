# Agentic RAG Assistant

![image](https://github.com/user-attachments/assets/9242b30a-45e2-440c-bdab-f929b17114f5)

This repository hosts an implementation of an **Agentic Retrieval-Augmented Generation (RAG) Assistant**. The assistant is designed to dynamically decide whether to:

- Utilize custom knowledge bases.
- Perform an internet search for real-time information.
- Rely on its internal knowledge as a fallback.

## Key Features

### 1. Advanced Retrieval Pipeline  
The retrieval process incorporates several advanced techniques to ensure accurate and relevant responses:
- **Relevance Checking**: Filters out irrelevant results to improve precision.
- **Data Cleaning**: Ensures input/output data is in optimal condition for processing.
- **Query Rewriting**: Refines user queries to enhance retrieval effectiveness.
- **Self-Reflection**: Evaluates its reasoning and responses for higher reliability.
- **Code Generation and Validation**: Dynamically generates code and validates results when required.

### 2. Conversational RAG Execution  
The assistant enables interactive conversations while the RAG pipeline operates seamlessly in the background.  
- **Short-term Memory**: Maintains chat history to provide contextual and coherent responses within a session.

### 3. Implementation Details  
- **LangChain**: Powers the assistant's language and retrieval capabilities.
- **LangGraph**: Manages and orchestrates the complex pipeline flow.
- **Streamlit-based UI**: Provides a user-friendly interface for interacting with the assistant.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rayanalsubbahi/RAG-Agent.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app
  ```bash
  streamlit run AssistantApp.py
