import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from graph.graph import RAGGraph

# Set Environment Variables
load_dotenv()

class Assistant:
    def __init__(self, llm):
        self.llm = llm
        self.pipeline = RAGGraph(llm)
        print("Assistant initialized")
    
    def chat(self, messages):
        return self.pipeline.invokePipeline(messages) 
    
    
if __name__ == "__main__":
    # LLM
    # llm = ChatAnthropic(model='claude-3.5-sonnet-20240620', anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    # llm = ChatAnthropic(model='claude-3-haiku-20240307', anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    # llm = ChatOpenAI(model='gpt-3.5-turbo-1106', openai_api_key=os.getenv("OPENAI_API_KEY"))
    # llm = ChatCohere(model='command-r-plus', cohere_api_key=os.getenv("COHERE_API_KEY"))
    # llm = ChatNVIDIA(model='nvidia / nemotron-4-340b-instruct', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
    # llm = ChatNVIDIA(model='meta / llama3-70b-instruct', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
    
    llm = ChatNVIDIA(model='google / gemma-2-27b-it', nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
    
    # Create Assistant
    assistant = Assistant(llm)
    # Chat
    question = "What are the types of LLM agenet memory?"
    messages = []
        # Initial message
    human_message = {"role": "user", "content": question}
    messages.append(human_message)
        
    answer = assistant.chat(messages)
    print(f"Answer: {answer}")


