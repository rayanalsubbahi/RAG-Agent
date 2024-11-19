from dotenv import load_dotenv

from graph.graph import RAGGraph

# Set Environment Variables
load_dotenv()

class Assistant:
    def __init__(self, llm, workflow_type, parse_str_output=True):
        self.llm = llm
        self.pipeline = RAGGraph(llm, workflow_type, parse_str_output)
        print("Assistant initialized")
    
    def chat(self, messages):
        return self.pipeline.invokePipeline(messages) 
    


