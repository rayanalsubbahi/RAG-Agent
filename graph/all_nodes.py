#nodes
from graph.nodes.rephrase_follow_up_question import rephrase_follow_up_question
from graph.nodes.check_required_search import check_required_search
from graph.nodes.check_code_generation import check_code_generation
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.nodes.clean_documents import clean_documents
from graph.nodes.grade_documents import grade_documents
from graph.nodes.transform_query import transform_query
from graph.nodes.generate_code import generate_code
from graph.nodes.generate_based_context import generate_context
from graph.nodes.generate import generate
from graph.nodes.execute_code import execute_code 
from graph.nodes.generate_based_error import generate_based_on_error


class GraphNodes:
    def __init__(self, llm, retriever) -> None:
        self.llm = llm
        self.retriever = retriever
        
    def rephrase_follow_up_question(self, state):
        return rephrase_follow_up_question(self.llm, state)
    
    def check_required_search(self, state):
        return check_required_search(self.llm, state)
    
    def check_code_generation(self, state):
        return check_code_generation(self.llm, state)
    
    def retrieve(self, state):
        return retrieve(self.retriever, state)
    
    def web_search(self, state):
        return web_search(state)

    def clean_documents(self, state):
        return clean_documents(self.llm, state)
    
    def grade_documents(self, state):
        return grade_documents(self.llm, state)
    
    def transform_query(self, state):
        return transform_query(self.llm, state)
    
    def generate_code(self, state):
        return generate_code(self.llm, state)
    
    def generate_context(self, state):
        return generate_context(self.llm, state)
    
    def generate(self, state):
        return generate(self.llm, state)
    
    def execute_code(self, state):
        return execute_code(state)
    
    def generate_based_on_error(self, state):
        return generate_based_on_error(self.llm, state)
    
    