import asyncio
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message, DocumentGrader


class GradeDocumentsNode(LLMNode):
    """Node for grading document relevance to the query."""
    
    def __init__(self):
        super().__init__("grade_documents")
    
    def execute(self, state: GraphState) -> GraphState:
        """Grade documents for relevance to the query."""
        self.log("Grading documents")
        
        documents = state["documents"]
        messages = state["messages"]
        question = get_last_human_message(messages)
        parse_str_output = state["parse_str_output"]
        
        # Data model for grading
        class Grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        
        # Setup LLM with structured output if needed
        if not parse_str_output:
            llm_with_structured_output = self.llm.with_structured_output(Grade)
        
        # Create prompt template - using original proven template
        template = """You are a grader assessing relevance of a retrieved document from the internet to a user question. \n
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        if parse_str_output:
            template += """\nYou MUST include your response in <answer> tags. \n"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        
        # Chain setup
        if parse_str_output:
            chain = prompt | self.llm | StrOutputParser()
        else:
            chain = prompt | llm_with_structured_output 
        
        # Use the general async processor for document grading
        relevant_docs, is_transform_query = asyncio.run(
            DocumentGrader.grade_documents_async(
                documents=documents,
                question=question,
                chain=chain,
                parse_str_output=parse_str_output
            )
        )
        
        state["documents"] = relevant_docs
        state["is_transform_query"] = is_transform_query
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that required fields are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "documents" in state and len(state["documents"]) > 0 and
                "parse_str_output" in state)


