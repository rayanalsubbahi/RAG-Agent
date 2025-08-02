import asyncio
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import extract_answer, get_last_human_message


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
        
        # Create prompt template
        template = """You are a document relevance grader. Given a user question and a retrieved document, 
                     you need to determine if the document is relevant to answer the user's question.
                     
                     Here is the user question: {question}
                     Here is the retrieved document: {document}
                     
                     Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
        
        if parse_str_output:
            template += "\nYou MUST include your response in <answer> tags."
        
        prompt = PromptTemplate(template=template, input_variables=["question", "document"])
        
        # Grade each document
        relevant_documents = []
        for document in documents:
            if parse_str_output:
                chain = prompt | self.llm | StrOutputParser()
                generation = chain.invoke({"question": question, "document": document})
                answer = extract_answer(generation)
            else:
                chain = prompt | llm_with_structured_output
                generation = chain.invoke({"question": question, "document": document})
                answer = generation.binary_score
            
            if answer.lower() == "yes":
                relevant_documents.append(document)
        
        # Update state with filtered documents
        state["documents"] = relevant_documents
        print(f"Filtered {len(documents)} documents to {len(relevant_documents)} relevant documents")
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that required fields are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "documents" in state and
                "parse_str_output" in state)


