import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message, DocumentCleaner


class CleanDocumentsNode(LLMNode):
    """Node for cleaning and preprocessing retrieved documents using LLM."""
    
    def __init__(self):
        super().__init__("clean_documents")
    
    def execute(self, state: GraphState) -> GraphState:
        """Clean documents using LLM to remove irrelevant content."""
        self.log("Cleaning documents")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        documents = state["documents"]
        
        # Create prompt template - using original proven template
        prompt = PromptTemplate(
            template="""You are a document cleaner. Your job is to clean the retrieved document of any unwanted content that is not relevant to the user question. \n
        You must not change the relevant content and only copy it to the output. \n
        Here is the user question: {question} \n
        Here is the retrieved document: \n\n {context} \n\n
        Return directly the cleaned content of the documents. \n""",
            input_variables=["context", "question"],
        )
        
        # Chain setup
        chain = prompt | self.llm | StrOutputParser()
        
        # Use the general async processor for document cleaning
        cleaned_docs = asyncio.run(
            DocumentCleaner.clean_documents_async(
                documents=documents,
                question=question,
                chain=chain
            )
        )
        
        state["documents"] = cleaned_docs
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that required fields are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "documents" in state and len(state["documents"]) > 0)


