from langchain.prompts import PromptTemplate
from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message


class GenerateContextNode(LLMNode):
    """Node for generating answers based on retrieved documents context."""
    
    def __init__(self):
        super().__init__("generate_context")
    
    def execute(self, state: GraphState) -> GraphState:
        """Generate answer based on retrieved documents."""
        self.log("Generating answer based on context")
        
        documents = state["documents"]
        messages = state["messages"]
        question = get_last_human_message(messages)
        
        # Create prompt template for context-based generation
        prompt = PromptTemplate(
            template="""You are a smart user. Your job is to generate an answer to the user question based on the retrieved documents.
                        Here are the retrieved documents: 

                        {context} 

                        Here is the user question: {question}
                        You MUST accurately reflect the retrieved contents in your answer.
                        YOU MUST NOT use any external knowledge or information outside of the provided context.
                        You MUST NOT directly reference or mention the name of source documents.
                        
                        Be detailed and provide a comprehensive response to the user question.
                        """,
            input_variables=["context", "question"]
        )
        
        # Create chain and invoke
        rag_chain = prompt | self.llm
        generation = rag_chain.invoke({"context": documents, "question": question})
        
        # Update state
        state["generation"] = generation
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages and documents are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "documents" in state)


