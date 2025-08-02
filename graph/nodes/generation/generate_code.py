from langchain.prompts import PromptTemplate
from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message


class GenerateCodeNode(LLMNode):
    """Node for generating code snippets based on retrieved documents."""
    
    def __init__(self):
        super().__init__("generate_code")
    
    def execute(self, state: GraphState) -> GraphState:
        """Generate code snippet based on documents and question."""
        self.log("Generating code")
        
        documents = state["documents"]
        messages = state["messages"]
        question = get_last_human_message(messages)
        
        # Create prompt template for code generation
        prompt = PromptTemplate(
            template="""You are a smart programmer. Your job is to write a code snippet that answers the user question based on the retrieved documents.
                        You must only return the code snippet without any additional text so that the code can be executed directly.
                        Do not include any mark that indicates the language of the code.
                        Here are the retrieved documents: 

                        {context} 

                        Here is the user question: {question}""",
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


