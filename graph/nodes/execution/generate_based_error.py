from langchain.prompts import PromptTemplate
from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message


class GenerateBasedErrorNode(LLMNode):
    """Node for generating corrected code based on execution errors."""
    
    def __init__(self):
        super().__init__("generate_based_on_error")
    
    def execute(self, state: GraphState) -> GraphState:
        """Generate new code based on stack trace error."""
        self.log("Generating new answer based on error")
        
        documents = state["documents"]
        messages = state["messages"]
        question = get_last_human_message(messages)
        generation = state["generation"]
        stack_trace = state["stack_trace"]
        n_iterations = state.get("n_iterations", 0)
        
        # Increment iteration counter
        n_iterations += 1
        
        # Create prompt template for error-based code generation
        prompt = PromptTemplate(
            template="""You are a code debugger trying to fix a code issue based on the stack trace of the code execution 
                        and the retrieved documents from the internet to a user question.
                        Here are the retrieved documents: 

                        {context} 

                        Here is the user question: {question}
                        Here is the code snippet that was executed: 

                        {generation} 

                        Here is the stack trace: 

                        {stack_trace} 

                        Provide a new code snippet that fixes the issue based on the stack trace and the documents.
                        You must only return the code snippet without any additional text so that the code can be executed directly.
                        Do not include any mark that indicates the language of the code.""",
            input_variables=["context", "question", "generation", "stack_trace"]
        )
        
        # Create chain and invoke
        rag_chain = prompt | self.llm
        new_generation = rag_chain.invoke({
            "context": documents,
            "question": question,
            "generation": generation,
            "stack_trace": stack_trace
        })
        
        # Update state
        state["generation"] = new_generation
        state["stack_trace"] = None  # Clear the stack trace for next iteration
        state["n_iterations"] = n_iterations
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that required fields are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "documents" in state and
                "generation" in state and
                "stack_trace" in state and state["stack_trace"])


