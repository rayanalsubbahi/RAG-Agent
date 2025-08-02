from langchain.prompts import PromptTemplate
from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message


class TransformQueryNode(LLMNode):
    """Node for transforming queries to improve search results."""
    
    def __init__(self):
        super().__init__("transform_query")
    
    def execute(self, state: GraphState) -> GraphState:
        """Transform the query for better search results."""
        self.log("Transforming query")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        n_iterations = state.get("n_iterations", 0)
        
        # Create prompt for query transformation
        prompt = PromptTemplate(
            template="""You are a query transformation expert. Your job is to rephrase the user's question to make it more effective for search.
                        
                        Original question: {question}
                        
                        Please provide a transformed version that:
                        1. Uses more specific and searchable terms
                        2. Expands abbreviations and acronyms
                        3. Adds relevant context keywords
                        4. Maintains the original intent
                        
                        Transformed question:""",
            input_variables=["question"]
        )
        
        # Execute transformation
        chain = prompt | self.llm
        transformed_result = chain.invoke({"question": question})
        
        # Update the last message with transformed query
        if hasattr(transformed_result, 'content'):
            transformed_query = transformed_result.content.strip()
        else:
            transformed_query = str(transformed_result).strip()
        
        # Update the messages with transformed query
        state["messages"][-1]["content"] = transformed_query
        state["is_transform_query"] = False  # Reset the flag
        state["n_iterations"] = n_iterations + 1
        
        print(f"Original: {question}")
        print(f"Transformed: {transformed_query}")
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages are present in state."""
        return "messages" in state and len(state["messages"]) > 0