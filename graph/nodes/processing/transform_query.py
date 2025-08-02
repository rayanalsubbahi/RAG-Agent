from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
        n_iterations += 1
        
        # Create prompt for query transformation - using original proven template
        prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for searching the internet. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question. \n
        Return a JSON object with the key 'better_question' and the value as the improved question. \n""",
            input_variables=["question"],
        )
        
        # Execute transformation with JSON output parser (original logic)
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            better_question_result = chain.invoke({"question": question})
            better_question = better_question_result["better_question"]
        except Exception as e:
            print(f"Error in query transformation: {e}")
            # Fallback to original question if transformation fails
            better_question = question
        
        # Update messages with improved question (original logic)
        messages.append({"role": "user", "content": better_question, "ai_message": True})
        print(f"Improved question: {better_question}")
        
        # Update state
        state["n_iterations"] = n_iterations
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages are present in state."""
        return "messages" in state and len(state["messages"]) > 0