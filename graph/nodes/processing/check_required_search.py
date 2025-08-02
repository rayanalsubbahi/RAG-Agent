from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import extract_answer, get_last_human_message


class CheckRequiredSearchNode(LLMNode):
    """Node for determining the type of search required for a query."""
    
    def __init__(self):
        super().__init__("check_required_search")
    
    def execute(self, state: GraphState) -> GraphState:
        """Check what type of search is required for the query."""
        self.log("Checking type of search required")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        parse_str_output = state["parse_str_output"]
        
        # Define search type model
        class SearchType(BaseModel):
            """Define the type of search required."""
            search_type: str = Field(
                description="Value of 'own', 'custom_knowledge_base' or 'web' to determine the most appropriate search type"
            )
        
        # Setup LLM with structured output if needed
        if not parse_str_output:
            llm_with_structured_output = self.llm.with_structured_output(SearchType)
        
        # Create prompt template - using original proven template with better structure
        template = """You will be given a user's search query and need to determine the most appropriate search type to use in order to find the best information to address the query. 
                
                The three search types available are:
                1. Custom knowledge search - searches a curated knowledge base on the following two specific topics: 
                - Adversarial Attacks on Large Language Models
                - Large Language Models Powered Autonomous Agents
                2. Own model knowledge search - 
                searches the advanced language model's comprehensive knowledge spanning a vast range of topics including science, technology, history, culture, and more.
                This language model possesses remarkable reasoning, analysis, coding, and creative capabilities. Its knowledge base was trained on a massive corpus of
                high-quality data, allowing it to draw connections and synthesize information from diverse sources.
                3. Web search - searches the internet for the most current and up-to-date information
                
                The user has asked the following question:
                <question>
                {question}
                </question>
                
                Carefully analyze the query and determine which of the three search types would be most likely to surface the most relevant and useful information for the user.

                Decide the best source to answer the question:

                <answer>
                'own' if your own knowledge is sufficient
                'custom_knowledge_base' if the custom external knowledge base with the specific is likely to contain the answer
                'web' if a web search is necessary to find the most accurate and current information
                </answer>
                
                Return the search type as a string value of 'own', 'custom_knowledge_base' or 'web'
                """
        
        if parse_str_output:
            template += """\nYou MUST include your response in <answer> tags. \n"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
        )
        
        # Execute chain
        if parse_str_output:
            chain = prompt | self.llm | StrOutputParser()
            score = chain.invoke({"question": question})
            search_type_val = extract_answer(score)
        else:
            chain = prompt | llm_with_structured_output
            score = chain.invoke({"question": question})
            search_type_val = score.search_type

        print(f"Score: {score}")
        
        # Add original logging messages
        if search_type_val == "own":
            print("---WILL USE OWN KNOWLEDGE---")
        elif search_type_val == "custom_knowledge_base":
            print("---WILL RUN KNOWLEDGE BASE SEARCH---")
        else:
            print("---WILL RUN WEB SEARCH---")    
            
        # Update state - include documents initialization like original
        state["search_type"] = search_type_val
        state["documents"] = []
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages and parse_str_output are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "parse_str_output" in state)


