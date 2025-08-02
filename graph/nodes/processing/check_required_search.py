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
        
        # Create prompt template
        template = """You will be given a user's search query and need to determine the most appropriate search type to use in order to find the best information to address the query. 
                    
                    The three search types available are:
                    1. Custom knowledge search - searches a curated knowledge base on the following two specific topics: 
                       - Adversarial Attacks on Large Language Models
                       - Large Language Models Powered Autonomous Agents
                    2. Own model knowledge search - searches the advanced language model's comprehensive knowledge spanning a vast range of topics including science, technology, history, culture, and more.
                       This language model possesses remarkable reasoning, analysis, coding, and creative capabilities. Its knowledge base was trained on a massive corpus of
                       high-quality data, allowing it to draw connections and synthesize information from diverse sources.
                    3. Web search - searches the internet for the most current and up-to-date information
                    
                    The user has asked the following question:
                    <question>
                    {question}
                    </question>
                    
                    The answer MUST be one of the following:
                    - 'own' for Own model knowledge search
                    - 'custom_knowledge_base' for Custom knowledge search
                    - 'web' for Web search
                    
                    Determine which search type would be most appropriate for answering this question effectively."""
        
        if parse_str_output:
            template += "\nYou MUST include your response in <answer> tags."
        
        prompt = PromptTemplate(template=template, input_variables=["question"])
        
        # Execute chain
        if parse_str_output:
            chain = prompt | self.llm | StrOutputParser()
            generation = chain.invoke({"question": question})
            answer = extract_answer(generation)
        else:
            try:
                chain = prompt | llm_with_structured_output
                generation = chain.invoke({"question": question})
                answer = generation.search_type
            except Exception as e:
                print(f"Structured output failed: {e}")
                # Fallback to string parsing
                chain = prompt | self.llm | StrOutputParser()
                generation = chain.invoke({"question": question})
                answer = str(generation).lower()
                # Simple parsing for search type
                if "web" in answer:
                    answer = "web"
                elif "knowledge" in answer or "custom" in answer:
                    answer = "custom_knowledge_base"
                else:
                    answer = "own"
        
        # Update state
        state["search_type"] = answer
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages and parse_str_output are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "parse_str_output" in state)


