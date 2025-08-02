from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import extract_answer, get_last_human_message


class CheckCodeGenerationNode(LLMNode):
    """Node for determining if code generation is required for a query."""
    
    def __init__(self):
        super().__init__("check_code_generation")
    
    def execute(self, state: GraphState) -> GraphState:
        """Check if the query requires code generation."""
        self.log("Deciding to generate code or general purpose generation")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        parse_str_output = state["parse_str_output"]
        
        # Define code requirement model
        class IsCodeRequired(BaseModel):
            """Binary score for code requirement check."""
            requires_code: str = Field(
                description="Score of 'yes' or 'no' to determine if a code implementation is required in response to the question or not"
            )
        
        # Setup LLM with structured output if needed
        if not parse_str_output:
            llm_with_structured_output = self.llm.with_structured_output(IsCodeRequired)
        
        # Create prompt template
        template = """You are a smart user. Your job is to determine whether the answer to question requires a code snippet or can be answered without a code.
                     Here is the user question: {question}
                     Give a binary score 'yes' or 'no' to indicate whether the user question requires a code snippet execution or not."""
        
        if parse_str_output:
            template += "\nYou MUST include your response in <answer> tags."
        
        prompt = PromptTemplate(template=template, input_variables=["question"])
        
        # Execute chain
        if parse_str_output:
            chain = prompt | self.llm | StrOutputParser()
            generation = chain.invoke({"question": question})
            answer = extract_answer(generation)
        else:
            chain = prompt | llm_with_structured_output
            generation = chain.invoke({"question": question})
            answer = generation.requires_code
        
        # Update state
        is_gen_code = answer.lower() == "yes"
        state["is_gen_code"] = is_gen_code
        
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages and parse_str_output are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "parse_str_output" in state)


