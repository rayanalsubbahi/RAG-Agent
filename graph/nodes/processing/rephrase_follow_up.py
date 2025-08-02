from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from graph.core.base_node import LLMNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message


class RephraseFollowUpNode(LLMNode):
    """Node for rephrasing follow-up questions to be standalone questions."""
    
    def __init__(self):
        super().__init__("rephrase_follow_up_question")
    
    def execute(self, state: GraphState) -> GraphState:
        """Rephrase follow-up question to be standalone given chat history."""
        self.log("Rephrasing follow-up question")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        chat_history = messages[:-1]
        parse_str_output = state["parse_str_output"]
        
        # If no chat history, just classify as question
        if len(chat_history) == 0:
            state["query_type"] = "question"
            return state
        
        # Data model for rephrasing
        class RephrasedInput(BaseModel):
            """Rephrased input for standalone question."""
            rephrased_input: str = Field(description="Rephrased standalone user input")
            input_type: str = Field(description="Type of user input as 'question' or 'statement'")
        
        # Create prompt template
        template = """Given a chat history and a follow-up user input, process the input as follows:

                     1. Determine if the user input is a question, query, or request (collectively referred to as "question") or not
                     2. If it's a question: rephrase it to be a standalone question by incorporating relevant context from the chat history
                     3. If it's not a question: return it as-is and classify it as a "statement"

                     Chat History:
                     {chat_history}

                     Follow-up Input: {question}

                     Provide your response in the following JSON format:
                     {{
                         "rephrased_input": "your rephrased standalone question or original statement",
                         "input_type": "question" or "statement"
                     }}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
        
        # Setup parser and chain
        parser = JsonOutputParser(pydantic_object=RephrasedInput)
        chain = prompt | self.llm | parser
        
        # Execute chain
        result = chain.invoke({
            "chat_history": chat_history,
            "question": question
        })
        
        # Update state
        state["messages"][-1]["content"] = result["rephrased_input"]
        state["query_type"] = result["input_type"]
        
        print(f"Rephrased input: {result['rephrased_input']}")
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages and parse_str_output are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "parse_str_output" in state)


