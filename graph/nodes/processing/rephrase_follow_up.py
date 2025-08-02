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
        
        # Create prompt template - using original proven template
        template = '''
        Given a chat history and a follow-up user input, process the input as follows:

        1. Determine if the user input is a question, query, or request (collectively referred to as "question") or not

        2. If it's a question:
        a) If the question relies on context from the chat history, rephrase it to be a standalone question that incorporates necessary context.
        b) If the question is clear and standalone, keep it as is.
        c) If the question is not relevant to the conversation, still consider rephrasing it to be a standalone question.
        d) Ensure the rephrased question maintains the conversation flow and context established by the previous history.
        e) Do NOT answer the question.

        3. If it's not a question:
        Return the user input exactly as it is, without any modifications.

        4. For all inputs, determine the type: either 'question' or 'statement'.

        5. Return the processed input and its type as rephrased_input and input_type.

        Chat History: {chat_history}
        Follow-up User Input: {question}
        '''
        
        if parse_str_output:
            template += """\nYou MUST only return a JSON object with the keys 'rephrased_input' and 'input_type'. \n"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "chat_history"]
        )
        
        # Chain setup
        if not parse_str_output:
            llm_structured_output = self.llm.with_structured_output(RephrasedInput)
            rag_chain = prompt | llm_structured_output
        else:
            rag_chain = prompt | self.llm | JsonOutputParser()

        # Execute chain
        response = rag_chain.invoke({"question": question, "chat_history": chat_history})
        
        if parse_str_output:
            rephrased_question = response["rephrased_input"]
            input_type = response["input_type"]
        else:
            rephrased_question = response.rephrased_input
            input_type = response.input_type
            
        print(f"Rephrased response: {response}")
        
        # Update state - append rephrased question like original
        messages.append({"role": "user", "content": rephrased_question, "ai_message": True})
        state["query_type"] = input_type
        return state
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages and parse_str_output are present in state."""
        return ("messages" in state and len(state["messages"]) > 0 and
                "parse_str_output" in state)


