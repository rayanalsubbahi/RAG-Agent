from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from Utils import getLastHumanMessage


def rephrase_follow_up_question(llm, state):
    """rephrase the follow-up question to be a standalone question given a chat history"""
    print("----Rephrasing follow-up question----")
    
    messages = state["messages"]
    question = getLastHumanMessage(messages)
    chat_history = messages[:-1]
    parse_str_output = state["parse_str_output"]
    
    if len(chat_history) == 0:
        state["query_type"] = "question"   
        return state
    
    class RephrasedInput(BaseModel):
        """Rephrased input for standalone question."""

        rephrased_input: str = Field(description="Rephrased standalone user input")
        input_type: str = Field(description="Type of user input as 'question' or 'statement'")

    # Prompt
    template = '''
        Given a chat history and a follow-up user input, process the input as follows:

        1. Determine if the user input is a question, query, or request (collectively referred to as "question") or not

        2. If it's a question:
        a) If the question relies on context from the chat history, rephrase it to be a standalone question that incorporates necessary context.
        b) If the question is clear and standalone, keep it as is.
        c) Ensure the rephrased question maintains the conversation flow and context established by the previous history.
        d) Do NOT answer the question.

        3. If it's not a question:
        Return the user input exactly as it is, without any modifications.

        4. For all inputs, determine the type: either 'question' or 'statement'.

        5. Return the processed input and its type as rephrased_input and input_type.

        Chat History: {chat_history}
        Follow-up User Input: {question}
        '''
    if parse_str_output:
        template += """\nYou MUST only return a JSON object with the keys 'rephrased_input' and 'input_type'. \n"""
        
    prompt =  PromptTemplate(
        # template= """Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
        #             The question must be rephrased in a way that maintains the conversation flow and context established by the previous history.
        #             Do NOT answer the question, just reformulate it if needed, otherwise return it as is. \
        #             Only return the final standalone question. \
        #             Here is the chat history: {chat_history} \n
        #             Here is the follow-up question: {question}
        #             """,
        template=template,
        input_variables=["question", "chat_history"],)
    
    # Chain
    if not parse_str_output:
        llm_structured_output = llm.with_structured_output(RephrasedInput)
        rag_chain = prompt | llm_structured_output
    else:
        rag_chain = prompt | llm | JsonOutputParser()

    # Run
    response = rag_chain.invoke({"question": question, "chat_history": chat_history})
    if parse_str_output:
        rephrased_question = response["rephrased_input"]
        input_type = response["input_type"]
    else:
        rephrased_question = response.rephrased_input
        input_type = response.input_type
        
    print(f"Rephrased response: {response}")
    messages.append({"role": "user", "content": rephrased_question, "ai_message": True})
    state["query_type"] = input_type
    
    return state