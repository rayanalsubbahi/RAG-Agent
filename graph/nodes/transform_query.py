from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils import getLastHumanMessage


def transform_query(llm, state):
    '''rewrite the query for a better answer'''
    print("----Transforming query----")
    messages = state["messages"]
    #get last human message and chat history
    question = getLastHumanMessage(messages)
    
    n_iterations = state["n_iterations"]
    n_iterations += 1
    
    # Create a prompt template with format instructions and the query
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

    # Prompt
    chain = prompt | llm | JsonOutputParser()
    better_question = chain.invoke({"question": question})
    better_question = better_question["better_question"]
    messages.append({"role": "user", "content": better_question, "ai_message": True})
    print(f"Improved question: {better_question}")

    return state