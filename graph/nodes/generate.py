from langchain.prompts import PromptTemplate
from utils import beautify_chat_history, getLastHumanMessage    


def generate(llm, state):
    '''generate from LLM'''
    print("----Generating answer----")
    
    messages = state["messages"]
    question = getLastHumanMessage(messages)
    chat_history = beautify_chat_history(messages[:-1])

    # prompt
    prompt = PromptTemplate(template="""You are a smart user. Your job is to generate an answer to the user question based on your knowledge.\n
                                    You need to carefully analyze the user question and the chat history to provide a relevant and accurate answer.\n
                                    Here is the chat history: {chat_history} \n
                                    Here is the user question: {question}
                                    """
                                    , input_variables=["question", "chat_history"])
    # Chain
    rag_chain = prompt | llm
    
    # Run
    generation = rag_chain.invoke({"question": question, "chat_history": chat_history})
    
    state["generation"] = generation
    state["documents"] = []
    
    return state