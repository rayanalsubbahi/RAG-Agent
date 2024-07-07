from langchain.prompts import PromptTemplate
from Utils import getLastHumanMessage


def generate_context(llm, state):
    '''generate answer based on documents'''
    print("----Generating answer based on context----")
    documents = state["documents"]
    messages = state["messages"]
    question = getLastHumanMessage(messages)

    # prompt
    prompt = PromptTemplate(template="""You are a smart user. Your job is to generate an answer to the user question based on the retrieved documents.\n
                                    Here are the retrieved documents: \n\n {context} \n\n
                                    Here is the user question: {question}
                                    You must not directly reference or mention the source documents or information in your answer.
                                    Be detailed and provide a comprehensive response to the user question.
                                    """
                                    , input_variables=["context", "question"])
    # Chain
    rag_chain = prompt | llm 

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    
    state["generation"] = generation
    
    return state