from langchain.prompts import PromptTemplate
from Utils import getLastHumanMessage

def generate_code(llm, state):
    '''generate answer based on documents'''
    print("----Generating code----")
    documents = state["documents"]
    messages = state["messages"]
    question = getLastHumanMessage(messages)
    
    # prompt
    prompt = PromptTemplate(template="""You are a smart programmer. Your job is to write a code snippet that answers the user question based on the retrieved documents.\n
                            You must only return the code snippet without any additional text so that the code can be executed directly.\n
                            Do not include any mark that indicates the language of the code.\n
                            Here are the retrieved documents: \n\n {context} \n\n
                            Here is the user question: {question}"""
                            , input_variables=["context", "question"]) 
    # Chain
    rag_chain = prompt | llm 

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    
    state["generation"] = generation
    
    return state