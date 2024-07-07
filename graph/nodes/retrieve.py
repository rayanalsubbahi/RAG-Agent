from Utils import getLastHumanMessage

def retrieve(retriever, state):
    '''retrieve content from documents'''
    print("----Retrieving documents----")
    
    messages = state["messages"]
    #get last human message and chat history
    question = getLastHumanMessage(messages)
    
    # Retrieve
    documents = retriever.vectorStore.get_relevant_documents(question)
    
    print(f"Retrieved {len(documents)} documents")
    state["documents"] = documents

    return state