from langchain.prompts import PromptTemplate
from Utils import getLastHumanMessage

def generate_based_on_error(llm, state):
    '''generate answer based on documents and stack trace'''
    print("----Generating new answer based on error----")
    documents = state["documents"]
    messages = state["messages"]
    question = getLastHumanMessage(messages)
    generation = state["generation"]
    stack_trace = state["stack_trace"]
    n_iterations = state["n_iterations"]
    n_iterations += 1
    
    # prompt
    prompt = PromptTemplate(
        template="""You are a code debugger trying to fix a code issue based on the stack trace of the code execution 
        and the retrieved documents from the internet to a user question. \n 
        Here are the retrieved documents: \n\n {context} \n\n
        Here is the user question: {question} \n
        Here is the code snippet that was executed: \n\n {generation} \n\n
        Here is the stack trace: \n\n {stack_trace} \n\n
        Provide a new code snippet that fixes the issue based on the stack trace and the documents.
        You must only return the code snippet without any additional text so that the code can be executed directly. \n
        Don not include any mark that indicates the language of the code. \n""",
        input_variables=["context", "question", "generation", "stack_trace"],
    )
        
    # Chain
    rag_chain = prompt | llm 
    
    # Run
    new_generation = rag_chain.invoke({"context": documents, "messages": messages, "generation": generation, "stack_trace": stack_trace, "question": question})
        
    state["generation"] = new_generation   
    state["stack_trace"] = None
        
    return state