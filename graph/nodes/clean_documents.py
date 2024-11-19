import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from utils import getLastHumanMessage


def clean_documents(llm, state):
    '''clean documents of any unwanted content'''
    print("----Cleaning documents----")
    messages = state["messages"]
    question = getLastHumanMessage(messages)
    documents = state["documents"]
    
    # Prompt
    prompt = PromptTemplate(
        template="""You are a document cleaner. Your job is to clean the retrieved document of any unwanted content that is not relevant to the user question. \n
        You must not change the relevant content and only copy it to the output. \n
        Here is the user question: {question} \n
        Here is the retrieved document: \n\n {context} \n\n
        Return directly the cleaned content of the documents. \n""",
        input_variables=["context", "question"],
    )
    
    # Chain
    chain = prompt | llm | StrOutputParser()
    
    # Run chain in parallel
    async def invoke_chain_async(chain, question, doc):
        print(f"Cleaning document")
        result = await asyncio.to_thread(chain.invoke, {"question": question, "context": doc.page_content})
        return Document(page_content=result)

    async def run_chain_in_parallel(chain, question, documents):
        cleaned_docs = []
        # Create a list of coroutines for each document
        tasks = [invoke_chain_async(chain, question, doc) for doc in documents]
        
        # Run all tasks concurrently and wait for all of them to complete
        results = await asyncio.gather(*tasks)
        
        # Collect the results
        cleaned_docs.extend(results)
        return cleaned_docs

    cleaned_docs = asyncio.run(run_chain_in_parallel(chain, question, documents))
    state["documents"] = cleaned_docs
    
    return state
