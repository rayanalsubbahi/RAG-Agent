from googlesearch import search
from langchain_community.document_loaders import WebBaseLoader
from utils import getLastHumanMessage
    
def web_search(state):
    '''search the web for more documents'''
    print("----Performing web search----")

    messages = state["messages"]
    #get last human message and chat history
    question = getLastHumanMessage(messages)
    
    # Search
    urls = []
    documents=[]
    for url in search(question, num_results=3):
        if 'pdf' in url:
            continue
        print('Url', url)
        urls.append(url)
    
    #take the first 3 urls
    urls = urls[:3]
    # Load docs in parallel    
    loader = WebBaseLoader(urls)
    documents = loader.aload()
    # web_search_tool = TavilySearchResults(k=3)
    # docs = web_search_tool.invoke({"query": question})
    # docs = [Document(page_content=d["content"]) for d in docs]
    # documents.extend(docs)
    state["documents"] = documents
    
    return state