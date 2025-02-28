import os
from googlesearch import search
from googleapiclient.discovery import build
from langchain_community.document_loaders import WebBaseLoader
from utils import getLastHumanMessage
    
def web_search(state):
    '''search the web for more documents'''
    print("----Performing web search----")

    messages = state["messages"]
    #get last human message and chat history
    question = getLastHumanMessage(messages)
     
    # Search
    # urls = searchgoogle(question, max_results=3)
    urls = alternative_search(question, max_results=3)
  
    # Load docs in parallel    
    loader = WebBaseLoader(urls)
    documents = loader.aload()
    state["documents"] = documents
    return state

def searchgoogle(query, max_results=3):
    '''search the official google search engine'''
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_API_KEY"))
    res = service.cse().list(q=query, cx=os.getenv("GOOGLE_CSI_ID"), num=max_results).execute()
    
    urls = []
    for item in res['items']:
        if 'pdf' in item['link'] or 'perplexity' in item['link']:
            continue
        print(item['title'], item['link'])
        urls.append(item['link'])
    
    return urls

def alternative_search(query, max_results=3):
    '''search the web for more documents'''
    urls = []
    for url in search(query, num_results=max_results):
        if 'pdf' in url or 'perplexity' in url:
            continue
        print('Url', url)
        urls.append(url)
    urls = urls[:max_results]
    return urls