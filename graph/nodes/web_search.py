import os
from googlesearch import search
from googleapiclient.discovery import build
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from utils import getLastHumanMessage
    
def web_search(state):
    '''search the web for more documents'''
    print("----Performing web search----")

    messages = state["messages"]
    #get last human message and chat history
    question = getLastHumanMessage(messages)
     
    # Search
    # urls = google_search(question, max_results=3)
    # urls = alternative_search(question, max_results=3)
    urls = tavily_search(question, max_results=1)
  
    # Load docs in parallel    
    loader = WebBaseLoader(urls)
    documents = loader.aload()
    state["documents"] = documents
    return state

def google_search(query, max_results=3):
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

def alternative_search(query, max_results=5):
    '''search the web for more documents'''
    urls = []
    for url in search(query, num_results=max_results):
        if 'pdf' in url or 'perplexity' in url:
            continue
        print('Url', url)
        urls.append(url)
    urls = urls[:max_results]
    return urls

def tavily_search(query, max_results=5):
    '''search the web for more documents'''
    tool = TavilySearch(
        max_results=max_results,
        topic="general",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        # include_image_descriptions=False,
        # search_depth="basic",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )
    result = tool.invoke({"query": query})  
    print(result)
    
    urls = []
    for item in result['results']:
        if 'pdf' in item['url'] or 'perplexity' in item['url']:
            continue
        print(item['title'], item['url'])
        urls.append(item['url'])

    urls = urls[:max_results]
    return urls