import os
from googlesearch import search
from googleapiclient.discovery import build
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader

from graph.core.base_node import BaseNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message


class WebSearchNode(BaseNode):
    """Node for performing web searches to retrieve documents."""
    
    def __init__(self, search_method: str = "tavily", max_results: int = 3):
        super().__init__("web_search")
        self.search_method = search_method
        self.max_results = max_results
    
    def execute(self, state: GraphState) -> GraphState:
        """Perform web search and load documents."""
        self.log("Performing web search")
        
        messages = state["messages"]
        question = get_last_human_message(messages)
        
        # Search based on configured method
        if self.search_method == "google":
            urls = self._google_search(question, self.max_results)
        elif self.search_method == "alternative":
            urls = self._alternative_search(question, self.max_results)
        else:  # tavily (default)
            urls = self._tavily_search(question, self.max_results)
        
        # Load documents from URLs
        if urls:
            loader = WebBaseLoader(urls)
            documents = loader.aload()
            state["documents"] = documents
        else:
            state["documents"] = []
        
        return state
    
    def _google_search(self, query: str, max_results: int) -> list:
        """Search using Google Custom Search API."""
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_API_KEY"))
        res = service.cse().list(q=query, cx=os.getenv("GOOGLE_CSI_ID"), num=max_results).execute()
        
        urls = []
        for item in res['items']:
            if 'pdf' in item['link'] or 'perplexity' in item['link']:
                continue
            print(item['title'], item['link'])
            urls.append(item['link'])
        
        return urls
    
    def _alternative_search(self, query: str, max_results: int) -> list:
        """Search using alternative search method."""
        urls = []
        for url in search(query, num_results=max_results):
            if 'pdf' in url or 'perplexity' in url:
                continue
            print('Url', url)
            urls.append(url)
        return urls[:max_results]
    
    def _tavily_search(self, query: str, max_results: int) -> list:
        """Search using Tavily search API."""
        tool = TavilySearch(
            max_results=max_results,
            topic="general",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
        result = tool.invoke({"query": query})
        print(result)
        
        urls = []
        for item in result['results']:
            if 'pdf' in item['url'] or 'perplexity' in item['url']:
                continue
            print(item['title'], item['url'])
            urls.append(item['url'])
        
        return urls[:max_results]
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages are present in state."""
        return "messages" in state and len(state["messages"]) > 0


