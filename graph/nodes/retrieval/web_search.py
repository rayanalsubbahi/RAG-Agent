import os
from typing import Optional
from googlesearch import search
from googleapiclient.discovery import build
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader

from graph.core.base_node import BaseNode
from graph.state import GraphState
from graph.core.utils import get_last_human_message
from graph.core.config import get_search_config


class WebSearchNode(BaseNode):
    """Node for performing web searches to retrieve documents."""
    
    def __init__(self, search_method: Optional[str] = None, max_results: Optional[int] = None):
        super().__init__("web_search")
        
        # Use config values if not specified
        search_config = get_search_config()
        self.search_method = search_method or search_config.web_search_method
        self.max_results = max_results or search_config.web_max_results
    
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
            try:
                loader = WebBaseLoader(urls)
                documents = loader.aload()
                state["documents"] = documents
                print(f"Successfully loaded {len(documents)} documents")
            except Exception as e:
                print(f"Error loading documents from URLs: {e}")
                state["documents"] = []
        else:
            print("No URLs found, setting empty documents")
            state["documents"] = []
        
        return state
    
    def _google_search(self, query: str, max_results: int) -> list:
        """Search using Google Custom Search API."""
        try:
            search_config = get_search_config()
            api_key = search_config.google_api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
            csi_id = search_config.google_csi_id or os.getenv("GOOGLE_CSI_ID")
            
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=query, cx=csi_id, num=max_results).execute()
            
            urls = []
            for item in res.get('items', []):
                if 'pdf' in item['link'] or 'perplexity' in item['link']:
                    continue
                print(item['title'], item['link'])
                urls.append(item['link'])
            
            return urls
        except Exception as e:
            print(f"Error in Google search: {e}")
            return []
    
    def _alternative_search(self, query: str, max_results: int) -> list:
        """Search using alternative search method."""
        try:
            urls = []
            for url in search(query, num_results=max_results):
                if 'pdf' in url or 'perplexity' in url:
                    continue
                print('Url', url)
                urls.append(url)
            return urls[:max_results]
        except Exception as e:
            print(f"Error in alternative search: {e}")
            return []
    
    def _tavily_search(self, query: str, max_results: int) -> list:
        """Search using Tavily search API."""
        try:
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
            for item in result.get('results', []):
                if 'pdf' in item['url'] or 'perplexity' in item['url']:
                    continue
                print(item['title'], item['url'])
                urls.append(item['url'])
            
            return urls[:max_results]
        except Exception as e:
            print(f"Error in Tavily search: {e}")
            return []
    
    def validate_inputs(self, state: GraphState) -> bool:
        """Validate that messages are present in state."""
        return "messages" in state and len(state["messages"]) > 0


