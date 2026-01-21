"""
Web search tool using DuckDuckGo Search (ddgs).

This module provides a wrapper around the ddgs library for performing
web searches as part of the SearchQA workflow.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time

from mm_orch.exceptions import NetworkError
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class SearchResult:
    """
    Web search result.
    
    Attributes:
        title: Page title
        url: Page URL
        snippet: Text snippet from the page
        source: Search engine source
        timestamp: When the result was retrieved
    """
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "timestamp": self.timestamp
        }


class WebSearchTool:
    """
    Web search tool using DuckDuckGo Search.
    
    Provides methods for searching the web and retrieving results
    for use in the SearchQA workflow.
    """
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        Initialize the web search tool.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._ddgs = None
    
    def _get_ddgs(self):
        """Lazy initialization of DDGS client."""
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS()
            except ImportError:
                raise NetworkError(
                    "duckduckgo-search library not installed. "
                    "Install with: pip install duckduckgo-search",
                    context={"library": "duckduckgo-search"}
                )
        return self._ddgs
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> List[SearchResult]:
        """
        Perform a web search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            region: Region code for search (default: worldwide)
            safesearch: Safe search level ('on', 'moderate', 'off')
        
        Returns:
            List of SearchResult objects
        
        Raises:
            NetworkError: If search fails after retries
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
        
        logger.info(
            "Performing web search",
            query=query[:100],
            max_results=max_results
        )
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                ddgs = self._get_ddgs()
                
                # Perform the search
                raw_results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch
                ))
                
                # Convert to SearchResult objects
                results = []
                for r in raw_results:
                    result = SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", r.get("link", "")),
                        snippet=r.get("body", r.get("snippet", ""))
                    )
                    results.append(result)
                
                logger.info(
                    "Search completed",
                    query=query[:50],
                    results_count=len(results)
                )
                
                return results
                
            except ImportError as e:
                raise NetworkError(
                    f"Search library not available: {e}",
                    context={"query": query}
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Search attempt {attempt + 1} failed",
                    error=str(e),
                    query=query[:50]
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise NetworkError(
            f"Search failed after {self.max_retries} attempts: {last_error}",
            context={"query": query, "attempts": self.max_retries}
        )
    
    def search_news(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt"
    ) -> List[SearchResult]:
        """
        Search for news articles.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            region: Region code
        
        Returns:
            List of SearchResult objects
        """
        if not query or not query.strip():
            return []
        
        try:
            ddgs = self._get_ddgs()
            
            raw_results = list(ddgs.news(
                query,
                max_results=max_results,
                region=region
            ))
            
            results = []
            for r in raw_results:
                result = SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", r.get("link", "")),
                    snippet=r.get("body", r.get("excerpt", "")),
                    source="duckduckgo_news"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning(f"News search failed: {e}")
            return []


# Global instance
_web_search_tool: Optional[WebSearchTool] = None


def get_web_search_tool(timeout: int = 10, max_retries: int = 3) -> WebSearchTool:
    """
    Get the global web search tool instance.
    
    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    
    Returns:
        WebSearchTool instance
    """
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool(timeout=timeout, max_retries=max_retries)
    return _web_search_tool
