"""
External tool integrations (web search, scraping, etc.).

This module provides tools for interacting with external services
and resources, including web search and content extraction.
"""

from mm_orch.tools.web_search import (
    WebSearchTool,
    SearchResult,
    get_web_search_tool
)
from mm_orch.tools.fetch_url import (
    URLFetchTool,
    FetchedContent,
    get_url_fetch_tool
)

__all__ = [
    "WebSearchTool",
    "SearchResult",
    "get_web_search_tool",
    "URLFetchTool",
    "FetchedContent",
    "get_url_fetch_tool"
]
