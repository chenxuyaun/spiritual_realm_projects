"""
URL content fetching tool using trafilatura.

This module provides a wrapper around trafilatura for extracting
clean text content from web pages as part of the SearchQA workflow.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time
import concurrent.futures

from mm_orch.exceptions import NetworkError
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FetchedContent:
    """
    Fetched web page content.

    Attributes:
        url: Source URL
        content: Extracted text content
        title: Page title (if available)
        success: Whether fetch was successful
        error: Error message if failed
        timestamp: When content was fetched
    """

    url: str
    content: str
    title: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "content": self.content,
            "title": self.title,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class URLFetchTool:
    """
    URL content fetching tool using trafilatura.

    Extracts clean text content from web pages, removing
    boilerplate, navigation, and other non-content elements.
    """

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 3,
        include_tables: bool = True,
        include_links: bool = False,
    ):
        """
        Initialize the URL fetch tool.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            include_tables: Whether to include table content
            include_links: Whether to include link URLs in output
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.include_tables = include_tables
        self.include_links = include_links

    def fetch(self, url: str) -> FetchedContent:
        """
        Fetch and extract content from a URL.

        Args:
            url: URL to fetch

        Returns:
            FetchedContent object with extracted text
        """
        if not url or not url.strip():
            return FetchedContent(
                url=url or "", content="", success=False, error="Empty URL provided"
            )

        logger.debug(f"Fetching URL: {url[:100]}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                content, title = self._extract_content(url)

                if content:
                    logger.debug(
                        f"Content extracted from URL", url=url[:50], content_length=len(content)
                    )
                    return FetchedContent(url=url, content=content, title=title, success=True)
                else:
                    return FetchedContent(
                        url=url, content="", success=False, error="No content extracted"
                    )

            except ImportError as e:
                return FetchedContent(
                    url=url,
                    content="",
                    success=False,
                    error=f"trafilatura library not installed: {e}",
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Fetch attempt {attempt + 1} failed for {url[:50]}", error=str(e))
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Brief delay before retry

        return FetchedContent(
            url=url,
            content="",
            success=False,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
        )

    def _extract_content(self, url: str) -> tuple:
        """
        Extract content using trafilatura.

        Args:
            url: URL to extract from

        Returns:
            Tuple of (content, title)
        """
        try:
            import trafilatura
            from trafilatura.settings import use_config

            # Configure trafilatura
            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", str(self.timeout))

            # Download the page
            downloaded = trafilatura.fetch_url(url)

            if not downloaded:
                return "", None

            # Extract content
            content = trafilatura.extract(
                downloaded,
                include_tables=self.include_tables,
                include_links=self.include_links,
                output_format="txt",
                config=config,
            )

            # Try to get title
            title = None
            try:
                metadata = trafilatura.extract_metadata(downloaded)
                if metadata:
                    title = metadata.title
            except Exception:
                pass

            return content or "", title

        except ImportError:
            raise
        except Exception as e:
            raise NetworkError(f"Content extraction failed: {e}", context={"url": url})

    def fetch_multiple(self, urls: List[str], max_workers: int = 3) -> List[FetchedContent]:
        """
        Fetch content from multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch
            max_workers: Maximum concurrent workers

        Returns:
            List of FetchedContent objects
        """
        if not urls:
            return []

        logger.info(f"Fetching {len(urls)} URLs")

        results = []

        # Use ThreadPoolExecutor for concurrent fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.fetch, url): url for url in urls}

            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    url = future_to_url[future]
                    results.append(FetchedContent(url=url, content="", success=False, error=str(e)))

        # Sort results to match input order
        url_to_result = {r.url: r for r in results}
        ordered_results = [
            url_to_result.get(
                url, FetchedContent(url=url, content="", success=False, error="Not found")
            )
            for url in urls
        ]

        successful = sum(1 for r in ordered_results if r.success)
        logger.info(f"Fetched {successful}/{len(urls)} URLs successfully")

        return ordered_results


# Global instance
_url_fetch_tool: Optional[URLFetchTool] = None


def get_url_fetch_tool(timeout: int = 10, max_retries: int = 3) -> URLFetchTool:
    """
    Get the global URL fetch tool instance.

    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts

    Returns:
        URLFetchTool instance
    """
    global _url_fetch_tool
    if _url_fetch_tool is None:
        _url_fetch_tool = URLFetchTool(timeout=timeout, max_retries=max_retries)
    return _url_fetch_tool
