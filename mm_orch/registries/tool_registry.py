"""
Tool Registry for managing external tools.

This module provides a registry for discovering and managing external tools
such as web search, URL fetching, calculators, and translators. Tools are
registered with metadata describing their capabilities and parameters.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ToolMetadata:
    """
    Metadata for a registered tool.

    Attributes:
        name: Unique identifier for the tool
        capabilities: List of capability tags (e.g., ["search", "web"])
        description: Human-readable description of what the tool does
        parameters: Dictionary describing expected parameters and their types
    """

    name: str
    capabilities: List[str]
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not self.capabilities:
            raise ValueError(f"Tool '{self.name}' must have at least one capability")


class ToolRegistry:
    """
    Registry for external tools.

    Provides centralized registration and discovery of external tools
    with metadata about their capabilities and parameters.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(
        ...     name="web_search",
        ...     tool=search_function,
        ...     metadata=ToolMetadata(
        ...         name="web_search",
        ...         capabilities=["search", "web"],
        ...         description="Search the web using DuckDuckGo",
        ...         parameters={"query": "str", "max_results": "int"}
        ...     )
        ... )
        >>> tool = registry.get("web_search")
        >>> results = registry.find_by_capability("search")
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        logger.debug("Tool registry initialized")

    def register(self, name: str, tool: Callable, metadata: ToolMetadata) -> None:
        """
        Register a tool with metadata.

        Args:
            name: Unique identifier for the tool
            tool: Callable that implements the tool functionality
            metadata: ToolMetadata describing the tool

        Raises:
            ValueError: If name is empty or tool is already registered
            TypeError: If tool is not callable
        """
        if not name:
            raise ValueError("Tool name cannot be empty")

        if not callable(tool):
            raise TypeError(f"Tool '{name}' must be callable, got {type(tool)}")

        if name in self._tools:
            logger.warning(f"Tool '{name}' is already registered, overwriting")

        if metadata.name != name:
            raise ValueError(
                f"Metadata name '{metadata.name}' does not match registration name '{name}'"
            )

        self._tools[name] = tool
        self._metadata[name] = metadata

        logger.info(
            f"Registered tool '{name}'",
            capabilities=metadata.capabilities,
            description=metadata.description[:50],
        )

    def get(self, name: str) -> Callable:
        """
        Retrieve a tool by name.

        Args:
            name: Tool identifier

        Returns:
            The registered tool callable

        Raises:
            KeyError: If tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        return self._tools[name]

    def get_metadata(self, name: str) -> ToolMetadata:
        """
        Retrieve metadata for a tool.

        Args:
            name: Tool identifier

        Returns:
            ToolMetadata for the tool

        Raises:
            KeyError: If tool is not registered
        """
        if name not in self._metadata:
            raise KeyError(f"Tool '{name}' is not registered")

        return self._metadata[name]

    def find_by_capability(self, capability: str) -> List[str]:
        """
        Find tools with a specific capability.

        Args:
            capability: Capability tag to search for

        Returns:
            List of tool names that have the specified capability
        """
        matching_tools = [
            name for name, meta in self._metadata.items() if capability in meta.capabilities
        ]

        logger.debug(f"Found {len(matching_tools)} tools with capability '{capability}'")

        return matching_tools

    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: Tool identifier

        Returns:
            True if tool is registered, False otherwise
        """
        return name in self._tools

    def list_all(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of all registered tool names
        """
        return list(self._tools.keys())

    def unregister(self, name: str) -> None:
        """
        Unregister a tool.

        Args:
            name: Tool identifier

        Raises:
            KeyError: If tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        del self._tools[name]
        del self._metadata[name]

        logger.info(f"Unregistered tool '{name}'")


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        Global ToolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        _register_default_tools(_global_registry)
    return _global_registry


def _register_default_tools(registry: ToolRegistry) -> None:
    """
    Register default tools in the registry.

    Args:
        registry: ToolRegistry to register tools in
    """
    # Import tools
    try:
        from mm_orch.tools import get_web_search_tool, get_url_fetch_tool

        # Register web search tool
        web_search_tool = get_web_search_tool()
        registry.register(
            name="web_search",
            tool=web_search_tool.search,
            metadata=ToolMetadata(
                name="web_search",
                capabilities=["search", "web", "information_retrieval"],
                description="Search the web using DuckDuckGo search engine",
                parameters={
                    "query": "str - Search query string",
                    "max_results": "int - Maximum number of results (default: 5)",
                    "region": "str - Region code for search (default: wt-wt)",
                    "safesearch": "str - Safe search level (on/moderate/off)",
                },
            ),
        )

        # Register URL fetch tool
        url_fetch_tool = get_url_fetch_tool()
        registry.register(
            name="fetch_url",
            tool=url_fetch_tool.fetch,
            metadata=ToolMetadata(
                name="fetch_url",
                capabilities=["fetch", "web", "content_extraction"],
                description="Fetch and extract clean text content from web pages",
                parameters={
                    "url": "str - URL to fetch content from",
                },
            ),
        )

        # Register fetch_multiple as a separate tool
        registry.register(
            name="fetch_multiple",
            tool=url_fetch_tool.fetch_multiple,
            metadata=ToolMetadata(
                name="fetch_multiple",
                capabilities=["fetch", "web", "content_extraction", "batch"],
                description="Fetch content from multiple URLs concurrently",
                parameters={
                    "urls": "List[str] - List of URLs to fetch",
                    "max_workers": "int - Maximum concurrent workers (default: 3)",
                },
            ),
        )

        logger.info("Registered default tools: web_search, fetch_url, fetch_multiple")

    except ImportError as e:
        logger.warning(f"Could not register default tools: {e}")

    # Register placeholder tools
    _register_placeholder_tools(registry)


def _register_placeholder_tools(registry: ToolRegistry) -> None:
    """
    Register placeholder tools for future implementation.

    Args:
        registry: ToolRegistry to register tools in
    """

    def calculator_placeholder(*args, **kwargs):
        """Placeholder for calculator tool."""
        raise NotImplementedError("Calculator tool not yet implemented")

    def translator_placeholder(*args, **kwargs):
        """Placeholder for translator tool."""
        raise NotImplementedError("Translator tool not yet implemented")

    # Register calculator placeholder
    registry.register(
        name="calculator",
        tool=calculator_placeholder,
        metadata=ToolMetadata(
            name="calculator",
            capabilities=["math", "calculation", "arithmetic"],
            description="Perform basic arithmetic calculations (placeholder)",
            parameters={
                "expression": "str - Mathematical expression to evaluate",
            },
        ),
    )

    # Register translator placeholder
    registry.register(
        name="translator",
        tool=translator_placeholder,
        metadata=ToolMetadata(
            name="translator",
            capabilities=["translation", "language", "multilingual"],
            description="Translate text between languages (placeholder)",
            parameters={
                "text": "str - Text to translate",
                "source_lang": "str - Source language code",
                "target_lang": "str - Target language code",
            },
        ),
    )

    logger.info("Registered placeholder tools: calculator, translator")
