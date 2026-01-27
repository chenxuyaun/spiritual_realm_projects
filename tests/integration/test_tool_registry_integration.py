"""
Integration tests for Tool Registry with actual tools.

Tests that the Tool Registry correctly integrates with
real web search and URL fetch tools.
"""

import pytest
from mm_orch.registries.tool_registry import get_tool_registry


class TestToolRegistryIntegration:
    """Integration tests for Tool Registry."""

    def test_web_search_tool_is_callable(self):
        """Test that registered web_search tool is callable."""
        registry = get_tool_registry()

        web_search = registry.get("web_search")
        assert callable(web_search)

    def test_fetch_url_tool_is_callable(self):
        """Test that registered fetch_url tool is callable."""
        registry = get_tool_registry()

        fetch_url = registry.get("fetch_url")
        assert callable(fetch_url)

    def test_fetch_multiple_tool_is_callable(self):
        """Test that registered fetch_multiple tool is callable."""
        registry = get_tool_registry()

        fetch_multiple = registry.get("fetch_multiple")
        assert callable(fetch_multiple)

    def test_find_web_tools(self):
        """Test finding all web-related tools."""
        registry = get_tool_registry()

        web_tools = registry.find_by_capability("web")
        assert "web_search" in web_tools
        assert "fetch_url" in web_tools
        assert "fetch_multiple" in web_tools

    def test_find_search_tools(self):
        """Test finding search-capable tools."""
        registry = get_tool_registry()

        search_tools = registry.find_by_capability("search")
        assert "web_search" in search_tools

    def test_find_fetch_tools(self):
        """Test finding fetch-capable tools."""
        registry = get_tool_registry()

        fetch_tools = registry.find_by_capability("fetch")
        assert "fetch_url" in fetch_tools
        assert "fetch_multiple" in fetch_tools

    def test_all_default_tools_registered(self):
        """Test that all expected default tools are registered."""
        registry = get_tool_registry()

        expected_tools = [
            "web_search",
            "fetch_url",
            "fetch_multiple",
            "calculator",
            "translator",
        ]

        for tool_name in expected_tools:
            assert registry.has(tool_name), f"Tool '{tool_name}' should be registered"

    def test_tool_metadata_completeness(self):
        """Test that all registered tools have complete metadata."""
        registry = get_tool_registry()

        for tool_name in registry.list_all():
            metadata = registry.get_metadata(tool_name)

            # Check required fields
            assert metadata.name == tool_name
            assert len(metadata.capabilities) > 0
            assert len(metadata.description) > 0
            assert isinstance(metadata.parameters, dict)
