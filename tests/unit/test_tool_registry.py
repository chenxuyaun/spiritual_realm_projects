"""
Unit tests for Tool Registry.

Tests the registration, retrieval, and discovery of tools
in the Tool Registry system.
"""

import pytest
from mm_orch.registries.tool_registry import ToolRegistry, ToolMetadata, get_tool_registry


class TestToolMetadata:
    """Tests for ToolMetadata dataclass."""

    def test_valid_metadata_creation(self):
        """Test creating valid tool metadata."""
        metadata = ToolMetadata(
            name="test_tool",
            capabilities=["test", "example"],
            description="A test tool",
            parameters={"param1": "str"},
        )

        assert metadata.name == "test_tool"
        assert metadata.capabilities == ["test", "example"]
        assert metadata.description == "A test tool"
        assert metadata.parameters == {"param1": "str"}

    def test_metadata_with_default_parameters(self):
        """Test metadata creation with default empty parameters."""
        metadata = ToolMetadata(
            name="simple_tool", capabilities=["simple"], description="Simple tool"
        )

        assert metadata.parameters == {}

    def test_metadata_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            ToolMetadata(name="", capabilities=["test"], description="Test")

    def test_metadata_empty_capabilities_raises_error(self):
        """Test that empty capabilities list raises ValueError."""
        with pytest.raises(ValueError, match="must have at least one capability"):
            ToolMetadata(name="test", capabilities=[], description="Test")


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes empty."""
        registry = ToolRegistry()

        assert registry.list_all() == []

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        def sample_tool(x):
            return x * 2

        metadata = ToolMetadata(
            name="sample", capabilities=["math"], description="Sample tool"
        )

        registry.register("sample", sample_tool, metadata)

        assert registry.has("sample")
        assert "sample" in registry.list_all()

    def test_register_tool_with_mismatched_name_raises_error(self):
        """Test that mismatched metadata name raises ValueError."""
        registry = ToolRegistry()

        def sample_tool(x):
            return x

        metadata = ToolMetadata(
            name="different_name", capabilities=["test"], description="Test"
        )

        with pytest.raises(ValueError, match="does not match registration name"):
            registry.register("sample", sample_tool, metadata)

    def test_register_non_callable_raises_error(self):
        """Test that registering non-callable raises TypeError."""
        registry = ToolRegistry()

        metadata = ToolMetadata(name="invalid", capabilities=["test"], description="Test")

        with pytest.raises(TypeError, match="must be callable"):
            registry.register("invalid", "not_a_function", metadata)

    def test_register_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        registry = ToolRegistry()

        def sample_tool(x):
            return x

        # Empty name should raise error during metadata creation
        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            metadata = ToolMetadata(name="", capabilities=["test"], description="Test")

    def test_register_overwrites_existing_tool(self):
        """Test that registering same name overwrites previous tool."""
        registry = ToolRegistry()

        def tool_v1(x):
            return x

        def tool_v2(x):
            return x * 2

        metadata1 = ToolMetadata(name="tool", capabilities=["v1"], description="Version 1")
        metadata2 = ToolMetadata(name="tool", capabilities=["v2"], description="Version 2")

        registry.register("tool", tool_v1, metadata1)
        registry.register("tool", tool_v2, metadata2)

        # Should have the second version
        retrieved_tool = registry.get("tool")
        assert retrieved_tool(5) == 10  # tool_v2 behavior

        retrieved_metadata = registry.get_metadata("tool")
        assert retrieved_metadata.capabilities == ["v2"]

    def test_get_tool(self):
        """Test retrieving a registered tool."""
        registry = ToolRegistry()

        def add_tool(a, b):
            return a + b

        metadata = ToolMetadata(name="add", capabilities=["math"], description="Addition")

        registry.register("add", add_tool, metadata)

        retrieved = registry.get("add")
        assert retrieved(2, 3) == 5

    def test_get_nonexistent_tool_raises_error(self):
        """Test that getting non-existent tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_get_metadata(self):
        """Test retrieving tool metadata."""
        registry = ToolRegistry()

        def sample_tool(x):
            return x

        metadata = ToolMetadata(
            name="sample",
            capabilities=["test", "example"],
            description="A sample tool",
            parameters={"x": "int"},
        )

        registry.register("sample", sample_tool, metadata)

        retrieved_metadata = registry.get_metadata("sample")
        assert retrieved_metadata.name == "sample"
        assert retrieved_metadata.capabilities == ["test", "example"]
        assert retrieved_metadata.description == "A sample tool"
        assert retrieved_metadata.parameters == {"x": "int"}

    def test_get_metadata_nonexistent_raises_error(self):
        """Test that getting metadata for non-existent tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_metadata("nonexistent")

    def test_find_by_capability(self):
        """Test finding tools by capability."""
        registry = ToolRegistry()

        def tool1(x):
            return x

        def tool2(x):
            return x * 2

        def tool3(x):
            return x + 1

        registry.register(
            "tool1",
            tool1,
            ToolMetadata(name="tool1", capabilities=["math", "simple"], description="Tool 1"),
        )
        registry.register(
            "tool2",
            tool2,
            ToolMetadata(name="tool2", capabilities=["math", "complex"], description="Tool 2"),
        )
        registry.register(
            "tool3",
            tool3,
            ToolMetadata(name="tool3", capabilities=["simple"], description="Tool 3"),
        )

        # Find tools with "math" capability
        math_tools = registry.find_by_capability("math")
        assert set(math_tools) == {"tool1", "tool2"}

        # Find tools with "simple" capability
        simple_tools = registry.find_by_capability("simple")
        assert set(simple_tools) == {"tool1", "tool3"}

        # Find tools with non-existent capability
        none_tools = registry.find_by_capability("nonexistent")
        assert none_tools == []

    def test_has_tool(self):
        """Test checking if tool exists."""
        registry = ToolRegistry()

        def sample_tool(x):
            return x

        metadata = ToolMetadata(name="sample", capabilities=["test"], description="Test")

        registry.register("sample", sample_tool, metadata)

        assert registry.has("sample") is True
        assert registry.has("nonexistent") is False

    def test_list_all(self):
        """Test listing all registered tools."""
        registry = ToolRegistry()

        def tool1(x):
            return x

        def tool2(x):
            return x

        registry.register(
            "tool1", tool1, ToolMetadata(name="tool1", capabilities=["a"], description="Tool 1")
        )
        registry.register(
            "tool2", tool2, ToolMetadata(name="tool2", capabilities=["b"], description="Tool 2")
        )

        all_tools = registry.list_all()
        assert set(all_tools) == {"tool1", "tool2"}

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        def sample_tool(x):
            return x

        metadata = ToolMetadata(name="sample", capabilities=["test"], description="Test")

        registry.register("sample", sample_tool, metadata)
        assert registry.has("sample")

        registry.unregister("sample")
        assert not registry.has("sample")

        with pytest.raises(KeyError):
            registry.get("sample")

    def test_unregister_nonexistent_raises_error(self):
        """Test that unregistering non-existent tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")


class TestGlobalRegistry:
    """Tests for global registry instance."""

    def test_get_tool_registry_returns_singleton(self):
        """Test that get_tool_registry returns the same instance."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        assert registry1 is registry2

    def test_global_registry_has_default_tools(self):
        """Test that global registry has default tools registered."""
        registry = get_tool_registry()

        # Should have web_search and fetch_url
        assert registry.has("web_search")
        assert registry.has("fetch_url")
        assert registry.has("fetch_multiple")

        # Should have placeholder tools
        assert registry.has("calculator")
        assert registry.has("translator")

    def test_web_search_tool_metadata(self):
        """Test web_search tool metadata."""
        registry = get_tool_registry()

        metadata = registry.get_metadata("web_search")
        assert metadata.name == "web_search"
        assert "search" in metadata.capabilities
        assert "web" in metadata.capabilities
        assert "query" in metadata.parameters

    def test_fetch_url_tool_metadata(self):
        """Test fetch_url tool metadata."""
        registry = get_tool_registry()

        metadata = registry.get_metadata("fetch_url")
        assert metadata.name == "fetch_url"
        assert "fetch" in metadata.capabilities
        assert "web" in metadata.capabilities
        assert "url" in metadata.parameters

    def test_calculator_placeholder_raises_not_implemented(self):
        """Test that calculator placeholder raises NotImplementedError."""
        registry = get_tool_registry()

        calculator = registry.get("calculator")
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            calculator("2 + 2")

    def test_translator_placeholder_raises_not_implemented(self):
        """Test that translator placeholder raises NotImplementedError."""
        registry = get_tool_registry()

        translator = registry.get("translator")
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            translator("Hello", "en", "es")

    def test_find_by_capability_search(self):
        """Test finding tools with search capability."""
        registry = get_tool_registry()

        search_tools = registry.find_by_capability("search")
        assert "web_search" in search_tools

    def test_find_by_capability_fetch(self):
        """Test finding tools with fetch capability."""
        registry = get_tool_registry()

        fetch_tools = registry.find_by_capability("fetch")
        assert "fetch_url" in fetch_tools
        assert "fetch_multiple" in fetch_tools

    def test_find_by_capability_math(self):
        """Test finding tools with math capability."""
        registry = get_tool_registry()

        math_tools = registry.find_by_capability("math")
        assert "calculator" in math_tools
