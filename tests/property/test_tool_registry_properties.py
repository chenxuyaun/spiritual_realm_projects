"""
Property-based tests for Tool Registry.

Tests universal properties that must hold for all tool registrations
and capability queries using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from mm_orch.registries.tool_registry import ToolRegistry, ToolMetadata


# Strategy for generating valid tool names
tool_names = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',)),  # Exclude surrogates
    min_size=1,
    max_size=30
).filter(lambda x: x.strip() != "")

# Strategy for generating capability lists
capabilities = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=('Cs',)),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.strip() != ""),
    min_size=1,
    max_size=5,
    unique=True
)

# Strategy for generating descriptions (simpler to avoid slowness)
descriptions = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',)),
    min_size=1,
    max_size=100
)

# Strategy for generating parameter dictionaries (smaller to avoid slowness)
parameters = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(blacklist_categories=('Cs',)),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.strip() != ""),
    values=st.text(
        alphabet=st.characters(blacklist_categories=('Cs',)),
        min_size=1,
        max_size=50
    ),
    max_size=5
)


def sample_tool_function(*args, **kwargs):
    """Sample tool function for testing."""
    return "result"


class TestToolMetadataPersistence:
    """
    Property 6: Tool Metadata Persistence
    
    For any tool registered in Tool_Registry, retrieving the tool by name
    must return the same callable, and the stored metadata must contain
    the name and capabilities provided during registration.
    
    Validates: Requirements 4.3, 4.4
    """

    # Feature: extensible-orchestration-phase-b, Property 6: Tool Metadata Persistence
    @given(
        name=tool_names,
        caps=capabilities,
        desc=descriptions,
        params=parameters
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None
    )
    def test_tool_metadata_persistence(self, name, caps, desc, params):
        """
        Property 6: Tool Metadata Persistence
        
        Verifies that registered tools and their metadata are correctly
        persisted and can be retrieved with all original information intact.
        """
        registry = ToolRegistry()
        
        # Create metadata
        metadata = ToolMetadata(
            name=name,
            capabilities=caps,
            description=desc,
            parameters=params
        )
        
        # Register tool
        registry.register(name, sample_tool_function, metadata)
        
        # Retrieve tool - must return the same callable
        retrieved_tool = registry.get(name)
        assert retrieved_tool is sample_tool_function, \
            "Retrieved tool must be the same callable that was registered"
        
        # Retrieve metadata - must contain original name and capabilities
        retrieved_metadata = registry.get_metadata(name)
        assert retrieved_metadata.name == name, \
            "Retrieved metadata name must match registered name"
        assert retrieved_metadata.capabilities == caps, \
            "Retrieved metadata capabilities must match registered capabilities"
        assert retrieved_metadata.description == desc, \
            "Retrieved metadata description must match registered description"
        assert retrieved_metadata.parameters == params, \
            "Retrieved metadata parameters must match registered parameters"

    # Feature: extensible-orchestration-phase-b, Property 6: Tool Metadata Persistence
    @given(
        tools=st.lists(
            st.tuples(tool_names, capabilities, descriptions, parameters),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]  # Unique by name
        )
    )
    @settings(max_examples=100)
    def test_multiple_tools_metadata_persistence(self, tools):
        """
        Property 6: Tool Metadata Persistence (Multiple Tools)
        
        Verifies that multiple tools can be registered and all their
        metadata persists correctly without interference.
        """
        registry = ToolRegistry()
        
        # Register all tools
        for name, caps, desc, params in tools:
            metadata = ToolMetadata(
                name=name,
                capabilities=caps,
                description=desc,
                parameters=params
            )
            registry.register(name, sample_tool_function, metadata)
        
        # Verify all tools are retrievable with correct metadata
        for name, caps, desc, params in tools:
            # Tool must be retrievable
            retrieved_tool = registry.get(name)
            assert retrieved_tool is sample_tool_function
            
            # Metadata must be correct
            retrieved_metadata = registry.get_metadata(name)
            assert retrieved_metadata.name == name
            assert retrieved_metadata.capabilities == caps
            assert retrieved_metadata.description == desc
            assert retrieved_metadata.parameters == params

    # Feature: extensible-orchestration-phase-b, Property 6: Tool Metadata Persistence
    @given(
        name=tool_names,
        caps=capabilities,
        desc=descriptions
    )
    @settings(max_examples=100)
    def test_tool_persistence_after_operations(self, name, caps, desc):
        """
        Property 6: Tool Metadata Persistence (After Operations)
        
        Verifies that tool metadata persists correctly even after
        other registry operations like listing and capability queries.
        """
        registry = ToolRegistry()
        
        metadata = ToolMetadata(
            name=name,
            capabilities=caps,
            description=desc
        )
        
        registry.register(name, sample_tool_function, metadata)
        
        # Perform various operations
        _ = registry.list_all()
        _ = registry.has(name)
        for cap in caps:
            _ = registry.find_by_capability(cap)
        
        # Metadata must still be intact
        retrieved_metadata = registry.get_metadata(name)
        assert retrieved_metadata.name == name
        assert retrieved_metadata.capabilities == caps
        assert retrieved_metadata.description == desc


class TestCapabilityQueryCorrectness:
    """
    Property 8: Capability Query Correctness (Tool Registry part)
    
    For any capability string and Tool_Registry, querying by that capability
    must return only items whose capabilities list contains that capability,
    and must return all such items.
    
    Validates: Requirements 4.4
    """

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        tools=st.lists(
            st.tuples(tool_names, capabilities, descriptions),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]  # Unique by name
        ),
        query_capability=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() != "")
    )
    @settings(max_examples=100)
    def test_capability_query_returns_only_matching_tools(self, tools, query_capability):
        """
        Property 8: Capability Query Correctness
        
        Verifies that capability queries return exactly the tools that
        have the queried capability - no more, no less.
        """
        registry = ToolRegistry()
        
        # Register all tools
        for name, caps, desc in tools:
            metadata = ToolMetadata(
                name=name,
                capabilities=caps,
                description=desc
            )
            registry.register(name, sample_tool_function, metadata)
        
        # Query by capability
        matching_tools = registry.find_by_capability(query_capability)
        
        # Determine expected matches
        expected_matches = {
            name for name, caps, _ in tools
            if query_capability in caps
        }
        
        # Verify correctness
        assert set(matching_tools) == expected_matches, \
            f"Query for '{query_capability}' must return exactly the tools with that capability"
        
        # Verify no false positives
        for tool_name in matching_tools:
            metadata = registry.get_metadata(tool_name)
            assert query_capability in metadata.capabilities, \
                f"Tool '{tool_name}' in results must have capability '{query_capability}'"
        
        # Verify no false negatives
        for name, caps, _ in tools:
            if query_capability in caps:
                assert name in matching_tools, \
                    f"Tool '{name}' with capability '{query_capability}' must be in results"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        tools=st.lists(
            st.tuples(tool_names, capabilities, descriptions),
            min_size=2,
            max_size=15,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_completeness(self, tools):
        """
        Property 8: Capability Query Completeness
        
        Verifies that for each capability present in any tool,
        querying for that capability returns all tools with it.
        """
        registry = ToolRegistry()
        
        # Register all tools
        for name, caps, desc in tools:
            metadata = ToolMetadata(
                name=name,
                capabilities=caps,
                description=desc
            )
            registry.register(name, sample_tool_function, metadata)
        
        # Collect all unique capabilities
        all_capabilities = set()
        for _, caps, _ in tools:
            all_capabilities.update(caps)
        
        # For each capability, verify query returns all matching tools
        for capability in all_capabilities:
            matching_tools = registry.find_by_capability(capability)
            
            # Find expected tools with this capability
            expected_tools = {
                name for name, caps, _ in tools
                if capability in caps
            }
            
            assert set(matching_tools) == expected_tools, \
                f"Query for '{capability}' must return all tools with that capability"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        tools=st.lists(
            st.tuples(tool_names, capabilities, descriptions),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        ),
        nonexistent_capability=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() != "")
    )
    @settings(max_examples=100)
    def test_capability_query_empty_for_nonexistent(self, tools, nonexistent_capability):
        """
        Property 8: Capability Query for Nonexistent Capability
        
        Verifies that querying for a capability that no tool has
        returns an empty list.
        """
        registry = ToolRegistry()
        
        # Register all tools
        for name, caps, desc in tools:
            metadata = ToolMetadata(
                name=name,
                capabilities=caps,
                description=desc
            )
            registry.register(name, sample_tool_function, metadata)
        
        # Collect all capabilities
        all_capabilities = set()
        for _, caps, _ in tools:
            all_capabilities.update(caps)
        
        # Assume the query capability doesn't exist in any tool
        assume(nonexistent_capability not in all_capabilities)
        
        # Query should return empty list
        matching_tools = registry.find_by_capability(nonexistent_capability)
        assert matching_tools == [], \
            f"Query for nonexistent capability '{nonexistent_capability}' must return empty list"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        tools=st.lists(
            st.tuples(tool_names, capabilities, descriptions),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_consistency(self, tools):
        """
        Property 8: Capability Query Consistency
        
        Verifies that querying the same capability multiple times
        returns consistent results.
        """
        registry = ToolRegistry()
        
        # Register all tools
        for name, caps, desc in tools:
            metadata = ToolMetadata(
                name=name,
                capabilities=caps,
                description=desc
            )
            registry.register(name, sample_tool_function, metadata)
        
        # Collect all capabilities
        all_capabilities = set()
        for _, caps, _ in tools:
            all_capabilities.update(caps)
        
        # Query each capability multiple times
        for capability in all_capabilities:
            result1 = registry.find_by_capability(capability)
            result2 = registry.find_by_capability(capability)
            result3 = registry.find_by_capability(capability)
            
            # Results must be consistent
            assert set(result1) == set(result2) == set(result3), \
                f"Multiple queries for '{capability}' must return consistent results"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        tools=st.lists(
            st.tuples(tool_names, capabilities, descriptions),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_intersection(self, tools):
        """
        Property 8: Capability Query Intersection
        
        Verifies that tools with multiple capabilities appear in
        query results for each of their capabilities.
        """
        registry = ToolRegistry()
        
        # Register all tools
        for name, caps, desc in tools:
            metadata = ToolMetadata(
                name=name,
                capabilities=caps,
                description=desc
            )
            registry.register(name, sample_tool_function, metadata)
        
        # For each tool with multiple capabilities
        for name, caps, _ in tools:
            if len(caps) > 1:
                # Tool must appear in results for each of its capabilities
                for capability in caps:
                    matching_tools = registry.find_by_capability(capability)
                    assert name in matching_tools, \
                        f"Tool '{name}' with capability '{capability}' must appear in query results"
