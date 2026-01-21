"""
Property-based tests for SearchQA workflow.

Tests the following properties from the design document:
- Property 2: Workflow execution step order
- Property 5: SearchQA degradation strategy

Uses Hypothesis for property-based testing with at least 100 iterations.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, MagicMock
from typing import List, Optional

from mm_orch.workflows.search_qa import (
    SearchQAWorkflow,
    SearchQAContext,
    SearchQAStep
)
from mm_orch.tools.web_search import SearchResult, WebSearchTool
from mm_orch.tools.fetch_url import FetchedContent, URLFetchTool
from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.exceptions import NetworkError


# Configure Hypothesis for at least 100 examples
settings.register_profile("property_tests", max_examples=100)
settings.load_profile("property_tests")


# Custom strategies for generating test data
@st.composite
def search_result_strategy(draw):
    """Strategy for generating SearchResult objects."""
    title = draw(st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' '
    )))
    url = draw(st.from_regex(r'https?://[a-z]+\.[a-z]+(/[a-z0-9]+)?', fullmatch=True))
    snippet = draw(st.text(min_size=10, max_size=500, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' .,!?'
    )))
    return SearchResult(title=title, url=url, snippet=snippet)


@st.composite
def fetched_content_strategy(draw, success: Optional[bool] = None):
    """Strategy for generating FetchedContent objects."""
    url = draw(st.from_regex(r'https?://[a-z]+\.[a-z]+(/[a-z0-9]+)?', fullmatch=True))
    is_success = success if success is not None else draw(st.booleans())
    
    if is_success:
        content = draw(st.text(min_size=50, max_size=2000, alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'Z'),
            whitelist_characters=' .,!?'
        )))
        return FetchedContent(url=url, content=content, success=True)
    else:
        error = draw(st.sampled_from([
            "Connection timeout",
            "404 Not Found",
            "SSL Error",
            "Content extraction failed"
        ]))
        return FetchedContent(url=url, content="", success=False, error=error)


@st.composite
def query_strategy(draw):
    """Strategy for generating valid query strings."""
    # Generate meaningful query-like strings
    words = draw(st.lists(
        st.text(min_size=2, max_size=15, alphabet=st.characters(
            whitelist_categories=('L',)
        )),
        min_size=2,
        max_size=10
    ))
    return " ".join(words)


class TestProperty2WorkflowStepOrder:
    """
    Property 2: Workflow execution step order
    
    For any workflow type and valid parameters, the workflow orchestrator
    should call components in the predefined step order, and each step's
    output should be the input for the next step.
    
    **Validates: Requirements 1.2, 2.1, 2.2, 2.3, 2.4**
    """
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_step_order_is_search_fetch_summarize_generate(self, query):
        """
        Feature: muai-orchestration-system, Property 2: Workflow execution step order
        
        For any valid query, the SearchQA workflow should execute steps
        in the order: search → fetch → summarize → generate.
        
        **Validates: Requirements 1.2, 2.1, 2.2, 2.3, 2.4**
        """
        assume(len(query.strip()) > 0)
        
        # Setup mocks
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(title="Test", url="https://test.com", snippet="Test snippet")
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url="https://test.com", content="Test content", success=True)
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        # Execute
        result = workflow.execute({"query": query})
        
        # Verify step order
        steps = result.metadata.get("steps", [])
        step_names = [s["name"] for s in steps]
        
        expected_order = ["search", "fetch", "summarize", "generate"]
        assert step_names == expected_order, \
            f"Steps should be in order {expected_order}, got {step_names}"
    
    @given(
        query=query_strategy(),
        num_results=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_search_output_is_fetch_input(self, query, num_results):
        """
        Feature: muai-orchestration-system, Property 2: Workflow execution step order
        
        The output of the search step (URLs) should be passed as input
        to the fetch step.
        
        **Validates: Requirements 2.1, 2.2**
        """
        assume(len(query.strip()) > 0)
        
        # Generate search results
        search_results = [
            SearchResult(
                title=f"Result {i}",
                url=f"https://example{i}.com",
                snippet=f"Snippet {i}"
            )
            for i in range(num_results)
        ]
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = search_results
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url=r.url, content=f"Content for {r.url}", success=True)
            for r in search_results
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        # Execute
        result = workflow.execute({"query": query})
        
        # Verify fetch was called with URLs from search
        fetch_tool.fetch_multiple.assert_called_once()
        called_urls = fetch_tool.fetch_multiple.call_args[0][0]
        expected_urls = [r.url for r in search_results]
        
        assert called_urls == expected_urls, \
            "Fetch should receive URLs from search results"
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_each_step_has_timing_info(self, query):
        """
        Feature: muai-orchestration-system, Property 2: Workflow execution step order
        
        Each step should record its execution duration.
        
        **Validates: Requirements 1.2**
        """
        assume(len(query.strip()) > 0)
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(title="Test", url="https://test.com", snippet="Test")
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url="https://test.com", content="Content", success=True)
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        steps = result.metadata.get("steps", [])
        
        for step in steps:
            assert "duration" in step, f"Step {step['name']} should have duration"
            assert step["duration"] >= 0, f"Step {step['name']} duration should be non-negative"


class TestProperty5SearchQADegradation:
    """
    Property 5: SearchQA degradation strategy
    
    For any SearchQA workflow execution, when any step (search, fetch,
    summarize, generate) fails, the system should attempt degradation
    strategies (skip failed URLs, use partial results), and the final
    WorkflowResult status should be 'partial' rather than 'failed'.
    
    **Validates: Requirements 2.5, 15.1**
    """
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_partial_fetch_failure_returns_partial_status(self, query):
        """
        Feature: muai-orchestration-system, Property 5: SearchQA degradation strategy
        
        When some URLs fail to fetch, the workflow should continue with
        successful fetches and return 'partial' status.
        
        **Validates: Requirements 2.5**
        """
        assume(len(query.strip()) > 0)
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(title="Good", url="https://good.com", snippet="Good snippet"),
            SearchResult(title="Bad", url="https://bad.com", snippet="Bad snippet")
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url="https://good.com", content="Good content", success=True),
            FetchedContent(url="https://bad.com", content="", success=False, error="Failed")
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        # Should have partial results, not complete failure
        assert result.result is not None, "Should have some result despite partial failure"
        assert result.metadata["degraded"] is True, "Should be marked as degraded"
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_all_fetch_failure_degrades_to_snippets(self, query):
        """
        Feature: muai-orchestration-system, Property 5: SearchQA degradation strategy
        
        When all URL fetches fail, the workflow should degrade to using
        search snippets as content.
        
        **Validates: Requirements 2.5, 15.1**
        """
        assume(len(query.strip()) > 0)
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(
                title="Result",
                url="https://example.com",
                snippet="This is useful snippet content for the query"
            )
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(
                url="https://example.com",
                content="",
                success=False,
                error="Connection failed"
            )
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        # Should still produce a result using snippets
        assert result.result is not None, "Should produce result using snippets"
        assert result.status == "partial", "Status should be partial"
        assert result.metadata["degraded"] is True, "Should be marked as degraded"
    
    @given(
        query=query_strategy(),
        num_successful=st.integers(min_value=1, max_value=5),
        num_failed=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=100)
    def test_degradation_preserves_successful_results(self, query, num_successful, num_failed):
        """
        Feature: muai-orchestration-system, Property 5: SearchQA degradation strategy
        
        When some fetches fail, all successful fetches should still be
        processed and included in the result.
        
        **Validates: Requirements 2.5**
        """
        assume(len(query.strip()) > 0)
        
        # Generate search results - at least one successful fetch
        search_results = []
        fetch_results = []
        
        for i in range(num_successful):
            search_results.append(SearchResult(
                title=f"Success {i}",
                url=f"https://success{i}.com",
                snippet=f"Success snippet {i}"
            ))
            fetch_results.append(FetchedContent(
                url=f"https://success{i}.com",
                content=f"Success content {i} with enough text to process",
                success=True
            ))
        
        for i in range(num_failed):
            search_results.append(SearchResult(
                title=f"Fail {i}",
                url=f"https://fail{i}.com",
                snippet=f"Fail snippet {i}"
            ))
            fetch_results.append(FetchedContent(
                url=f"https://fail{i}.com",
                content="",
                success=False,
                error="Failed"
            ))
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = search_results
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = fetch_results
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        # Verify fetched count matches successful fetches
        assert result.metadata["fetched_count"] == num_successful, \
            f"Should have {num_successful} successful fetches"
        
        # Should have a result (either from content or snippets)
        assert result.result is not None, "Should produce a result"
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_no_search_results_returns_partial(self, query):
        """
        Feature: muai-orchestration-system, Property 5: SearchQA degradation strategy
        
        When search returns no results, the workflow should return
        'partial' status with appropriate error message.
        
        **Validates: Requirements 2.5**
        """
        assume(len(query.strip()) > 0)
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = []  # No results
        
        fetch_tool = Mock(spec=URLFetchTool)
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        assert result.status == "partial", "Should return partial status"
        assert result.error is not None, "Should have error message"
        assert "search" in result.error.lower() or "result" in result.error.lower(), \
            "Error should mention search or results"
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_degradation_metadata_tracks_failures(self, query):
        """
        Feature: muai-orchestration-system, Property 5: SearchQA degradation strategy
        
        When degradation occurs, the metadata should track which steps
        were degraded and why.
        
        **Validates: Requirements 2.5, 15.1**
        """
        assume(len(query.strip()) > 0)
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(title="Test", url="https://test.com", snippet="Test snippet")
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url="https://test.com", content="", success=False, error="Failed")
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        # Check that steps track degradation
        steps = result.metadata.get("steps", [])
        degraded_steps = [s for s in steps if s.get("degraded", False)]
        
        assert len(degraded_steps) > 0, "Should have at least one degraded step"
        
        # Fetch step should be marked as degraded
        fetch_step = next((s for s in steps if s["name"] == "fetch"), None)
        assert fetch_step is not None, "Should have fetch step"
        assert fetch_step["degraded"] is True, "Fetch step should be marked as degraded"


class TestSearchQAResultStructure:
    """
    Additional property tests for result structure consistency.
    
    **Validates: Requirements 1.4**
    """
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_result_always_has_required_fields(self, query):
        """
        Feature: muai-orchestration-system, Property 4: Result structure completeness
        
        For any execution, the result should always have required fields.
        
        **Validates: Requirements 1.4**
        """
        assume(len(query.strip()) > 0)
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(title="Test", url="https://test.com", snippet="Test")
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url="https://test.com", content="Content", success=True)
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        # Check required fields
        assert hasattr(result, 'result'), "Result should have 'result' field"
        assert hasattr(result, 'metadata'), "Result should have 'metadata' field"
        assert hasattr(result, 'status'), "Result should have 'status' field"
        
        # Check metadata structure
        assert "workflow" in result.metadata, "Metadata should have 'workflow'"
        assert "query" in result.metadata, "Metadata should have 'query'"
        assert "steps" in result.metadata, "Metadata should have 'steps'"
        assert "degraded" in result.metadata, "Metadata should have 'degraded'"
    
    @given(query=query_strategy())
    @settings(max_examples=100)
    def test_status_is_valid_value(self, query):
        """
        Feature: muai-orchestration-system, Property 4: Result structure completeness
        
        The status field should always be one of the valid values.
        
        **Validates: Requirements 1.4**
        """
        assume(len(query.strip()) > 0)
        
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(title="Test", url="https://test.com", snippet="Test")
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url="https://test.com", content="Content", success=True)
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        result = workflow.execute({"query": query})
        
        valid_statuses = {"success", "partial", "failed"}
        assert result.status in valid_statuses, \
            f"Status should be one of {valid_statuses}, got {result.status}"
