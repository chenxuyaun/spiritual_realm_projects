"""
Unit tests for SearchQA workflow.

Tests the SearchQA workflow implementation including:
- Parameter validation
- Step execution order
- Degradation strategies
- Result structure
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from mm_orch.workflows.search_qa import (
    SearchQAWorkflow,
    SearchQAContext,
    SearchQAStep
)
from mm_orch.tools.web_search import SearchResult, WebSearchTool
from mm_orch.tools.fetch_url import FetchedContent, URLFetchTool
from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.exceptions import ValidationError


class TestSearchQAWorkflow:
    """Tests for SearchQAWorkflow class."""
    
    @pytest.fixture
    def mock_search_tool(self):
        """Create a mock search tool."""
        tool = Mock(spec=WebSearchTool)
        tool.search.return_value = [
            SearchResult(
                title="Test Result 1",
                url="https://example.com/1",
                snippet="This is test snippet 1"
            ),
            SearchResult(
                title="Test Result 2",
                url="https://example.com/2",
                snippet="This is test snippet 2"
            )
        ]
        return tool
    
    @pytest.fixture
    def mock_fetch_tool(self):
        """Create a mock fetch tool."""
        tool = Mock(spec=URLFetchTool)
        tool.fetch_multiple.return_value = [
            FetchedContent(
                url="https://example.com/1",
                content="This is the full content of page 1. It contains detailed information.",
                title="Test Result 1",
                success=True
            ),
            FetchedContent(
                url="https://example.com/2",
                content="This is the full content of page 2. More detailed information here.",
                title="Test Result 2",
                success=True
            )
        ]
        return tool
    
    @pytest.fixture
    def workflow(self, mock_search_tool, mock_fetch_tool):
        """Create a workflow instance with mocked tools."""
        return SearchQAWorkflow(
            search_tool=mock_search_tool,
            fetch_tool=mock_fetch_tool,
            model_manager=None  # No model manager for unit tests
        )
    
    def test_workflow_type(self, workflow):
        """Test that workflow has correct type."""
        assert workflow.workflow_type == WorkflowType.SEARCH_QA
        assert workflow.name == "SearchQA"
    
    def test_required_parameters(self, workflow):
        """Test required parameters list."""
        required = workflow.get_required_parameters()
        assert "query" in required
    
    def test_optional_parameters(self, workflow):
        """Test optional parameters with defaults."""
        optional = workflow.get_optional_parameters()
        assert "max_results" in optional
        assert "include_sources" in optional
        assert optional["include_sources"] is True
    
    def test_validate_parameters_valid(self, workflow):
        """Test parameter validation with valid parameters."""
        params = {"query": "What is Python?"}
        assert workflow.validate_parameters(params) is True
    
    def test_validate_parameters_empty_query(self, workflow):
        """Test parameter validation with empty query."""
        params = {"query": ""}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_missing_query(self, workflow):
        """Test parameter validation with missing query."""
        params = {}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_invalid_max_results(self, workflow):
        """Test parameter validation with invalid max_results."""
        params = {"query": "test", "max_results": 0}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_execute_success(self, workflow, mock_search_tool, mock_fetch_tool):
        """Test successful workflow execution."""
        params = {"query": "What is Python?"}
        result = workflow.execute(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.status in ["success", "partial"]
        assert result.result is not None
        assert "steps" in result.metadata
        assert len(result.metadata["steps"]) == 4  # search, fetch, summarize, generate
    
    def test_execute_step_order(self, workflow, mock_search_tool, mock_fetch_tool):
        """Test that steps execute in correct order."""
        params = {"query": "What is Python?"}
        result = workflow.execute(params)
        
        steps = result.metadata["steps"]
        step_names = [s["name"] for s in steps]
        
        assert step_names == ["search", "fetch", "summarize", "generate"]
    
    def test_execute_includes_sources(self, workflow):
        """Test that sources are included in metadata."""
        params = {"query": "What is Python?", "include_sources": True}
        result = workflow.execute(params)
        
        assert "sources" in result.metadata
        assert len(result.metadata["sources"]) > 0
    
    def test_execute_no_sources(self, workflow):
        """Test execution without sources."""
        params = {"query": "What is Python?", "include_sources": False}
        result = workflow.execute(params)
        
        assert "sources" not in result.metadata
    
    def test_execute_no_search_results(self, workflow, mock_search_tool):
        """Test handling of no search results."""
        mock_search_tool.search.return_value = []
        
        params = {"query": "What is Python?"}
        result = workflow.execute(params)
        
        assert result.status == "partial"
        assert "No search results" in result.error
    
    def test_execute_fetch_failure_degradation(self, workflow, mock_fetch_tool):
        """Test degradation when fetch fails."""
        mock_fetch_tool.fetch_multiple.return_value = [
            FetchedContent(
                url="https://example.com/1",
                content="",
                success=False,
                error="Connection timeout"
            ),
            FetchedContent(
                url="https://example.com/2",
                content="",
                success=False,
                error="Connection timeout"
            )
        ]
        
        params = {"query": "What is Python?"}
        result = workflow.execute(params)
        
        # Should degrade to using snippets
        assert result.metadata["degraded"] is True
        assert result.status == "partial"
    
    def test_execute_partial_fetch_success(self, workflow, mock_fetch_tool):
        """Test handling of partial fetch success."""
        mock_fetch_tool.fetch_multiple.return_value = [
            FetchedContent(
                url="https://example.com/1",
                content="Full content here",
                success=True
            ),
            FetchedContent(
                url="https://example.com/2",
                content="",
                success=False,
                error="Failed"
            )
        ]
        
        params = {"query": "What is Python?"}
        result = workflow.execute(params)
        
        # Should continue with partial results
        assert result.result is not None
        assert result.metadata["fetched_count"] == 1
    
    def test_run_method_with_validation(self, workflow):
        """Test the run method which includes validation."""
        params = {"query": "What is Python?"}
        result = workflow.run(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.execution_time is not None
        assert result.execution_time >= 0
    
    def test_run_method_validation_failure(self, workflow):
        """Test run method with invalid parameters."""
        params = {"query": ""}
        result = workflow.run(params)
        
        assert result.status == "failed"
        assert "Validation error" in result.error


class TestSearchQAContext:
    """Tests for SearchQAContext dataclass."""
    
    def test_context_initialization(self):
        """Test context initialization."""
        ctx = SearchQAContext(query="test query")
        
        assert ctx.query == "test query"
        assert ctx.search_results == []
        assert ctx.fetched_contents == []
        assert ctx.summaries == []
        assert ctx.answer == ""
        assert ctx.steps == []
        assert ctx.degraded is False
    
    def test_add_step(self):
        """Test adding steps to context."""
        ctx = SearchQAContext(query="test")
        step = SearchQAStep(name="search", success=True)
        
        ctx.add_step(step)
        
        assert len(ctx.steps) == 1
        assert ctx.steps[0].name == "search"
    
    def test_add_degraded_step(self):
        """Test that degraded step marks context as degraded."""
        ctx = SearchQAContext(query="test")
        step = SearchQAStep(name="fetch", success=False, degraded=True)
        
        ctx.add_step(step)
        
        assert ctx.degraded is True


class TestSearchQAStep:
    """Tests for SearchQAStep dataclass."""
    
    def test_step_initialization(self):
        """Test step initialization."""
        step = SearchQAStep(name="search", success=True)
        
        assert step.name == "search"
        assert step.success is True
        assert step.duration == 0.0
        assert step.input_count == 0
        assert step.output_count == 0
        assert step.error is None
        assert step.degraded is False
    
    def test_step_with_all_fields(self):
        """Test step with all fields set."""
        step = SearchQAStep(
            name="fetch",
            success=False,
            duration=1.5,
            input_count=5,
            output_count=3,
            error="Some URLs failed",
            degraded=True
        )
        
        assert step.name == "fetch"
        assert step.success is False
        assert step.duration == 1.5
        assert step.input_count == 5
        assert step.output_count == 3
        assert step.error == "Some URLs failed"
        assert step.degraded is True


class TestSearchQADegradation:
    """Tests for SearchQA degradation strategies."""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow with mocked tools."""
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(
                title="Result",
                url="https://example.com",
                snippet="Test snippet content"
            )
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(
                url="https://example.com",
                content="",
                success=False,
                error="Failed"
            )
        ]
        
        return SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
    
    def test_degrade_to_snippets(self, workflow):
        """Test degradation to using search snippets."""
        params = {"query": "test query"}
        result = workflow.execute(params)
        
        # Should use snippets when fetch fails
        assert result.metadata["degraded"] is True
        assert result.result is not None
    
    def test_simple_summarize_short_content(self):
        """Test simple summarization with short content."""
        workflow = SearchQAWorkflow()
        content = "This is short content."
        
        summary = workflow._simple_summarize(content, max_length=100)
        
        assert summary == content
    
    def test_simple_summarize_long_content(self):
        """Test simple summarization with long content."""
        workflow = SearchQAWorkflow()
        content = "This is a sentence. " * 50  # Long content
        
        summary = workflow._simple_summarize(content, max_length=100)
        
        assert len(summary) <= 103  # max_length + "..."
    
    def test_simple_summarize_sentence_boundary(self):
        """Test that summarization respects sentence boundaries."""
        workflow = SearchQAWorkflow()
        content = "First sentence. Second sentence. Third sentence is longer."
        
        summary = workflow._simple_summarize(content, max_length=40)
        
        # Should end at a sentence boundary
        assert summary.endswith('.') or summary.endswith('...')


class TestSearchQAIntegration:
    """Integration-style tests for SearchQA workflow."""
    
    def test_full_workflow_mock(self):
        """Test full workflow with all mocked components."""
        # Setup mocks
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(
                title="Python Tutorial",
                url="https://python.org/tutorial",
                snippet="Python is a programming language"
            )
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(
                url="https://python.org/tutorial",
                content="Python is a high-level programming language. It is easy to learn.",
                title="Python Tutorial",
                success=True
            )
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        # Execute
        result = workflow.run({"query": "What is Python?"})
        
        # Verify
        assert result.status in ["success", "partial"]
        assert result.result is not None
        assert "Python" in result.result or "programming" in result.result.lower()
        
        # Verify all steps executed
        steps = result.metadata["steps"]
        assert len(steps) == 4
        
        # Verify step success
        search_step = next(s for s in steps if s["name"] == "search")
        assert search_step["success"] is True
        assert search_step["output_count"] == 1
    
    def test_workflow_metrics(self):
        """Test that workflow collects metrics."""
        search_tool = Mock(spec=WebSearchTool)
        search_tool.search.return_value = [
            SearchResult(title="Test", url="https://test.com", snippet="Test")
        ]
        
        fetch_tool = Mock(spec=URLFetchTool)
        fetch_tool.fetch_multiple.return_value = [
            FetchedContent(url="https://test.com", content="Test content", success=True)
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool
        )
        
        # Execute multiple times
        workflow.run({"query": "test 1"})
        workflow.run({"query": "test 2"})
        
        metrics = workflow.get_metrics()
        
        assert metrics["execution_count"] == 2
        assert metrics["total_execution_time"] > 0
        assert metrics["average_execution_time"] > 0
