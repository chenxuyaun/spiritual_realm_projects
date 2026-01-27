"""
Unit tests for new Phase B3 workflow variants.

Tests the three new workflows:
- summarize_url: Fetch and summarize a single URL
- search_qa_fast: Fast search QA with reduced summarization
- search_qa_strict_citations: Search QA with citation validation
"""

import pytest
from unittest.mock import Mock

from mm_orch.orchestration import (
    State,
    GraphExecutor,
    SimpleTracer,
    WebSearchStep,
    FetchUrlStep,
    SummarizeStep,
    AnswerGenerateStep
)
from mm_orch.orchestration.workflow_steps import (
    FetchSingleUrlStep,
    FetchTopNStep,
    CitationValidationStep,
    SummarizeToAnswerStep,
    AnswerGenerateFromDocsStep
)
from mm_orch.registries.workflow_definitions import (
    create_summarize_url_workflow,
    create_search_qa_fast_workflow,
    create_search_qa_strict_citations_workflow
)


@pytest.fixture
def mock_runtime():
    """Create a mock runtime."""
    runtime = Mock()
    runtime.model_manager = None  # Use fallback methods
    return runtime


@pytest.fixture
def mock_search_tool():
    """Create a mock search tool."""
    from mm_orch.tools.web_search import SearchResult
    
    tool = Mock()
    tool.search = Mock(return_value=[
        SearchResult(
            title="Python Tutorial",
            url="https://example.com/python",
            snippet="Learn Python programming basics."
        ),
        SearchResult(
            title="Python Advanced",
            url="https://example.com/python-advanced",
            snippet="Advanced Python concepts."
        ),
        SearchResult(
            title="Python Best Practices",
            url="https://example.com/python-best",
            snippet="Best practices for Python development."
        )
    ])
    return tool


@pytest.fixture
def mock_fetch_tool():
    """Create a mock fetch tool."""
    from mm_orch.tools.fetch_url import FetchedContent
    
    tool = Mock()
    
    # For single URL fetch
    tool.fetch = Mock(return_value=FetchedContent(
        url="https://example.com/article",
        content="This is a detailed article about Python programming. " * 50,
        title="Python Article",
        success=True
    ))
    
    # For multiple URL fetch
    tool.fetch_multiple = Mock(return_value=[
        FetchedContent(
            url="https://example.com/python",
            content="Python is a high-level programming language. " * 30,
            title="Python Tutorial",
            success=True
        ),
        FetchedContent(
            url="https://example.com/python-advanced",
            content="Advanced Python features include decorators and generators. " * 30,
            title="Python Advanced",
            success=True
        )
    ])
    
    return tool


class TestSummarizeUrlWorkflow:
    """Tests for summarize_url workflow."""
    
    def test_workflow_definition(self):
        """Test that workflow definition is created correctly."""
        workflow = create_summarize_url_workflow()
        
        assert workflow.name == "summarize_url"
        assert "fetch_single_url" in [node.step_name for node in workflow.graph.values()]
        assert "summarize_to_answer" in [node.step_name for node in workflow.graph.values()]
        assert "fetch" in workflow.required_capabilities
        assert "summarize" in workflow.required_capabilities
    
    def test_fetch_single_url_step(self, mock_runtime, mock_fetch_tool):
        """Test FetchSingleUrlStep execution."""
        step = FetchSingleUrlStep(fetch_tool=mock_fetch_tool)
        
        state: State = {
            "question": "https://example.com/article"
        }
        
        result = step.run(state, mock_runtime)
        
        assert "docs" in result
        assert "citations" in result
        assert "https://example.com/article" in result["docs"]
        assert "https://example.com/article" in result["citations"]
        assert len(result["docs"]["https://example.com/article"]) > 0
    
    def test_summarize_to_answer_step(self, mock_runtime):
        """Test SummarizeToAnswerStep execution."""
        step = SummarizeToAnswerStep()
        
        state: State = {
            "docs": {
                "https://example.com/article": "This is a long article. " * 100
            },
            "citations": ["https://example.com/article"]
        }
        
        result = step.run(state, mock_runtime)
        
        assert "final_answer" in result
        assert "citations" in result
        assert len(result["final_answer"]) > 0
        assert result["citations"] == ["https://example.com/article"]
    
    def test_complete_workflow(self, mock_runtime, mock_fetch_tool):
        """Test complete summarize_url workflow execution."""
        # Create steps
        fetch_step = FetchSingleUrlStep(fetch_tool=mock_fetch_tool)
        summarize_step = SummarizeToAnswerStep()
        
        # Create step registry
        step_registry = {
            "fetch_single_url": fetch_step,
            "summarize_to_answer": summarize_step
        }
        
        # Get workflow definition
        workflow = create_summarize_url_workflow()
        
        # Create executor
        tracer = SimpleTracer()
        executor = GraphExecutor(step_registry, tracer=tracer)
        
        # Initial state
        initial_state: State = {
            "question": "https://example.com/article"
        }
        
        # Execute
        final_state = executor.execute(workflow.graph, initial_state, mock_runtime)
        
        # Verify output
        assert "final_answer" in final_state
        assert "citations" in final_state
        assert len(final_state["final_answer"]) > 0
        assert "https://example.com/article" in final_state["citations"]
        
        # Verify traces
        traces = tracer.get_traces()
        assert len(traces) == 2
        assert all(t["success"] for t in traces)


class TestSearchQaFastWorkflow:
    """Tests for search_qa_fast workflow."""
    
    def test_workflow_definition(self):
        """Test that workflow definition is created correctly."""
        workflow = create_search_qa_fast_workflow()
        
        assert workflow.name == "search_qa_fast"
        assert "web_search" in [node.step_name for node in workflow.graph.values()]
        assert "fetch_top_n" in [node.step_name for node in workflow.graph.values()]
        assert "answer_generate_from_docs" in [node.step_name for node in workflow.graph.values()]
        # Should NOT have summarize step
        assert "summarize" not in [node.step_name for node in workflow.graph.values()]
    
    def test_fetch_top_n_step(self, mock_runtime, mock_fetch_tool):
        """Test FetchTopNStep execution."""
        step = FetchTopNStep(fetch_tool=mock_fetch_tool, n=2)
        
        state: State = {
            "search_results": [
                {"title": "Result 1", "url": "https://example.com/1"},
                {"title": "Result 2", "url": "https://example.com/2"},
                {"title": "Result 3", "url": "https://example.com/3"}
            ]
        }
        
        result = step.run(state, mock_runtime)
        
        assert "docs" in result
        # Should only fetch top 2
        assert len(result["docs"]) <= 2
    
    def test_complete_workflow(self, mock_runtime, mock_search_tool, mock_fetch_tool):
        """Test complete search_qa_fast workflow execution."""
        # Create steps
        search_step = WebSearchStep(search_tool=mock_search_tool)
        fetch_step = FetchTopNStep(fetch_tool=mock_fetch_tool, n=2)
        answer_step = AnswerGenerateFromDocsStep()
        
        # Create step registry
        step_registry = {
            "web_search": search_step,
            "fetch_top_n": fetch_step,
            "answer_generate_from_docs": answer_step
        }
        
        # Get workflow definition
        workflow = create_search_qa_fast_workflow()
        
        # Create executor
        tracer = SimpleTracer()
        executor = GraphExecutor(step_registry, tracer=tracer)
        
        # Initial state
        initial_state: State = {
            "question": "What is Python?"
        }
        
        # Execute
        final_state = executor.execute(workflow.graph, initial_state, mock_runtime)
        
        # Verify output
        assert "final_answer" in final_state
        assert "citations" in final_state
        assert "search_results" in final_state
        assert "docs" in final_state
        
        # Should skip summarization
        assert "summaries" not in final_state
        
        # Verify traces (should be 3 steps, not 4)
        traces = tracer.get_traces()
        assert len(traces) == 3
        assert all(t["success"] for t in traces)


class TestSearchQaStrictCitationsWorkflow:
    """Tests for search_qa_strict_citations workflow."""
    
    def test_workflow_definition(self):
        """Test that workflow definition is created correctly."""
        workflow = create_search_qa_strict_citations_workflow()
        
        assert workflow.name == "search_qa_strict_citations"
        assert "citation_validation" in [node.step_name for node in workflow.graph.values()]
    
    def test_citation_validation_step_pass(self, mock_runtime):
        """Test CitationValidationStep with valid citations."""
        step = CitationValidationStep()
        
        state: State = {
            "final_answer": "Python is a programming language [1]. It has many features [2].",
            "citations": ["https://example.com/1", "https://example.com/2"]
        }
        
        result = step.run(state, mock_runtime)
        
        assert "validation_passed" in result
        assert "validation_errors" in result
        assert result["validation_passed"] is True
        assert len(result["validation_errors"]) == 0
    
    def test_citation_validation_step_fail_no_citations(self, mock_runtime):
        """Test CitationValidationStep with missing citations."""
        step = CitationValidationStep()
        
        state: State = {
            "final_answer": "Python is a programming language. It has many features.",
            "citations": ["https://example.com/1"]
        }
        
        result = step.run(state, mock_runtime)
        
        assert result["validation_passed"] is False
        assert len(result["validation_errors"]) > 0
    
    def test_citation_validation_step_fail_invalid_refs(self, mock_runtime):
        """Test CitationValidationStep with invalid citation references."""
        step = CitationValidationStep()
        
        state: State = {
            "final_answer": "Python is a programming language [5].",
            "citations": ["https://example.com/1"]
        }
        
        result = step.run(state, mock_runtime)
        
        assert result["validation_passed"] is False
        assert any("Invalid citation reference" in err for err in result["validation_errors"])
    
    def test_complete_workflow(self, mock_runtime, mock_search_tool, mock_fetch_tool):
        """Test complete search_qa_strict_citations workflow execution."""
        # Create steps
        search_step = WebSearchStep(search_tool=mock_search_tool)
        fetch_step = FetchUrlStep(fetch_tool=mock_fetch_tool)
        summarize_step = SummarizeStep()
        answer_step = AnswerGenerateStep()
        validate_step = CitationValidationStep()
        
        # Create step registry
        step_registry = {
            "web_search": search_step,
            "fetch_url": fetch_step,
            "summarize": summarize_step,
            "answer_generate": answer_step,
            "citation_validation": validate_step
        }
        
        # Get workflow definition
        workflow = create_search_qa_strict_citations_workflow()
        
        # Create executor
        tracer = SimpleTracer()
        executor = GraphExecutor(step_registry, tracer=tracer)
        
        # Initial state
        initial_state: State = {
            "question": "What is Python?"
        }
        
        # Execute
        final_state = executor.execute(workflow.graph, initial_state, mock_runtime)
        
        # Verify output
        assert "final_answer" in final_state
        assert "citations" in final_state
        assert "validation_passed" in final_state
        assert "validation_errors" in final_state
        
        # Verify traces (should be 5 steps)
        traces = tracer.get_traces()
        assert len(traces) == 5
        assert all(t["success"] for t in traces)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
