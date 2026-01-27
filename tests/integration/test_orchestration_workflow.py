"""
Integration test for Phase B orchestration with refactored workflow steps.

This test demonstrates the new Step API and Graph Executor working together
to execute a simple search-based workflow.
"""

import pytest
from unittest.mock import Mock, MagicMock

from mm_orch.orchestration import (
    State,
    GraphExecutor,
    GraphNode,
    SimpleTracer,
    WebSearchStep,
    FetchUrlStep,
    SummarizeStep,
    AnswerGenerateStep
)


@pytest.fixture
def mock_runtime():
    """Create a mock runtime with model manager."""
    runtime = Mock()
    runtime.model_manager = Mock()
    runtime.model_manager.infer = Mock(return_value="Mocked model output")
    return runtime


@pytest.fixture
def mock_search_tool():
    """Create a mock search tool."""
    from mm_orch.tools.web_search import SearchResult
    
    tool = Mock()
    tool.search = Mock(return_value=[
        SearchResult(
            title="Test Result 1",
            url="https://example.com/1",
            snippet="This is a test snippet about Python programming."
        ),
        SearchResult(
            title="Test Result 2",
            url="https://example.com/2",
            snippet="Another test snippet about Python features."
        )
    ])
    return tool


@pytest.fixture
def mock_fetch_tool():
    """Create a mock fetch tool."""
    from mm_orch.tools.fetch_url import FetchedContent
    
    tool = Mock()
    tool.fetch_multiple = Mock(return_value=[
        FetchedContent(
            url="https://example.com/1",
            content="Python is a high-level programming language. " * 20,
            title="Test Result 1",
            success=True
        ),
        FetchedContent(
            url="https://example.com/2",
            content="Python has many features including dynamic typing. " * 20,
            title="Test Result 2",
            success=True
        )
    ])
    return tool


def test_search_qa_workflow_with_graph_executor(mock_runtime, mock_search_tool, mock_fetch_tool):
    """
    Test a complete search QA workflow using the new orchestration system.
    
    This demonstrates:
    1. Creating steps with the new Step API
    2. Building a workflow graph
    3. Executing with GraphExecutor
    4. Tracing execution
    """
    # Create steps
    search_step = WebSearchStep(search_tool=mock_search_tool, max_results=5)
    fetch_step = FetchUrlStep(fetch_tool=mock_fetch_tool, max_content_length=500)
    summarize_step = SummarizeStep(model_name="t5-small", max_summary_length=200)
    answer_step = AnswerGenerateStep(model_name="gpt2", max_context_length=1000)
    
    # Create step registry
    step_registry = {
        "search": search_step,
        "fetch": fetch_step,
        "summarize": summarize_step,
        "answer": answer_step
    }
    
    # Build workflow graph (linear chain)
    graph = {
        "start": GraphNode(step_name="search", next_nodes=["fetch_node"]),
        "fetch_node": GraphNode(step_name="fetch", next_nodes=["summarize_node"]),
        "summarize_node": GraphNode(step_name="summarize", next_nodes=["answer_node"]),
        "answer_node": GraphNode(step_name="answer", next_nodes=["end"]),
        "end": GraphNode(step_name="answer", next_nodes=[])
    }
    
    # Create tracer
    tracer = SimpleTracer()
    
    # Create executor
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    # Initial state
    initial_state: State = {
        "question": "What is Python?",
        "meta": {"mode": "default"}
    }
    
    # Execute workflow
    final_state = executor.execute(graph, initial_state, runtime=mock_runtime)
    
    # Verify final state has all expected fields
    assert "question" in final_state
    assert "search_results" in final_state
    assert "docs" in final_state
    assert "summaries" in final_state
    assert "final_answer" in final_state
    assert "citations" in final_state
    
    # Verify search results
    assert len(final_state["search_results"]) == 2
    assert final_state["search_results"][0]["title"] == "Test Result 1"
    
    # Verify docs fetched
    assert len(final_state["docs"]) == 2
    assert "https://example.com/1" in final_state["docs"]
    
    # Verify summaries generated
    assert len(final_state["summaries"]) == 2
    
    # Verify answer generated
    assert final_state["final_answer"]
    assert len(final_state["citations"]) == 2
    
    # Verify traces recorded
    traces = tracer.get_traces()
    assert len(traces) == 4  # 4 steps executed
    
    # Check each step was traced
    step_names = [t["step_name"] for t in traces]
    assert "search" in step_names
    assert "fetch" in step_names
    assert "summarize" in step_names
    assert "answer" in step_names
    
    # Check all steps succeeded
    assert all(t["success"] for t in traces)
    
    # Check latencies recorded
    assert all(t["latency_ms"] >= 0 for t in traces)


def test_workflow_with_partial_failure(mock_runtime, mock_search_tool):
    """
    Test workflow behavior when some steps fail gracefully.
    
    This demonstrates the degradation strategy where fetch failures
    don't stop the workflow.
    """
    # Create a fetch tool that fails for some URLs
    from mm_orch.tools.fetch_url import FetchedContent
    
    failing_fetch_tool = Mock()
    failing_fetch_tool.fetch_multiple = Mock(return_value=[
        FetchedContent(
            url="https://example.com/1",
            content="",
            title="",
            success=False,
            error="Network timeout"
        ),
        FetchedContent(
            url="https://example.com/2",
            content="Python is great. " * 10,
            title="Test Result 2",
            success=True
        )
    ])
    
    # Create steps
    search_step = WebSearchStep(search_tool=mock_search_tool)
    fetch_step = FetchUrlStep(fetch_tool=failing_fetch_tool)
    summarize_step = SummarizeStep()
    answer_step = AnswerGenerateStep()
    
    # Create step registry
    step_registry = {
        "search": search_step,
        "fetch": fetch_step,
        "summarize": summarize_step,
        "answer": answer_step
    }
    
    # Build graph
    graph = {
        "start": GraphNode(step_name="search", next_nodes=["fetch_node"]),
        "fetch_node": GraphNode(step_name="fetch", next_nodes=["summarize_node"]),
        "summarize_node": GraphNode(step_name="summarize", next_nodes=["answer_node"]),
        "answer_node": GraphNode(step_name="answer", next_nodes=["end"]),
        "end": GraphNode(step_name="answer", next_nodes=[])
    }
    
    # Execute
    tracer = SimpleTracer()
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    initial_state: State = {"question": "What is Python?"}
    final_state = executor.execute(graph, initial_state, runtime=mock_runtime)
    
    # Workflow should complete despite partial fetch failure
    assert "final_answer" in final_state
    
    # Should have only 1 doc (the successful one)
    assert len(final_state["docs"]) == 1
    assert "https://example.com/2" in final_state["docs"]
    
    # All steps should still succeed (graceful degradation)
    traces = tracer.get_traces()
    assert all(t["success"] for t in traces)


def test_workflow_state_preservation():
    """
    Test that State fields are preserved throughout workflow execution.
    
    This verifies Property 2: State Field Preservation.
    """
    from mm_orch.orchestration.base_step import BaseStep
    from typing import Dict, Any
    
    class TestStep(BaseStep):
        name = "test_step"
        input_keys = ["question"]
        output_keys = ["result"]
        
        def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
            return {"result": "processed"}
    
    step = TestStep()
    step_registry = {"test": step}
    
    graph = {
        "start": GraphNode(step_name="test", next_nodes=["end"]),
        "end": GraphNode(step_name="test", next_nodes=[])
    }
    
    executor = GraphExecutor(step_registry)
    
    # Initial state with extra fields
    initial_state: State = {
        "question": "test",
        "extra_field": "should be preserved",
        "meta": {"mode": "test", "user_id": "123"}
    }
    
    final_state = executor.execute(graph, initial_state, runtime=None)
    
    # Check new field added
    assert final_state["result"] == "processed"
    
    # Check original fields preserved
    assert final_state["extra_field"] == "should be preserved"
    assert final_state["meta"]["mode"] == "test"
    assert final_state["meta"]["user_id"] == "123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
