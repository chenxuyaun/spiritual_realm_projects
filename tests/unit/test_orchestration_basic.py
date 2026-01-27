"""
Basic unit tests for Phase B orchestration components.

These tests verify the core functionality of the Step API, State, BaseStep,
and GraphExecutor without requiring property-based testing.
"""

import pytest
from typing import Any, Dict

from mm_orch.orchestration import (
    Step,
    State,
    BaseStep,
    FunctionStep,
    GraphExecutor,
    GraphNode,
    SimpleTracer
)


class SimpleStep(BaseStep):
    """Simple test step that adds a field to state."""
    
    name = "simple_step"
    input_keys = ["input"]
    output_keys = ["output"]
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """Add output field based on input."""
        return {"output": f"processed: {state['input']}"}


class CounterStep(BaseStep):
    """Test step that increments a counter."""
    
    name = "counter_step"
    input_keys = []
    output_keys = ["count"]
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """Increment counter."""
        current = state.get("count", 0)
        return {"count": current + 1}


def test_step_protocol_compliance():
    """Test that BaseStep implements Step protocol."""
    step = SimpleStep()
    
    # Check protocol attributes
    assert hasattr(step, "name")
    assert hasattr(step, "input_keys")
    assert hasattr(step, "output_keys")
    assert hasattr(step, "run")
    
    # Check it's recognized as a Step
    assert isinstance(step, Step)


def test_base_step_execution():
    """Test BaseStep execution with validation."""
    step = SimpleStep()
    
    # Valid execution
    state: State = {"input": "test"}
    result = step.run(state, runtime=None)
    
    assert "output" in result
    assert result["output"] == "processed: test"
    assert "input" in result  # Original field preserved


def test_base_step_input_validation():
    """Test that BaseStep validates required input keys."""
    step = SimpleStep()
    
    # Missing required input
    state: State = {}
    
    with pytest.raises(KeyError) as exc_info:
        step.run(state, runtime=None)
    
    assert "input" in str(exc_info.value)
    assert "simple_step" in str(exc_info.value)


def test_state_field_preservation():
    """Test that State fields are preserved across step execution."""
    step = SimpleStep()
    
    state: State = {
        "input": "test",
        "other_field": "preserved",
        "meta": {"mode": "test"}
    }
    
    result = step.run(state, runtime=None)
    
    # Check new field added
    assert result["output"] == "processed: test"
    
    # Check original fields preserved
    assert result["other_field"] == "preserved"
    assert result["meta"] == {"mode": "test"}


def test_function_step():
    """Test FunctionStep wrapper for function-based steps."""
    def my_func(state: State, runtime: Any) -> State:
        return {**state, "result": state["input"].upper()}
    
    step = FunctionStep(
        name="my_step",
        input_keys=["input"],
        output_keys=["result"],
        func=my_func
    )
    
    state: State = {"input": "hello"}
    result = step.run(state, runtime=None)
    
    assert result["result"] == "HELLO"
    assert result["input"] == "hello"


def test_graph_executor_linear_chain():
    """Test GraphExecutor with a simple linear chain."""
    # Create steps
    step1 = CounterStep()
    step2 = CounterStep()
    step3 = CounterStep()
    
    # Create step registry
    step_registry = {
        "step1": step1,
        "step2": step2,
        "step3": step3
    }
    
    # Create graph (linear chain)
    graph = {
        "start": GraphNode(step_name="step1", next_nodes=["node2"]),
        "node2": GraphNode(step_name="step2", next_nodes=["node3"]),
        "node3": GraphNode(step_name="step3", next_nodes=["end"]),
        "end": GraphNode(step_name="step3", next_nodes=[])
    }
    
    # Execute
    executor = GraphExecutor(step_registry)
    initial_state: State = {"count": 0}
    
    final_state = executor.execute(graph, initial_state, runtime=None)
    
    # Should have incremented 3 times
    assert final_state["count"] == 3


def test_graph_executor_with_tracer():
    """Test that GraphExecutor records traces."""
    step = SimpleStep()
    step_registry = {"simple": step}
    
    graph = {
        "start": GraphNode(step_name="simple", next_nodes=["end"]),
        "end": GraphNode(step_name="simple", next_nodes=[])
    }
    
    tracer = SimpleTracer()
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    initial_state: State = {"input": "test"}
    final_state = executor.execute(graph, initial_state, runtime=None)
    
    # Check traces recorded
    traces = tracer.get_traces()
    assert len(traces) == 1
    assert traces[0]["step_name"] == "simple"
    assert traces[0]["success"] is True
    assert "latency_ms" in traces[0]


def test_graph_executor_error_handling():
    """Test that GraphExecutor handles step failures."""
    class FailingStep(BaseStep):
        name = "failing_step"
        input_keys = []
        output_keys = []
        
        def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
            raise ValueError("Step failed!")
    
    step = FailingStep()
    step_registry = {"failing": step}
    
    graph = {
        "start": GraphNode(step_name="failing", next_nodes=["end"]),
        "end": GraphNode(step_name="failing", next_nodes=[])
    }
    
    tracer = SimpleTracer()
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    initial_state: State = {}
    
    with pytest.raises(RuntimeError) as exc_info:
        executor.execute(graph, initial_state, runtime=None)
    
    # Check error message contains step name
    assert "failing" in str(exc_info.value)
    
    # Check trace recorded failure
    traces = tracer.get_traces()
    assert len(traces) == 1
    assert traces[0]["success"] is False
    assert "error" in traces[0]


def test_graph_validation_missing_start():
    """Test that graph validation catches missing start node."""
    step_registry = {"step1": SimpleStep()}
    
    graph = {
        "node1": GraphNode(step_name="step1", next_nodes=["end"])
    }
    
    executor = GraphExecutor(step_registry)
    
    with pytest.raises(ValueError) as exc_info:
        executor.execute(graph, {}, runtime=None)
    
    assert "start" in str(exc_info.value).lower()


def test_graph_validation_missing_step():
    """Test that graph validation catches missing steps."""
    step_registry = {"step1": SimpleStep()}
    
    graph = {
        "start": GraphNode(step_name="nonexistent", next_nodes=["end"]),
        "end": GraphNode(step_name="step1", next_nodes=[])
    }
    
    executor = GraphExecutor(step_registry)
    
    with pytest.raises(KeyError) as exc_info:
        executor.execute(graph, {}, runtime=None)
    
    assert "nonexistent" in str(exc_info.value)


def test_graph_cycle_detection():
    """Test that graph validation detects cycles."""
    step = SimpleStep()
    step_registry = {"step1": step}
    
    # Create a cycle: start -> node1 -> node2 -> node1
    graph = {
        "start": GraphNode(step_name="step1", next_nodes=["node1"]),
        "node1": GraphNode(step_name="step1", next_nodes=["node2"]),
        "node2": GraphNode(step_name="step1", next_nodes=["node1"]),
    }
    
    executor = GraphExecutor(step_registry)
    
    with pytest.raises(ValueError) as exc_info:
        executor.execute(graph, {"input": "test"}, runtime=None)
    
    assert "cycle" in str(exc_info.value).lower()


def test_simple_tracer():
    """Test SimpleTracer functionality."""
    tracer = SimpleTracer()
    
    # Start a trace
    state: State = {"test": "data"}
    trace_id = tracer.start_step("test_step", state)
    
    assert trace_id is not None
    assert "test_step" in trace_id
    
    # End the trace
    tracer.end_step(trace_id, state, success=True)
    
    # Check traces
    traces = tracer.get_traces()
    assert len(traces) == 1
    assert traces[0]["step_name"] == "test_step"
    assert traces[0]["success"] is True
    assert traces[0]["latency_ms"] >= 0


def test_simple_tracer_with_error():
    """Test SimpleTracer records errors."""
    tracer = SimpleTracer()
    
    state: State = {}
    trace_id = tracer.start_step("failing_step", state)
    
    error = ValueError("Test error")
    tracer.end_step(trace_id, state, success=False, error=error)
    
    traces = tracer.get_traces()
    assert len(traces) == 1
    assert traces[0]["success"] is False
    assert "error" in traces[0]
    assert "Test error" in traces[0]["error"]
    assert traces[0]["error_type"] == "ValueError"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
