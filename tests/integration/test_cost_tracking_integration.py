"""
Integration tests for cost tracking with BaseStep and GraphExecutor.
"""

import time
import pytest
from typing import Dict, Any
from mm_orch.orchestration.base_step import BaseStep
from mm_orch.orchestration.graph_executor import GraphExecutor, GraphNode, SimpleTracer
from mm_orch.orchestration.state import State
from mm_orch.orchestration.cost_tracker import CostTracker


class MockRuntime:
    """Mock runtime for testing."""
    
    def __init__(self):
        self.model_load_count = 0
        self.model_manager = self
    
    def get_load_count(self):
        """Get current model load count."""
        return self.model_load_count
    
    def load_model(self):
        """Simulate loading a model."""
        self.model_load_count += 1


class SimpleStep(BaseStep):
    """Simple test step."""
    
    name = "simple_step"
    input_keys = ["input"]
    output_keys = ["output"]
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """Execute simple logic."""
        time.sleep(0.01)  # Simulate work
        return {"output": f"processed_{state['input']}"}


class ModelLoadingStep(BaseStep):
    """Step that loads models."""
    
    name = "model_loading_step"
    input_keys = ["input"]
    output_keys = ["output"]
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """Execute with model loading."""
        time.sleep(0.01)
        
        # Simulate loading 2 models
        if hasattr(runtime, 'load_model'):
            runtime.load_model()
            runtime.load_model()
        
        return {"output": f"model_processed_{state['input']}"}


class TestCostTrackingWithBaseStep:
    """Test cost tracking integration with BaseStep."""
    
    def test_step_without_cost_tracker(self):
        """Test that step works without cost tracker."""
        step = SimpleStep()
        runtime = MockRuntime()
        
        state: State = {"input": "test"}
        result = step.run(state, runtime)
        
        assert result["output"] == "processed_test"
        # No cost tracking should occur
        assert "step_costs" not in result.get("meta", {})
    
    def test_step_with_cost_tracker(self):
        """Test that step records costs when tracker is set."""
        step = SimpleStep()
        tracker = CostTracker()
        step.set_cost_tracker(tracker)
        
        runtime = MockRuntime()
        state: State = {"input": "test"}
        
        result = step.run(state, runtime)
        
        assert result["output"] == "processed_test"
        
        # Cost should be recorded in state
        assert "meta" in result
        assert "step_costs" in result["meta"]
        assert len(result["meta"]["step_costs"]) == 1
        
        cost = result["meta"]["step_costs"][0]
        assert cost["step_name"] == "simple_step"
        assert cost["latency_ms"] > 0
        assert cost["normalized_cost"] > 0
    
    def test_step_tracks_model_loads(self):
        """Test that step tracks model loads from runtime."""
        step = ModelLoadingStep()
        tracker = CostTracker()
        step.set_cost_tracker(tracker)
        
        runtime = MockRuntime()
        state: State = {"input": "test"}
        
        result = step.run(state, runtime)
        
        # Check cost includes model loads
        cost = result["meta"]["step_costs"][0]
        assert cost["model_loads"] == 2
        assert cost["normalized_cost"] > 0
    
    def test_step_cost_on_error(self):
        """Test that cost is tracked even when step fails."""
        class FailingStep(BaseStep):
            name = "failing_step"
            input_keys = ["input"]
            output_keys = ["output"]
            
            def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
                time.sleep(0.01)
                raise ValueError("Step failed")
        
        step = FailingStep()
        tracker = CostTracker()
        step.set_cost_tracker(tracker)
        
        runtime = MockRuntime()
        state: State = {"input": "test"}
        
        with pytest.raises(ValueError, match="Step failed"):
            step.run(state, runtime)
        
        # Cost should still be tracked
        all_costs = tracker.get_all_costs()
        assert len(all_costs) == 1
        
        cost = list(all_costs.values())[0]
        assert cost.step_name == "failing_step"
        assert cost.latency_ms > 0
    
    def test_multiple_step_executions(self):
        """Test tracking costs across multiple step executions."""
        step = SimpleStep()
        tracker = CostTracker()
        step.set_cost_tracker(tracker)
        
        runtime = MockRuntime()
        
        # Execute step multiple times
        for i in range(3):
            state: State = {"input": f"test{i}"}
            result = step.run(state, runtime)
            assert len(result["meta"]["step_costs"]) == 1
        
        # Tracker should have all costs
        summary = tracker.get_summary()
        assert summary["total_steps"] == 3
        assert summary["avg_latency_ms"] > 0


class TestCostTrackingWithGraphExecutor:
    """Test cost tracking integration with GraphExecutor."""
    
    def test_graph_executor_tracks_all_steps(self):
        """Test that graph executor tracks costs for all steps."""
        # Create steps
        step1 = SimpleStep()
        step2 = ModelLoadingStep()
        
        # Create step registry
        step_registry = {
            "simple_step": step1,
            "model_loading_step": step2
        }
        
        # Create graph (end node should not have a step)
        graph = {
            "start": GraphNode(step_name="simple_step", next_nodes=["middle"]),
            "middle": GraphNode(step_name="model_loading_step", next_nodes=[])
        }
        
        # Create executor
        tracer = SimpleTracer()
        executor = GraphExecutor(step_registry, tracer)
        
        # Execute graph
        runtime = MockRuntime()
        initial_state: State = {"input": "test"}
        
        final_state = executor.execute(graph, initial_state, runtime)
        
        # Check that costs are tracked
        assert "meta" in final_state
        assert "step_costs" in final_state["meta"]
        assert len(final_state["meta"]["step_costs"]) == 2
        
        # Check first step cost
        cost1 = final_state["meta"]["step_costs"][0]
        assert cost1["step_name"] == "simple_step"
        assert cost1["latency_ms"] > 0
        assert cost1["model_loads"] == 0
        
        # Check second step cost
        cost2 = final_state["meta"]["step_costs"][1]
        assert cost2["step_name"] == "model_loading_step"
        assert cost2["latency_ms"] > 0
        assert cost2["model_loads"] == 2
    
    def test_graph_executor_tracer_receives_costs(self):
        """Test that tracer receives cost metrics."""
        step = SimpleStep()
        step_registry = {"simple_step": step}
        
        graph = {
            "start": GraphNode(step_name="simple_step", next_nodes=[])
        }
        
        tracer = SimpleTracer()
        executor = GraphExecutor(step_registry, tracer)
        
        runtime = MockRuntime()
        initial_state: State = {"input": "test"}
        
        executor.execute(graph, initial_state, runtime)
        
        # Check tracer recorded cost metrics
        traces = tracer.get_traces()
        assert len(traces) == 1
        
        trace = traces[0]
        assert trace["step_name"] == "simple_step"
        assert trace["latency_ms"] > 0
        assert "vram_peak_mb" in trace
        assert "model_loads" in trace
    
    def test_cost_tracker_summary_after_workflow(self):
        """Test getting cost summary after workflow execution."""
        step1 = SimpleStep()
        step2 = ModelLoadingStep()
        
        step_registry = {
            "simple_step": step1,
            "model_loading_step": step2
        }
        
        graph = {
            "start": GraphNode(step_name="simple_step", next_nodes=["middle"]),
            "middle": GraphNode(step_name="model_loading_step", next_nodes=[])
        }
        
        executor = GraphExecutor(step_registry)
        
        runtime = MockRuntime()
        initial_state: State = {"input": "test"}
        
        executor.execute(graph, initial_state, runtime)
        
        # Get cost summary from executor's tracker
        summary = executor.cost_tracker.get_summary()
        
        assert summary["total_steps"] == 2
        assert summary["total_model_loads"] == 2
        assert summary["avg_model_loads"] == 1.0
        assert summary["total_cost"] > 0
    
    def test_cost_accumulation_across_workflows(self):
        """Test that costs accumulate across multiple workflow executions."""
        step = SimpleStep()
        step_registry = {"simple_step": step}
        
        graph = {
            "start": GraphNode(step_name="simple_step", next_nodes=[])
        }
        
        executor = GraphExecutor(step_registry)
        runtime = MockRuntime()
        
        # Execute workflow multiple times
        for i in range(3):
            initial_state: State = {"input": f"test{i}"}
            executor.execute(graph, initial_state, runtime)
        
        # Check accumulated costs
        summary = executor.cost_tracker.get_summary()
        assert summary["total_steps"] == 3
        assert summary["avg_latency_ms"] > 0
    
    def test_cost_tracking_with_branching(self):
        """Test cost tracking with conditional branching."""
        class ConditionalStep(BaseStep):
            name = "conditional_step"
            input_keys = ["value"]
            output_keys = ["branch"]
            
            def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
                time.sleep(0.01)
                return {"branch": "left" if state["value"] > 5 else "right"}
        
        class ProcessStep(BaseStep):
            name = "process_step"
            input_keys = ["branch"]
            output_keys = ["result"]
            
            def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
                time.sleep(0.01)
                return {"result": f"processed_{state['branch']}"}
        
        step_registry = {
            "conditional_step": ConditionalStep(),
            "process_step": ProcessStep()
        }
        
        # Simple linear graph (branching logic is in the step itself)
        graph = {
            "start": GraphNode(step_name="conditional_step", next_nodes=["process"]),
            "process": GraphNode(step_name="process_step", next_nodes=[])
        }
        
        executor = GraphExecutor(step_registry)
        runtime = MockRuntime()
        
        # Execute with value > 5
        state1: State = {"value": 10}
        result1 = executor.execute(graph, state1, runtime)
        
        assert result1["result"] == "processed_left"
        assert len(result1["meta"]["step_costs"]) == 2  # conditional + process
        
        # Clear costs
        executor.cost_tracker.clear()
        
        # Execute with value <= 5
        state2: State = {"value": 3}
        result2 = executor.execute(graph, state2, runtime)
        
        assert result2["result"] == "processed_right"
        assert len(result2["meta"]["step_costs"]) == 2  # conditional + process
