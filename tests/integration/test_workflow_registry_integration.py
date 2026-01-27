"""
Integration tests for Workflow Registry with Graph Executor.

Tests the integration between WorkflowRegistry and GraphExecutor
to ensure workflows can be registered and executed.
"""

import pytest
from mm_orch.orchestration.graph_executor import GraphExecutor, GraphNode, SimpleTracer
from mm_orch.orchestration.base_step import BaseStep
from mm_orch.orchestration.state import State
from mm_orch.registries.workflow_registry import (
    WorkflowRegistry,
    WorkflowDefinition,
    reset_workflow_registry
)
from mm_orch.registries.workflow_definitions import (
    create_search_qa_workflow,
    register_default_workflows
)


class CounterStep(BaseStep):
    """Step that increments a counter in state."""
    
    def __init__(self, name: str, output_key: str):
        self.name = name
        self.input_keys = []
        self.output_keys = [output_key]
        self.output_key = output_key
    
    def execute(self, state: State, runtime) -> dict:
        current = state.get(self.output_key, 0)
        return {self.output_key: current + 1}


class MockRuntime:
    """Mock runtime for testing."""
    pass


class TestWorkflowRegistryIntegration:
    """Integration tests for Workflow Registry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_workflow_registry()
        
        # Create step registry
        self.step_registry = {
            "step1": CounterStep("step1", "count1"),
            "step2": CounterStep("step2", "count2"),
            "step3": CounterStep("step3", "count3"),
            "end": CounterStep("end", "done")
        }
        
        # Create workflow registry
        self.workflow_registry = WorkflowRegistry(self.step_registry)
        
        # Create tracer
        self.tracer = SimpleTracer()
        
        # Create graph executor
        self.executor = GraphExecutor(self.step_registry, self.tracer)
    
    def test_register_and_execute_workflow(self):
        """Test registering a workflow and executing it with GraphExecutor."""
        # Create workflow
        workflow = WorkflowDefinition(
            name="counter_workflow",
            description="Workflow that counts",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["node2"]),
                "node2": GraphNode(step_name="step2", next_nodes=["node3"]),
                "node3": GraphNode(step_name="step3", next_nodes=["end"]),
                # Note: "end" is a special node name that terminates execution
                # It doesn't execute a step, just signals completion
            }
        )
        
        # Register workflow
        self.workflow_registry.register(workflow)
        
        # Get workflow and execute
        registered_workflow = self.workflow_registry.get("counter_workflow")
        
        initial_state: State = {"question": "test"}
        runtime = MockRuntime()
        
        final_state = self.executor.execute(
            registered_workflow.graph,
            initial_state,
            runtime
        )
        
        # Verify execution (only 3 steps, "end" is not executed)
        assert final_state["count1"] == 1
        assert final_state["count2"] == 1
        assert final_state["count3"] == 1
        
        # Verify traces (only 3 steps traced)
        traces = self.tracer.get_traces()
        assert len(traces) == 3  # 3 steps executed (end is not a step)
        assert all(t["success"] for t in traces)
    
    def test_workflow_validation_prevents_invalid_execution(self):
        """Test that workflow validation prevents invalid workflows from being registered."""
        # Try to create workflow with non-existent step
        workflow = WorkflowDefinition(
            name="invalid_workflow",
            description="Invalid workflow",
            graph={
                "start": GraphNode(step_name="nonexistent", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        # Should fail validation
        with pytest.raises(KeyError, match="non-existent step"):
            self.workflow_registry.register(workflow)
        
        # Verify workflow was not registered
        assert not self.workflow_registry.has("invalid_workflow")
    
    def test_multiple_workflows_in_registry(self):
        """Test registering and executing multiple workflows."""
        # Create two workflows
        workflow1 = WorkflowDefinition(
            name="workflow1",
            description="First workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        workflow2 = WorkflowDefinition(
            name="workflow2",
            description="Second workflow",
            graph={
                "start": GraphNode(step_name="step2", next_nodes=["node2"]),
                "node2": GraphNode(step_name="step3", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        # Register both
        self.workflow_registry.register(workflow1)
        self.workflow_registry.register(workflow2)
        
        # Execute workflow1
        wf1 = self.workflow_registry.get("workflow1")
        state1 = self.executor.execute(wf1.graph, {}, MockRuntime())
        
        assert state1["count1"] == 1
        assert "count2" not in state1
        
        # Execute workflow2
        self.tracer.clear()
        wf2 = self.workflow_registry.get("workflow2")
        state2 = self.executor.execute(wf2.graph, {}, MockRuntime())
        
        assert "count1" not in state2
        assert state2["count2"] == 1
        assert state2["count3"] == 1


class TestDefaultWorkflowRegistration:
    """Tests for registering default workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_workflow_registry()
        
        # Create minimal step registry with required steps
        from mm_orch.orchestration.workflow_steps import (
            WebSearchStep,
            FetchUrlStep,
            SummarizeStep,
            AnswerGenerateStep
        )
        
        self.step_registry = {
            "web_search": WebSearchStep(),
            "fetch_url": FetchUrlStep(),
            "summarize": SummarizeStep(),
            "answer_generate": AnswerGenerateStep(),
            "end": CounterStep("end", "done")
        }
        
        self.workflow_registry = WorkflowRegistry(self.step_registry)
    
    def test_create_search_qa_workflow(self):
        """Test creating search_qa workflow definition."""
        workflow = create_search_qa_workflow()
        
        assert workflow.name == "search_qa"
        assert "search" in workflow.required_capabilities
        assert "qa" in workflow.tags
        assert len(workflow.graph) == 4  # start, fetch, summarize, answer (end is implicit)
    
    def test_register_search_qa_workflow(self):
        """Test registering search_qa workflow."""
        workflow = create_search_qa_workflow()
        
        # Should register successfully
        self.workflow_registry.register(workflow)
        
        assert self.workflow_registry.has("search_qa")
        
        # Verify workflow structure
        registered = self.workflow_registry.get("search_qa")
        assert registered.name == "search_qa"
        assert "start" in registered.graph
        assert registered.graph["start"].step_name == "web_search"
    
    def test_register_default_workflows_partial(self):
        """Test that register_default_workflows handles missing steps gracefully."""
        # This will try to register all workflows, but some may fail
        # due to missing steps (e.g., lesson_pack steps)
        register_default_workflows(self.workflow_registry)
        
        # At least search_qa should be registered
        assert self.workflow_registry.has("search_qa")
        
        # Other workflows may or may not be registered depending on available steps
        all_workflows = self.workflow_registry.list_all()
        assert len(all_workflows) >= 1


class TestWorkflowMetadata:
    """Tests for workflow metadata usage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.step_registry = {
            "step1": CounterStep("step1", "count"),
            "end": CounterStep("end", "done")
        }
        self.workflow_registry = WorkflowRegistry(self.step_registry)
    
    def test_workflow_with_metadata(self):
        """Test workflow with metadata."""
        workflow = WorkflowDefinition(
            name="test",
            description="Test workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            metadata={
                "typical_latency_ms": 1000,
                "typical_vram_mb": 500,
                "complexity": "low"
            }
        )
        
        self.workflow_registry.register(workflow)
        
        # Retrieve and check metadata
        registered = self.workflow_registry.get("test")
        assert registered.metadata["typical_latency_ms"] == 1000
        assert registered.metadata["typical_vram_mb"] == 500
        assert registered.metadata["complexity"] == "low"
    
    def test_query_by_capability(self):
        """Test querying workflows by capability."""
        # Register workflows with different capabilities
        wf1 = WorkflowDefinition(
            name="wf1",
            description="Workflow 1",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            required_capabilities=["search", "generate"]
        )
        
        wf2 = WorkflowDefinition(
            name="wf2",
            description="Workflow 2",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            required_capabilities=["generate", "summarize"]
        )
        
        self.workflow_registry.register(wf1)
        self.workflow_registry.register(wf2)
        
        # Query by capability
        search_workflows = self.workflow_registry.get_by_capability("search")
        generate_workflows = self.workflow_registry.get_by_capability("generate")
        
        assert len(search_workflows) == 1
        assert search_workflows[0].name == "wf1"
        
        assert len(generate_workflows) == 2
        assert set(w.name for w in generate_workflows) == {"wf1", "wf2"}
    
    def test_query_by_tag(self):
        """Test querying workflows by tag."""
        # Register workflows with different tags
        wf1 = WorkflowDefinition(
            name="wf1",
            description="Workflow 1",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            tags=["fast", "qa"]
        )
        
        wf2 = WorkflowDefinition(
            name="wf2",
            description="Workflow 2",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            tags=["slow", "comprehensive"]
        )
        
        self.workflow_registry.register(wf1)
        self.workflow_registry.register(wf2)
        
        # Query by tag
        fast_workflows = self.workflow_registry.get_by_tag("fast")
        qa_workflows = self.workflow_registry.get_by_tag("qa")
        slow_workflows = self.workflow_registry.get_by_tag("slow")
        
        assert len(fast_workflows) == 1
        assert fast_workflows[0].name == "wf1"
        
        assert len(qa_workflows) == 1
        assert qa_workflows[0].name == "wf1"
        
        assert len(slow_workflows) == 1
        assert slow_workflows[0].name == "wf2"
