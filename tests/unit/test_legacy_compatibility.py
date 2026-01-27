"""
Unit tests for legacy workflow compatibility layer.

Tests the adapters and helpers that enable Phase A workflows
to work in Phase B environment.
"""

import pytest
from typing import Any, Dict, List
from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.orchestration.legacy_adapter import (
    LegacyWorkflowAdapter,
    create_legacy_workflow_step,
    register_legacy_workflows
)
from mm_orch.orchestration.compatibility import (
    LegacyRuntime,
    execute_legacy_workflow,
    convert_parameters_to_state,
    convert_state_to_parameters,
    convert_workflow_result_to_state,
    LegacyWorkflowExecutor
)
from mm_orch.orchestration.state import State


# Mock workflow for testing
class MockWorkflow(BaseWorkflow):
    """Mock workflow for testing compatibility."""
    
    workflow_type = WorkflowType.SEARCH_QA
    name = "MockWorkflow"
    description = "Mock workflow for testing"
    
    def __init__(self):
        super().__init__()
        self.model_manager = None
        self.executed = False
        self.last_parameters = None
    
    def get_required_parameters(self) -> List[str]:
        return ["query"]
    
    def get_optional_parameters(self) -> Dict[str, Any]:
        return {"max_results": 5, "language": "en"}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        self._validate_required_parameters(parameters)
        return True
    
    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        self.executed = True
        self.last_parameters = parameters
        
        query = parameters["query"]
        answer = f"Mock answer for: {query}"
        
        return WorkflowResult(
            result=answer,
            metadata={
                "workflow": self.name,
                "query": query,
                "sources": [
                    {"url": "https://example.com", "title": "Example"}
                ]
            },
            status="success",
            execution_time=0.1
        )


class TestLegacyWorkflowAdapter:
    """Test suite for LegacyWorkflowAdapter."""
    
    def test_adapter_creation(self):
        """Test creating an adapter from a workflow."""
        workflow = MockWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        assert adapter.name == "MockWorkflow"
        # Adapter auto-maps "question" to "query" for workflows that need it
        assert "question" in adapter.input_keys
        assert "final_answer" in adapter.output_keys
        assert "workflow_result" in adapter.output_keys
    
    def test_adapter_execution(self):
        """Test executing workflow through adapter."""
        workflow = MockWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        state: State = {
            "question": "What is Python?",  # Use "question" which maps to "query"
            "meta": {}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        assert workflow.executed
        assert "final_answer" in result_state
        assert "Mock answer for: What is Python?" in result_state["final_answer"]
        assert result_state["meta"]["workflow_status"] == "success"
    
    def test_adapter_parameter_mapping(self):
        """Test parameter mapping from state to workflow."""
        workflow = MockWorkflow()
        
        # Map "question" in state to "query" in workflow
        adapter = LegacyWorkflowAdapter(workflow, parameter_mapping={"question": "query"})
        
        state: State = {
            "question": "What is Python?",
            "meta": {}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        assert workflow.executed
        assert workflow.last_parameters["query"] == "What is Python?"
        assert "final_answer" in result_state
    
    def test_adapter_runtime_injection(self):
        """Test that runtime dependencies are injected into workflow."""
        workflow = MockWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        # Create mock model manager
        class MockModelManager:
            pass
        
        model_manager = MockModelManager()
        runtime = LegacyRuntime(model_manager=model_manager)
        
        state: State = {
            "question": "Test query",  # Use "question"
            "meta": {}
        }
        
        result_state = adapter.run(state, runtime)
        
        # Verify model manager was injected
        assert workflow.model_manager is model_manager
    
    def test_adapter_error_handling(self):
        """Test adapter handles workflow errors gracefully."""
        class FailingWorkflow(MockWorkflow):
            def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
                raise ValueError("Workflow failed")
        
        workflow = FailingWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        state: State = {
            "question": "Test query",  # Use "question"
            "meta": {}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        assert result_state["meta"]["workflow_status"] == "failed"
        assert "Workflow failed" in result_state["meta"]["workflow_error"]
    
    def test_adapter_preserves_state_fields(self):
        """Test that adapter preserves existing state fields."""
        workflow = MockWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        state: State = {
            "question": "Test query",  # Use "question"
            "existing_field": "should be preserved",
            "meta": {"existing_meta": "also preserved"}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        assert result_state["existing_field"] == "should be preserved"
        assert result_state["meta"]["existing_meta"] == "also preserved"


class TestCompatibilityHelpers:
    """Test suite for compatibility helper functions."""
    
    def test_convert_parameters_to_state(self):
        """Test converting parameters to state."""
        parameters = {
            "query": "What is Python?",
            "max_results": 5,
            "language": "en"
        }
        
        state = convert_parameters_to_state(parameters, "search_qa")
        
        assert state["question"] == "What is Python?"
        assert state["meta"]["max_results"] == 5
        assert state["meta"]["language"] == "en"
        assert state["meta"]["workflow_type"] == "search_qa"
    
    def test_convert_state_to_parameters(self):
        """Test converting state to parameters."""
        state: State = {
            "question": "What is Python?",
            "meta": {
                "max_results": 5,
                "language": "en"
            }
        }
        
        parameters = convert_state_to_parameters(state)
        
        assert parameters["query"] == "What is Python?"
        assert parameters["question"] == "What is Python?"
        assert parameters["max_results"] == 5
        assert parameters["language"] == "en"
    
    def test_convert_workflow_result_to_state(self):
        """Test converting workflow result to state."""
        result = WorkflowResult(
            result="This is the answer",
            metadata={
                "workflow": "SearchQA",
                "sources": [
                    {"url": "https://example.com", "title": "Example"}
                ]
            },
            status="success",
            execution_time=0.5
        )
        
        state = convert_workflow_result_to_state(result)
        
        assert state["final_answer"] == "This is the answer"
        assert state["meta"]["workflow_status"] == "success"
        assert state["meta"]["workflow_execution_time"] == 0.5
        assert "https://example.com" in state["citations"]
    
    def test_execute_legacy_workflow(self):
        """Test executing legacy workflow with state."""
        workflow = MockWorkflow()
        
        state: State = {
            "question": "What is Python?",  # Use "question"
            "meta": {}
        }
        
        result_state = execute_legacy_workflow(workflow, state)
        
        assert workflow.executed
        assert "final_answer" in result_state
        assert result_state["meta"]["workflow_status"] == "success"


class TestLegacyWorkflowExecutor:
    """Test suite for LegacyWorkflowExecutor."""
    
    def test_executor_with_parameters(self):
        """Test executor with Phase A parameters."""
        executor = LegacyWorkflowExecutor()
        workflow = MockWorkflow()
        
        parameters = {"query": "What is Python?"}
        
        result_state = executor.execute(workflow, parameters=parameters)
        
        assert workflow.executed
        assert "final_answer" in result_state
        assert result_state["meta"]["workflow_status"] == "success"
    
    def test_executor_with_state(self):
        """Test executor with Phase B state."""
        executor = LegacyWorkflowExecutor()
        workflow = MockWorkflow()
        
        state: State = {
            "question": "What is Python?",  # Use "question"
            "meta": {}
        }
        
        result_state = executor.execute(workflow, state=state)
        
        assert workflow.executed
        assert "final_answer" in result_state
    
    def test_executor_with_result(self):
        """Test executor returning Phase A WorkflowResult."""
        executor = LegacyWorkflowExecutor()
        workflow = MockWorkflow()
        
        parameters = {"query": "What is Python?"}
        
        result = executor.execute_with_result(workflow, parameters)
        
        assert isinstance(result, WorkflowResult)
        assert result.status == "success"
        assert "Mock answer" in result.result
    
    def test_executor_requires_input(self):
        """Test executor raises error without parameters or state."""
        executor = LegacyWorkflowExecutor()
        workflow = MockWorkflow()
        
        with pytest.raises(ValueError, match="Must provide either parameters or state"):
            executor.execute(workflow)


class TestWorkflowRegistration:
    """Test suite for workflow registration helpers."""
    
    def test_create_legacy_workflow_step(self):
        """Test creating a step from workflow."""
        workflow = MockWorkflow()
        step = create_legacy_workflow_step(workflow)
        
        assert step.name == "MockWorkflow"
        assert isinstance(step, LegacyWorkflowAdapter)
    
    def test_register_legacy_workflows(self):
        """Test registering multiple workflows."""
        workflows = [
            MockWorkflow(),
            MockWorkflow()  # Register same workflow twice for testing
        ]
        
        step_registry = {}
        register_legacy_workflows(step_registry, workflows)
        
        assert "MockWorkflow" in step_registry
        assert isinstance(step_registry["MockWorkflow"], LegacyWorkflowAdapter)


class TestEndToEndCompatibility:
    """End-to-end tests for Phase A/B compatibility."""
    
    def test_full_workflow_execution_cycle(self):
        """Test complete execution cycle from parameters to result."""
        # Create workflow
        workflow = MockWorkflow()
        
        # Phase A style execution
        parameters = {"query": "What is Python?", "max_results": 3}
        result_a = workflow.run(parameters)
        
        # Phase B style execution via adapter
        adapter = LegacyWorkflowAdapter(workflow)
        state: State = {
            "question": "What is Python?",  # Use "question" which maps to "query"
            "meta": {"max_results": 3}
        }
        runtime = LegacyRuntime()
        result_b = adapter.run(state, runtime)
        
        # Both should produce equivalent results
        assert result_a.status == result_b["meta"]["workflow_status"]
        assert result_a.result in result_b["final_answer"]
    
    def test_workflow_with_all_features(self):
        """Test workflow using all compatibility features."""
        # Create executor with runtime
        class MockModelManager:
            pass
        
        runtime = LegacyRuntime(model_manager=MockModelManager())
        executor = LegacyWorkflowExecutor(runtime)
        
        # Create workflow
        workflow = MockWorkflow()
        
        # Execute with parameters
        parameters = {
            "query": "What is Python?",
            "max_results": 5,
            "language": "en"
        }
        
        result_state = executor.execute(workflow, parameters=parameters)
        
        # Verify all features work
        assert workflow.executed
        assert workflow.model_manager is runtime.model_manager
        assert result_state["final_answer"] is not None
        assert result_state["meta"]["workflow_status"] == "success"
        assert "citations" in result_state
        assert len(result_state["citations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
