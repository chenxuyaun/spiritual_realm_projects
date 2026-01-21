"""
Unit tests for WorkflowOrchestrator.

Tests cover:
- Workflow registration and retrieval
- Workflow execution
- Error handling
- Statistics tracking
- Consciousness integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from mm_orch.orchestrator import (
    WorkflowOrchestrator,
    create_orchestrator,
    get_orchestrator,
    reset_orchestrator,
    ExecutionContext
)
from mm_orch.schemas import (
    UserRequest,
    WorkflowResult,
    WorkflowSelection,
    WorkflowType,
    Task
)
from mm_orch.workflows.base import BaseWorkflow
from mm_orch.router import Router
from mm_orch.consciousness.core import ConsciousnessCore
from mm_orch.exceptions import ValidationError, OrchestrationError


class MockWorkflow(BaseWorkflow):
    """Mock workflow for testing."""
    
    workflow_type = WorkflowType.SEARCH_QA
    name = "MockWorkflow"
    description = "Mock workflow for testing"
    
    def __init__(self, should_succeed: bool = True, result_value: Any = "test result"):
        super().__init__()
        self.should_succeed = should_succeed
        self.result_value = result_value
        self.execute_called = False
        self.last_parameters = None
    
    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        self.execute_called = True
        self.last_parameters = parameters
        
        if self.should_succeed:
            return WorkflowResult(
                result=self.result_value,
                metadata={"mock": True},
                status="success"
            )
        else:
            return WorkflowResult(
                result=None,
                metadata={"mock": True},
                status="failed",
                error="Mock failure"
            )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return "query" in parameters
    
    def get_required_parameters(self):
        return ["query"]


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    reset_orchestrator()
    yield
    reset_orchestrator()


@pytest.fixture
def mock_router():
    """Create a mock router."""
    router = Mock(spec=Router)
    router.route.return_value = WorkflowSelection(
        workflow_type=WorkflowType.SEARCH_QA,
        confidence=0.9,
        parameters={"query": "test query"}
    )
    return router


@pytest.fixture
def mock_consciousness():
    """Create a mock consciousness core."""
    consciousness = Mock(spec=ConsciousnessCore)
    consciousness.get_strategy_suggestion.return_value = Mock(
        strategy="default",
        confidence=0.8,
        parameters={}
    )
    return consciousness


class TestWorkflowOrchestratorInit:
    """Tests for WorkflowOrchestrator initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        orch = create_orchestrator(auto_register_workflows=True)
        
        assert orch is not None
        assert len(orch.get_registered_workflows()) == 5  # All 5 workflow types
    
    def test_init_without_auto_register(self):
        """Test initialization without auto-registering workflows."""
        orch = create_orchestrator(auto_register_workflows=False)
        
        assert orch is not None
        assert len(orch.get_registered_workflows()) == 0
    
    def test_init_with_custom_router(self, mock_router):
        """Test initialization with custom router."""
        orch = create_orchestrator(
            router=mock_router,
            auto_register_workflows=False
        )
        
        assert orch.router is mock_router
    
    def test_init_with_custom_consciousness(self, mock_consciousness):
        """Test initialization with custom consciousness."""
        orch = create_orchestrator(
            consciousness=mock_consciousness,
            auto_register_workflows=False
        )
        
        assert orch.consciousness is mock_consciousness


class TestWorkflowRegistration:
    """Tests for workflow registration."""
    
    def test_register_workflow(self):
        """Test registering a workflow."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow()
        
        orch.register_workflow(mock_wf)
        
        assert WorkflowType.SEARCH_QA in orch.get_registered_workflows()
        assert orch.get_workflow(WorkflowType.SEARCH_QA) is mock_wf
    
    def test_register_invalid_workflow(self):
        """Test registering an invalid workflow raises error."""
        orch = create_orchestrator(auto_register_workflows=False)
        
        with pytest.raises(ValidationError):
            orch.register_workflow("not a workflow")
    
    def test_unregister_workflow(self):
        """Test unregistering a workflow."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow()
        
        orch.register_workflow(mock_wf)
        result = orch.unregister_workflow(WorkflowType.SEARCH_QA)
        
        assert result is True
        assert WorkflowType.SEARCH_QA not in orch.get_registered_workflows()
    
    def test_unregister_nonexistent_workflow(self):
        """Test unregistering a non-existent workflow returns False."""
        orch = create_orchestrator(auto_register_workflows=False)
        
        result = orch.unregister_workflow(WorkflowType.SEARCH_QA)
        
        assert result is False
    
    def test_get_workflow_returns_none_for_unregistered(self):
        """Test get_workflow returns None for unregistered workflow."""
        orch = create_orchestrator(auto_register_workflows=False)
        
        result = orch.get_workflow(WorkflowType.SEARCH_QA)
        
        assert result is None


class TestWorkflowExecution:
    """Tests for workflow execution."""
    
    def test_execute_workflow_success(self):
        """Test successful workflow execution."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True, result_value="success result")
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        assert result.status == "success"
        assert result.result == "success result"
        assert mock_wf.execute_called
    
    def test_execute_workflow_failure(self):
        """Test failed workflow execution."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=False)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        assert result.status == "failed"
        assert result.error is not None
    
    def test_execute_unregistered_workflow(self):
        """Test executing unregistered workflow returns error."""
        orch = create_orchestrator(auto_register_workflows=False)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        assert result.status == "failed"
        assert "not registered" in result.error.lower()
    
    def test_execute_workflow_adds_metadata(self):
        """Test that execution adds metadata to result."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        assert "workflow_type" in result.metadata
        assert result.metadata["workflow_type"] == "search_qa"
        assert "execution_time" in result.metadata
    
    def test_execute_workflow_sets_execution_time(self):
        """Test that execution sets execution_time."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        assert result.execution_time is not None
        assert result.execution_time >= 0


class TestProcessRequest:
    """Tests for process_request method."""
    
    def test_process_request_routes_and_executes(self, mock_router):
        """Test that process_request routes and executes workflow."""
        orch = create_orchestrator(
            router=mock_router,
            auto_register_workflows=False
        )
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        request = UserRequest(query="test query")
        result = orch.process_request(request)
        
        mock_router.route.assert_called_once_with(request)
        assert mock_wf.execute_called
    
    def test_process_request_adds_routing_metadata(self, mock_router):
        """Test that process_request adds routing info to metadata."""
        orch = create_orchestrator(
            router=mock_router,
            auto_register_workflows=False
        )
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        request = UserRequest(query="test query")
        result = orch.process_request(request)
        
        assert "routing" in result.metadata
        assert result.metadata["routing"]["workflow_type"] == "search_qa"
        assert result.metadata["routing"]["confidence"] == 0.9


class TestConsciousnessIntegration:
    """Tests for consciousness integration."""
    
    def test_notifies_consciousness_on_task_start(self, mock_consciousness):
        """Test that consciousness is notified on task start."""
        orch = create_orchestrator(
            consciousness=mock_consciousness,
            auto_register_workflows=False
        )
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        # Should have called update_state at least twice (start and complete)
        assert mock_consciousness.update_state.call_count >= 2
    
    def test_gets_strategy_suggestion(self, mock_consciousness):
        """Test that strategy suggestion is requested."""
        orch = create_orchestrator(
            consciousness=mock_consciousness,
            auto_register_workflows=False
        )
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        mock_consciousness.get_strategy_suggestion.assert_called_once()


class TestStatistics:
    """Tests for statistics tracking."""
    
    def test_initial_statistics(self):
        """Test initial statistics are zero."""
        orch = create_orchestrator(auto_register_workflows=False)
        
        stats = orch.get_statistics()
        
        assert stats["execution_count"] == 0
        assert stats["success_count"] == 0
        assert stats["failure_count"] == 0
    
    def test_statistics_after_success(self):
        """Test statistics after successful execution."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test"}
        )
        
        stats = orch.get_statistics()
        
        assert stats["execution_count"] == 1
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 0
    
    def test_statistics_after_failure(self):
        """Test statistics after failed execution."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=False)
        orch.register_workflow(mock_wf)
        
        orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test"}
        )
        
        stats = orch.get_statistics()
        
        assert stats["execution_count"] == 1
        assert stats["failure_count"] == 1
    
    def test_reset_statistics(self):
        """Test resetting statistics."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test"}
        )
        
        orch.reset_statistics()
        stats = orch.get_statistics()
        
        assert stats["execution_count"] == 0
    
    def test_get_workflow_metrics(self):
        """Test getting workflow metrics."""
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test"}
        )
        
        metrics = orch.get_workflow_metrics()
        
        assert "search_qa" in metrics
        assert "execution_count" in metrics["search_qa"]


class TestSingletonBehavior:
    """Tests for singleton behavior."""
    
    def test_get_orchestrator_returns_same_instance(self):
        """Test that get_orchestrator returns the same instance."""
        orch1 = get_orchestrator()
        orch2 = get_orchestrator()
        
        assert orch1 is orch2
    
    def test_reset_orchestrator_clears_singleton(self):
        """Test that reset_orchestrator clears the singleton."""
        orch1 = get_orchestrator()
        reset_orchestrator()
        orch2 = get_orchestrator()
        
        assert orch1 is not orch2


class TestExecutionContext:
    """Tests for ExecutionContext dataclass."""
    
    def test_execution_time_property(self):
        """Test execution_time property calculation."""
        import time
        
        request = UserRequest(query="test")
        ctx = ExecutionContext(request=request)
        
        time.sleep(0.01)  # Small delay
        
        assert ctx.execution_time > 0
    
    def test_execution_time_with_end_time(self):
        """Test execution_time when end_time is set."""
        import time
        
        request = UserRequest(query="test")
        start = time.time()
        ctx = ExecutionContext(request=request, start_time=start)
        ctx.end_time = start + 1.0  # 1 second later
        
        assert ctx.execution_time == pytest.approx(1.0, abs=0.01)
